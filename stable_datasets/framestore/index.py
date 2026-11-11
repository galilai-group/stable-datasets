"""Per-node frame index of a blob-encoded frame store: every row's frame location and MCU segment lengths, built once
and memory-mapped by every loader process on the node (one copy in RAM, not one per worker).

    idx = FrameIndex.shared("/vol/frames.lance")            # build once per node (in /dev/shm if it fits), or open it
    python -m stable_datasets.framestore.index /vol/frames.lance            # the same, ahead of a run
    c = idx.lookup(rows)                                    # fid, pos, size, h, w, seg, nseg per row (numpy)

Layout (a directory, named after the store's URI, version and data files, so a rewritten store gets a new index):
    meta.json     rows, files (the store's data and side files; fid indexes this list), store uri and version
    fid.npy pos.npy size.npy h.npy w.npy seg_off.npy    per row (seg_off: rows + 1, in segments)
    seg.bin       every row's MCU segment lengths, uint16, back to back

Where: $FRAMESTORE_INDEX_DIR if set, else /dev/shm/frame_index when the index takes at most half of both the free shared
memory and the free memory (/dev/shm is RAM, and containers often mount only 64 MB of it), else on local disk under
~/.cache/framestore/index, where the page cache keeps what the loader reads hot. Building a new index of a
store removes the node's older indexes of it (a process still using one keeps its mapping).

Building scans the store's fragments in parallel (lance scan, not take: a take of the seg_len column costs ~100 us
per row) and reads only the descriptors, (h, w) and seg_len -- never the frame bytes.
"""

import fcntl
import glob
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


COLUMNS = ("fid", "pos", "size", "h", "w", "seg_off")
VERSION = 1
SHM_BASE, DISK_BASE = "/dev/shm/frame_index", os.path.expanduser("~/.cache/framestore/index")


def index_bytes(ds, sample=256):
    """A store's index size, from a sample of its rows: ~40 B of columns per row plus its MCU segment lengths."""
    n = ds.count_rows()
    if not n:
        return 0
    rows = np.unique(np.linspace(0, n - 1, min(n, sample)).astype(np.int64))
    seg = ds.take(rows, columns=["seg_len"]).column("seg_len")
    return int(n * (40 + np.mean([len(x) for x in seg.to_pylist() if x is not None] or [0])))


def index_base(nbytes):
    """Where to build an index of nbytes: $FRAMESTORE_INDEX_DIR, else /dev/shm if it fits comfortably, else local disk."""
    if os.environ.get("FRAMESTORE_INDEX_DIR"):
        return os.environ["FRAMESTORE_INDEX_DIR"]
    from .resources import gb, memory_available, shm_free

    shm = shm_free()
    if shm is not None and nbytes <= 0.5 * min(shm, memory_available()):
        return SHM_BASE
    if shm is not None:
        print(
            f"INDEX {gb(nbytes)} would take over half of the free shared memory ({gb(shm)}) or memory "
            f"({gb(memory_available())}): building it on disk under {DISK_BASE}",
            flush=True,
        )
    return DISK_BASE


def store_files(path, frags):
    """The store's frame files in a fixed order: per fragment its data file, then its side files by blob_id.
    Returns (files, data_fid per fragment, side_fid[fragment, blob_id] or -1)."""
    files, data_fid, side = [], [], {}
    for i, frag in enumerate(frags):
        (df,) = frag.data_files()
        data_fid.append(len(files))
        files.append(f"{path}/data/{df.path}")
        found = {
            int(os.path.basename(b)[:-5][::-1], 2): b  # blob_id, written bit-reversed in the file name
            for b in glob.glob(f"{path}/data/{os.path.splitext(df.path)[0]}/*.blob")
        }
        for bid in sorted(found):
            side[i, bid] = len(files)
            files.append(found[bid])
    side_fid = np.full((len(frags), max([k[1] for k in side] + [0]) + 1), -1, np.int64)
    for (i, bid), f in side.items():
        side_fid[i, bid] = f
    return files, np.array(data_fid, np.int64), side_fid


def _scan_fragment(path, i, data_fid, side_fid, out_dir):
    """One fragment's rows: small columns returned, segment lengths written to out_dir/seg.<i>.bin."""
    import lance

    frag = lance.dataset(path).get_fragments()[i]
    cols = {k: [] for k in ("fid", "pos", "size", "h", "w", "nseg")}
    with open(f"{out_dir}/seg.{i}.bin", "wb") as f:
        for b in frag.to_batches(columns=["frame", "h", "w", "seg_len"], batch_size=1 << 16):
            d = b.column("frame")
            assert hasattr(d, "field"), "the scan returned frame bytes, not blob descriptors"
            kind = d.field("kind").to_numpy(zero_copy_only=False)
            assert kind.max(initial=0) <= 1, "unexpected blob descriptor"
            fid = np.where(kind == 0, data_fid[i], side_fid[i, d.field("blob_id").to_numpy(zero_copy_only=False)])
            assert fid.min(initial=0) >= 0, "blob in a side file the store does not have"
            cols["fid"].append(fid.astype(np.int32))
            cols["pos"].append(d.field("position").to_numpy(zero_copy_only=False).astype(np.int64))
            cols["size"].append(d.field("size").to_numpy(zero_copy_only=False).astype(np.int64))
            cols["h"].append(b.column("h").to_numpy(zero_copy_only=False).astype(np.int32))
            cols["w"].append(b.column("w").to_numpy(zero_copy_only=False).astype(np.int32))
            sl = b.column("seg_len")
            _, off, buf = sl.buffers()
            o = np.frombuffer(off, np.int32)[sl.offset : sl.offset + len(sl) + 1].astype(np.int64)
            cols["nseg"].append(np.diff(o) // 2)
            f.write(memoryview(np.frombuffer(buf, np.uint8)[o[0] : o[-1]]))
    return {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}


class FrameIndex:
    def __init__(self, root):
        meta = json.load(open(f"{root}/meta.json"))
        assert meta["version"] == VERSION, f"index {root} has format {meta['version']}, expected {VERSION}"
        self.root, self.meta, self.files = root, meta, meta["files"]
        for k in COLUMNS:
            setattr(self, k, np.load(f"{root}/{k}.npy", mmap_mode="r"))
        self.seg = np.memmap(f"{root}/seg.bin", np.uint16, mode="r") if self.seg_off[-1] else np.zeros(0, np.uint16)

    def __len__(self):
        return self.meta["rows"]

    def lookup(self, rows, gather=True):
        """Per row: fid (into self.files), pos, size, h, w; seg (the rows' MCU segment lengths, concatenated) and
        nseg (segments per row). gather=False: seg is the index's whole segment-length array instead, without a
        copy, and seg_start each row's first segment in it."""
        rows = np.asarray(rows)
        a, b = self.seg_off[rows], self.seg_off[rows + 1]
        n = b - a
        out = {
            "fid": self.fid[rows].astype(np.int64),
            "pos": self.pos[rows],
            "size": self.size[rows],
            "h": self.h[rows],
            "w": self.w[rows],
            "nseg": n,
        }
        if gather:
            idx = np.repeat(a - np.cumsum(n) + n, n) + np.arange(int(n.sum()))
            out.update(seg=self.seg[idx], seg_start=np.cumsum(n) - n)
        else:
            out.update(seg=np.asarray(self.seg), seg_start=a)
        return out

    @staticmethod
    def build(path, root, workers=None):
        """Scan the store into a new index directory `root` (written to root.tmp, then renamed); `workers`
        processes scan fragments in parallel (default: the CPUs the container may use)."""
        import lance

        ds = lance.dataset(path)
        frags = ds.get_fragments()
        files, data_fid, side_fid = store_files(path, frags)
        tmp = f"{root}.tmp.{os.getpid()}"
        os.makedirs(tmp)
        try:
            FrameIndex._build(ds, path, root, tmp, frags, files, data_fid, side_fid, workers)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _build(ds, path, root, tmp, frags, files, data_fid, side_fid, workers):
        t0 = time.time()
        args = (
            [path] * len(frags),
            range(len(frags)),
            [data_fid] * len(frags),
            [side_fid] * len(frags),
            [tmp] * len(frags),
        )
        if workers is None:
            from ._native import usable_cpus

            workers = usable_cpus()
        if workers <= 1 or multiprocessing.current_process().daemon:  # daemonic processes cannot have children
            parts = list(map(_scan_fragment, *args))
        else:  # spawn, not fork: a fork after lance has started its runtime threads can crash the child
            with ProcessPoolExecutor(min(workers, len(frags)), mp_context=multiprocessing.get_context("spawn")) as ex:
                parts = list(ex.map(_scan_fragment, *args))
        for k in ("fid", "pos", "size", "h", "w"):
            np.save(f"{tmp}/{k}.npy", np.concatenate([p[k] for p in parts]))
        nseg = np.concatenate([p["nseg"] for p in parts])
        np.save(f"{tmp}/seg_off.npy", np.concatenate(([0], np.cumsum(nseg))).astype(np.int64))
        with open(f"{tmp}/seg.bin", "wb") as out:
            for i in range(len(frags)):
                with open(f"{tmp}/seg.{i}.bin", "rb") as f:
                    shutil.copyfileobj(f, out, 64 << 20)
                os.remove(f"{tmp}/seg.{i}.bin")
        json.dump(
            {
                "version": VERSION,
                "rows": int(len(nseg)),
                "files": files,
                "store": os.path.abspath(path),
                "store_version": int(ds.version),
                "build_s": round(time.time() - t0, 1),
            },
            open(f"{tmp}/meta.json", "w"),
        )
        os.rename(tmp, root)

    @classmethod
    def shared(cls, path, base=None, workers=None):
        """The node's index of store `path` under `base` (default: index_base()): opened if present, else built by
        the first process to get here while the others wait."""
        import lance

        ds = lance.dataset(path)
        # path, version and data file names: a store rewritten at the same path restarts its versions, but its
        # files are new (lance names them by UUID)
        files = ",".join(d.path for f in ds.get_fragments() for d in f.data_files())
        key = f"{os.path.abspath(path)}@{ds.version}@{hashlib.sha1(files.encode()).hexdigest()}"
        name = hashlib.sha1(key.encode()).hexdigest()[:16]
        for b in [base] if base else [os.environ.get("FRAMESTORE_INDEX_DIR"), SHM_BASE, DISK_BASE]:
            if b and os.path.exists(f"{b}/{name}/meta.json"):
                return cls(f"{b}/{name}")
        nbytes = index_bytes(ds)
        base = base or index_base(nbytes)
        os.makedirs(base, exist_ok=True)
        if shutil.disk_usage(base).free < 1.1 * nbytes:
            raise RuntimeError(
                f"the index of {path} needs ~{nbytes / 1e9:.1f} GB; {base} has "
                f"{shutil.disk_usage(base).free / 1e9:.1f} GB free (set $FRAMESTORE_INDEX_DIR elsewhere)"
            )
        root = f"{base}/{name}"
        if not os.path.exists(f"{root}/meta.json"):
            with open(f"{root}.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not os.path.exists(f"{root}/meta.json"):
                    cls.build(path, root, workers)
                    remove_stale(path, root)
        return cls(root)


def remove_stale(path, keep):
    """Remove the node's other indexes of store `path` (older versions of it), which would otherwise fill /dev/shm."""
    for b in {os.environ.get("FRAMESTORE_INDEX_DIR"), SHM_BASE, DISK_BASE} - {None}:
        for meta in glob.glob(f"{b}/*/meta.json"):
            root = os.path.dirname(meta)
            try:
                same = json.load(open(meta))["store"] == os.path.abspath(path)
            except (OSError, ValueError, KeyError):
                continue
            if same and os.path.abspath(root) != os.path.abspath(keep):
                shutil.rmtree(root, ignore_errors=True)
                if os.path.exists(f"{root}.lock"):
                    os.remove(f"{root}.lock")


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        t = time.time()
        idx = FrameIndex.shared(p)
        print(f"INDEX {p}: {len(idx)} rows at {idx.root} ({time.time() - t:.0f} s; build {idx.meta['build_s']} s)")
