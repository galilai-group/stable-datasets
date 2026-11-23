"""Per-node frame index of a blob-encoded frame store: every row's frame location and MCU segment lengths, built once
and memory-mapped by every loader process on the node (storage.record_index.RecordIndex).

    idx = FrameIndex.shared("/vol/frames.lance")            # build once per node (in /dev/shm if it fits), or open it
    python -m stable_datasets.framestore.index /vol/frames.lance            # the same, ahead of a run
    c = idx.lookup(rows)                                    # fid, pos, size, h, w, seg, nseg per row (numpy)

Columns fid, pos, size, h, w per row; ragged column seg (each row's MCU segment lengths, uint16). The index is named
after the store's URI, version and data files, so a rewritten store gets a new index.

Building scans the store's fragments in parallel (lance scan, not take: a take of the seg_len column costs ~100 us
per row) and reads only the descriptors, (h, w) and seg_len -- never the frame bytes.
"""

import hashlib
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..storage.lance_files import blob_files, blob_locations
from ..storage.record_index import RecordIndex


def index_bytes(ds, sample=256):
    """A store's index size, from a sample of its rows: ~40 B of columns per row plus its MCU segment lengths."""
    n = ds.count_rows()
    if not n:
        return 0
    rows = np.unique(np.linspace(0, n - 1, min(n, sample)).astype(np.int64))
    seg = ds.take(rows, columns=["seg_len"]).column("seg_len")
    return int(n * (40 + np.mean([len(x) for x in seg.to_pylist() if x is not None] or [0])))


def _scan_fragment(path, i, data_fid, side_fid, out_dir):
    """One fragment's rows: small columns returned, segment lengths written to out_dir/seg.<i>.bin."""
    import lance

    frag = lance.dataset(path).get_fragments()[i]
    cols = {k: [] for k in ("fid", "pos", "size", "h", "w", "nseg")}
    with open(f"{out_dir}/seg.{i}.bin", "wb") as f:
        for b in frag.to_batches(columns=["frame", "h", "w", "seg_len"], batch_size=1 << 16):
            d = b.column("frame")
            assert hasattr(d, "field"), "the scan returned frame bytes, not blob descriptors"
            fid, pos, size = blob_locations(d, np.full(len(d), i), data_fid, side_fid)
            cols["fid"].append(fid.astype(np.int32))
            cols["pos"].append(pos)
            cols["size"].append(size)
            cols["h"].append(b.column("h").to_numpy(zero_copy_only=False).astype(np.int32))
            cols["w"].append(b.column("w").to_numpy(zero_copy_only=False).astype(np.int32))
            sl = b.column("seg_len")
            _, off, buf = sl.buffers()
            o = np.frombuffer(off, np.int32)[sl.offset : sl.offset + len(sl) + 1].astype(np.int64)
            cols["nseg"].append(np.diff(o) // 2)
            f.write(memoryview(np.frombuffer(buf, np.uint8)[o[0] : o[-1]]))
    return {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}


class FrameIndex(RecordIndex):
    def lookup(self, rows, gather=True):
        """Per row: fid (into self.files), pos, size, h, w; seg (the rows' MCU segment lengths, concatenated) and
        nseg (segments per row). gather=False: seg is the index's whole segment-length array instead, without a
        copy, and seg_start each row's first segment in it."""
        out = super().lookup(rows, gather)
        out["fid"] = out["fid"].astype(np.int64)
        out["nseg"] = out.pop("n")
        return out

    @property
    def seg_off(self):
        return self.ragged_off

    @property
    def seg(self):
        return self.ragged

    @staticmethod
    def build(path, root, workers=None):
        """Scan the store into a new index directory `root`; `workers` processes scan fragments in parallel
        (default: the CPUs the container may use)."""
        import lance

        ds = lance.dataset(path)
        frags = ds.get_fragments()
        files, data_fid, side_fid = blob_files(path, frags)
        parts_dir = f"{root}.parts.{os.getpid()}"
        os.makedirs(parts_dir)
        try:
            args = ([path] * len(frags), range(len(frags)), [data_fid] * len(frags), [side_fid] * len(frags),
                    [parts_dir] * len(frags))  # fmt: skip
            if workers is None:
                from ._native import usable_cpus

                workers = usable_cpus()
            if workers <= 1 or multiprocessing.current_process().daemon:  # daemonic processes cannot have children
                parts = list(map(_scan_fragment, *args))
            else:  # spawn, not fork: a fork after lance has started its runtime threads can crash the child
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(min(workers, len(frags)), mp_context=ctx) as ex:
                    parts = list(ex.map(_scan_fragment, *args))
            nseg = np.concatenate([p["nseg"] for p in parts])
            with open(f"{parts_dir}/seg.bin", "wb") as out:
                for i in range(len(frags)):
                    with open(f"{parts_dir}/seg.{i}.bin", "rb") as f:
                        shutil.copyfileobj(f, out, 64 << 20)
            RecordIndex.write(
                root,
                files,
                {k: np.concatenate([p[k] for p in parts]) for k in ("fid", "pos", "size", "h", "w")},
                ragged=("seg", np.concatenate(([0], np.cumsum(nseg))), f"{parts_dir}/seg.bin"),
                ragged_dtype=np.dtype(np.uint16).str,
                store=os.path.abspath(path),
                store_version=int(ds.version),
            )
        finally:
            shutil.rmtree(parts_dir, ignore_errors=True)

    @classmethod
    def shared(cls, path, base=None, workers=None):
        """The node's index of store `path` under `base` (default: storage.record_index.index_base): opened if
        present, else built by the first process to get here while the others wait."""
        import lance

        ds = lance.dataset(path)
        # path, version and data file names: a store rewritten at the same path restarts its versions, but its
        # files are new (lance names them by UUID)
        files = ",".join(d.path for f in ds.get_fragments() for d in f.data_files())
        key = f"frames:{os.path.abspath(path)}@{ds.version}@{hashlib.sha1(files.encode()).hexdigest()}"
        idx = RecordIndex.shared(
            key, lambda root: cls.build(path, root, workers), index_bytes(ds), base=base, owner=os.path.abspath(path)
        )
        return cls(idx.root)


if __name__ == "__main__":
    import sys
    import time

    for p in sys.argv[1:]:
        t = time.time()
        idx = FrameIndex.shared(p)
        print(f"INDEX {p}: {len(idx)} rows at {idx.root} ({time.time() - t:.0f} s; build {idx.meta['build_s']} s)")
