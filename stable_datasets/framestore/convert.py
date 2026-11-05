"""Convert a frame store (JPEG frames in a lance binary column) into the layout partial decoding reads, losslessly
and in place: every frame re-coded with a restart marker after each MCU (codec.to_restart: pixel-identical,
checked), the frames stored as a blob column, and each frame's per-MCU segment lengths in a new seg_len column.
Every other column, and the row order, is kept.

    python -m stable_datasets.framestore.convert /data/frames.lance --census-only          # what the store holds; nothing written
    python -m stable_datasets.framestore.convert /data/frames.lance                        # census, then convert in place
    python -m stable_datasets.framestore.convert /data/frames.lance --out /data/tok.lance  # keep the source (needs ~2.1x its size)
    stable_datasets.framestore.convert_store("/data/frames.lance", memory="16GB")          # the same from Python (ConvertError)

Census (first, from the frame headers): frames by chroma layout and coding (progressive frames re-code to
baseline losslessly), and the frames the loader will have to decode whole -- gray or unusual sampling, and episodes
whose frames differ in quantization tables, layout or size (a clip's frames share one header when spliced) -- plus
those that cannot be converted at all (arithmetic-coded, 12-bit, CMYK, unparseable); the conversion refuses those
unless --keep-unsupported, which stores them unchanged with a null seg_len.

Storage: fragments move one at a time into a sibling dataset (<store>.converting): a fragment is converted and
committed there, then deleted from the source and its files removed, so the store never takes more than its size
plus ~1.1x its largest fragment. Once all have moved, the sibling replaces the source. An interrupted run resumes
where it stopped (run the same command again); until it finishes, neither directory holds the whole store.

Speed: every stage runs off the GIL in threads -- re-coding in C (libjpeg), header probes and segment scans in
numba, verification decodes in libjpeg-turbo / nvJPEG / the NVJPG engines -- and verification (both versions of a
frame decoded and compared) goes to the GPU; CPU threads join in only when the GPU is what limits the rate.

Resources (resources.py): a node runs one job at a time, so conversion takes --cpu, --memory and --gpu-memory of it
(default 80% of the CPUs the container may use, of its free memory and of the GPU's free memory; or amounts such as
--memory 16GB), at lower CPU and I/O priority (--nice). Fixed needs are reserved first -- lance's scan (from the
store's page size), the census arrays, verification decodes, the GPU decoder (measured) -- and the rest bounds the
bytes of source and re-coded batches in flight (a byte gate, from each batch's real size), so the pipeline's
memory follows the frames' size instead of a batch count. It refuses before writing anything when the fixed needs
do not fit.
"""

import argparse
import datetime
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from . import codec
from .decoders import Decoder
from .resources import Budget, ResourceError, Resources, Stage, gb, limit_malloc_arenas


class ConvertError(Exception):
    """The conversion stopped: nothing of the fragment it was on was committed; run it again to resume."""


SPLICED = ("4:2:0", "4:2:2", "4:4:0", "4:4:4")
# bytes a batch in flight holds, per byte of its source batch: the source, the re-coded frames (+9 to 33%), the
# translator's scratch, and both kept until verified; in the census, the source and its header arrays
CONVERT_COST, CENSUS_COST = 3.5, 1.5


class Profile:
    """Wall seconds per stage, summed over the threads that ran it (--profile)."""

    def __init__(self):
        self.on, self.t, self.lock = False, {}, threading.Lock()

    def add(self, k, dt):
        if self.on:
            with self.lock:
                self.t[k] = self.t.get(k, 0.0) + dt

    @contextmanager
    def __call__(self, k):
        t = time.perf_counter()
        yield
        self.add(k, time.perf_counter() - t)

    def report(self, frames, wall, threads):
        print(
            f"PROFILE {frames:,} frames, {wall:.1f} s wall, {threads} threads: stage, thread-seconds, "
            f"us per frame, share of all threads' time",
            flush=True,
        )
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            print(f"   {k:40s} {v:8.1f} s {v / frames * 1e6:8.0f} us {100 * v / (wall * threads):6.1f}%", flush=True)


PROF = Profile()


def frames_of(col):
    """An arrow (large_)binary array -> (buf, off) numpy views of its buffers, no copy; null rows are empty."""
    import pyarrow as pa

    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    _, ob, db = col.buffers()
    off = np.frombuffer(ob, np.int64 if pa.types.is_large_binary(col.type) else np.int32)
    off = off[col.offset : col.offset + len(col) + 1].astype(np.int64)
    buf = np.frombuffer(db, np.uint8) if db is not None else np.zeros(0, np.uint8)
    return buf, off


def frame_column(ds, col):
    import pyarrow as pa

    t = ds.schema.field(col).type
    if not (pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        raise ConvertError(
            f"column {col!r} is {t}: expected JPEG bytes in a binary column (is the store already "
            f"converted, or its frames in a blob column?)"
        )


def fragment_bytes(ds, frag):
    try:
        return sum(os.path.getsize(f"{ds.uri}/data/{d.path}") for d in frag.data_files())
    except OSError:
        return None


def sample_frames(frag, col, n=256):
    """The fragment's first n frames as (buf, off)."""
    return frames_of(next(iter(frag.to_batches(columns=[col], batch_size=n))).column(col))


def page_bytes(ds, col, n=8):
    """The largest page of column `col` in (up to n of) the store's data files. A scan reads and decodes whole
    pages, so its memory grows with the page size, which is up to whoever wrote the store (one page per written
    batch: 242 MB for 4096 frames of 59 kB), not with the batch size it is read in."""
    from lance.file import LanceFileReader

    frags = ds.get_fragments()
    worst = 0
    for f in [frags[i] for i in np.unique(np.linspace(0, len(frags) - 1, min(n, len(frags))).astype(int))]:
        for d in f.data_files():
            try:
                m = LanceFileReader(f"{ds.uri}/data/{d.path}").metadata()
                names = [x.name for x in m.schema]
                if col in names:
                    worst = max([worst] + [sum(b.size for b in p.buffers) for p in m.columns[names.index(col)].pages])
            except Exception:  # other storage or file versions: no estimate
                pass
    return worst


def frame_size(buf, off):
    """(mean bytes, largest pixel count) of sample frames, for sizing buffers; 1080p when there are none."""
    if len(off) < 2:
        return 300e3, 1920 * 1088
    info = codec.probe(buf, off)
    px = (info["h"].astype(np.int64) * info["w"])[info["ok"] == 1]
    return float(np.diff(off).mean()), int(px.max()) if len(px) else 1920 * 1088


def cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def committed(path):
    """Whether the lance store at `path` has a committed version (write_dataset makes the directory first)."""
    v = os.path.join(path, "_versions")
    return os.path.isdir(v) and any(f.endswith(".manifest") for f in os.listdir(v))


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.exists(p) else 0


# ---------------------------------------------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------------------------------------------
def census(ds, col, ep_col, pool, gate, sample=2048):
    """Headers of every frame -> counts by layout and coding, the frames the loader must decode whole, and an
    estimate of the size after conversion (re-coding the first `sample` frames). Batches in flight are bounded by
    `gate` (bytes) and by twice the pool's threads."""
    t0 = time.time()
    cols = [col] + ([ep_col] if ep_col else [])
    parts, est, futs = [], [0, 0, 0], deque()  # sampled frames, their bytes before and after re-coding

    def one(b):
        t = time.perf_counter()
        buf, off = frames_of(b.column(col))
        info = codec.probe(buf, off)
        PROF.add("census: probe headers (pool)", time.perf_counter() - t)
        r = {
            "layout": codec.layout(info),
            "prog": info["sof"] == codec.SOF_PROGRESSIVE,
            "q": info["qhash"],
            "h": info["h"],
            "w": info["w"],
            "n": np.diff(off),
        }
        # a copy: a zero-copy view can share its allocation with the batch's frame bytes and keep them all alive
        r["ep"] = np.array(b.column(ep_col).to_numpy(zero_copy_only=False)) if ep_col else None
        return r, (buf, off)

    def collect(item):
        fut, cost = item
        r, (buf, off) = fut.result()
        parts.append(r)
        if est[0] < sample:  # size estimate: re-code a sample
            k = min(len(off) - 1, sample - est[0])
            out, ooff, st = codec.to_restart(buf, off[: k + 1])
            ok = st == 0
            est[0] += k
            est[1] += int(np.diff(off[: k + 1])[ok].sum())
            est[2] += int(np.diff(ooff)[ok].sum())
        gate.release(cost)

    it = iter(ds.to_batches(columns=cols, batch_size=4096, batch_readahead=2, fragment_readahead=1))
    while True:
        with PROF("census: lance scan (main thread)"):
            b = next(it, None)
        if b is None:
            break
        cost = int(CENSUS_COST * b.nbytes)
        while not gate.try_acquire(cost):  # this thread releases too: collect the oldest until it fits
            collect(futs.popleft())
        futs.append((pool.submit(one, b), cost))
        while len(futs) >= 2 * pool._max_workers or (futs and futs[0][0].done()):
            collect(futs.popleft())
    while futs:
        collect(futs.popleft())
    c = {k: np.concatenate([p[k] for p in parts]) for k in ("layout", "prog", "q", "h", "w", "n")}
    c["ep"] = np.concatenate([p["ep"] for p in parts]) if ep_col else None
    c["growth"] = est[2] / max(1, est[1])
    c["s"] = time.time() - t0
    return c


def mixed_episodes(c):
    """Frames whose episode mixes quantization tables, layout or frame size (spliced clips share one header)."""
    if c["ep"] is None:
        return np.zeros(len(c["n"]), bool)
    o = np.argsort(c["ep"], kind="stable")
    ep = c["ep"][o]
    start = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    first = np.repeat(start, np.diff(np.r_[start, len(ep)]))
    lay = np.unique(c["layout"], return_inverse=True)[1]
    bad = np.zeros(len(ep), bool)
    for v in (c["q"], c["h"], c["w"], lay):
        v = v[o]
        bad |= v != v[first]
    bad_ep = np.zeros(len(start), bool)
    np.logical_or.at(bad_ep, np.searchsorted(start, np.arange(len(ep)), side="right") - 1, bad)
    mixed = np.repeat(bad_ep, np.diff(np.r_[start, len(ep)]))
    out = np.zeros(len(ep), bool)
    out[o] = mixed
    return out


def report(c, ds, frags, fbytes):
    n = len(c["n"])
    total = int(c["n"].sum())
    eps = len(np.unique(c["ep"])) if c["ep"] is not None else 0
    print(
        f"CENSUS {ds.uri}: {n:,} frames"
        + (f" in {eps:,} episodes" if eps else "")
        + f", {total / 1e9:.2f} GB of JPEG, {len(frags)} fragments ({c['s']:.0f} s)",
        flush=True,
    )
    note = {
        "4:2:0": "spliced",
        "4:2:2": "spliced (16 x 8 MCUs)",
        "4:4:0": "spliced (8 x 16 MCUs)",
        "4:4:4": "spliced (8 x 8 MCUs: ~3x the MCUs of 4:2:0 to splice)",
        "gray": "whole-frame decode",
        "other": "whole-frame decode (sampling the splicer does not handle)",
        "unsupported": "cannot be converted",
    }
    for lay in list(note):
        for prog in (False, True):
            k = int(((c["layout"] == lay) & (c["prog"] == prog)).sum())
            if k:
                coding = "" if lay == "unsupported" else ("progressive " if prog else "baseline ")
                extra = ", re-coded to baseline" if prog and lay != "unsupported" else ""
                print(f"   {coding + lay:24s} {k:>12,}  {100 * k / n:5.1f}%   {note[lay]}{extra}", flush=True)
    mixed = mixed_episodes(c) & np.isin(c["layout"], SPLICED)
    if c["ep"] is not None:
        k = int(mixed.sum())
        print(
            f"   {'in mixed episodes':24s} {k:>12,}  {100 * k / n:5.1f}%   whole-frame decode (an episode mixes "
            f"quantization tables, layout or size: {len(np.unique(c['ep'][mixed])):,} episodes)",
            flush=True,
        )
    spliced = np.isin(c["layout"], SPLICED) & ~mixed
    print(
        f"   => partial decoding can splice {100 * spliced.mean():.1f}% of frames; "
        f"{100 * (~spliced & (c['layout'] != 'unsupported')).mean():.1f}% need whole-frame decode; "
        f"{int((c['layout'] == 'unsupported').sum()):,} cannot be converted",
        flush=True,
    )
    if fbytes and None not in fbytes:
        print(
            f"   after conversion ~{total * c['growth'] / 1e9:.2f} GB (+{100 * (c['growth'] - 1):.1f}%, from a "
            f"sample); peak extra storage while converting ~{1.1 * c['growth'] * max(fbytes) / 1e9:.2f} GB "
            f"(the largest fragment)",
            flush=True,
        )
    return spliced


# ---------------------------------------------------------------------------------------------------------------
# verification: CPU threads and the GPU share one queue of chunks
# ---------------------------------------------------------------------------------------------------------------
class Job:
    def __init__(self, a, b):
        self.a, self.b = a, b  # lists of frame byte views: originals, re-coded
        self.claim = threading.Lock()
        self.done = threading.Event()
        self.bad, self.by, self.error = [], None, None

    def take(self):
        return self.claim.acquire(blocking=False)

    def finish(self, bad, by):
        self.bad, self.by = bad, by
        self.done.set()

    def fail(self, e):  # the decoder raised: the writer stops instead of waiting on this job forever
        self.error = e
        self.done.set()


class Verifier:
    """Decode both versions of every frame and compare. Chunks queue for the GPU thread; CPU threads take some
    only when the writer stalls on verification, i.e. when the GPU and not re-coding is what limits the rate."""

    def __init__(self, pool, gpu=None, cpu=True, chunk=512):
        self.pool, self.cpu, self.chunk = pool, cpu, chunk  # chunk: frame pairs per GPU decode call
        self.gpu = gpu  # the GPU decoder, or None: CPU only
        self.cpu_dec = Decoder("cpu")
        self.q = queue.Queue()
        self.lock, self.helpers = threading.Lock(), 0  # CPU helpers submitted and not yet started
        self.count = {"gpu": 0, "cpu": 0}
        if self.gpu is not None:
            threading.Thread(target=self._gpu_loop, daemon=True).start()

    def submit(self, a, b):
        job = Job(a, b)
        if self.gpu is None:
            self.pool.submit(self._cpu, job)
        else:
            self.q.put(job)
        return job

    def stalled(self, threads):
        """The writer has waited on verification for a while: the GPU is the bottleneck, so CPU threads join in.
        (While re-coding is the bottleneck the GPU keeps up, the writer never waits that long, and they never do.)"""
        if self.cpu and self.gpu is not None:
            with self.lock:
                n = min(threads, self.q.qsize()) - self.helpers
                self.helpers += max(n, 0)
            for _ in range(n):
                self.pool.submit(self._help)

    def _cpu(self, job):
        if job.take():
            try:
                self._run(job, self.cpu_dec, "cpu")
            except Exception as e:
                job.fail(e)

    def _help(self):
        try:
            job = self.q.get_nowait()  # the oldest chunk the GPU has not started
        except queue.Empty:
            job = None
        with self.lock:
            self.helpers -= 1
        if job is not None:
            self._cpu(job)

    def _gpu_loop(self):
        import torch

        stream = torch.cuda.Stream()
        while True:
            job = self.q.get()
            if job.take():
                try:
                    with torch.cuda.stream(stream):
                        self._run(job, self.gpu, "gpu")
                except Exception:  # a frame the GPU decoder rejects: this chunk on the CPU decoder instead
                    try:
                        self._run(job, self.cpu_dec, "cpu")
                    except Exception as e:
                        job.fail(e)

    def _run(self, job, dec, by):
        n = len(job.a)
        t = time.perf_counter()
        if by == "cpu":  # one pair at a time: a chunk's decoded frames would take ~1 GB per thread
            bad = []
            for i in range(n):
                x, y = dec([job.a[i], job.b[i]])
                if x.shape != y.shape or not np.array_equal(x.numpy(), y.numpy()):
                    bad.append(i)
            PROF.add("verify: decode + compare (cpu)", time.perf_counter() - t)
        else:
            bad = self._gpu_compare(job, dec)
            PROF.add(f"verify: decode + compare (gpu {dec.kind})", time.perf_counter() - t)
        self.count[by] += n
        job.finish(bad, by)

    def _gpu_compare(self, job, dec):
        bad = []
        for i in range(0, len(job.a), self.chunk):  # decoded frames of a whole batch may not fit the GPU budget
            a, b = job.a[i : i + self.chunk], job.b[i : i + self.chunk]
            bad += [i + k for k in self._gpu_compare_chunk(a, b, dec)]
        return bad

    def _gpu_compare_chunk(self, ja, jb, dec):
        import torch

        n = len(ja)
        imgs = dec(ja + jb)
        same = [i for i in range(n) if imgs[i].shape == imgs[n + i].shape]
        bad = sorted(set(range(n)) - set(same))
        if same:  # one host sync per chunk
            d = torch.stack([(imgs[i] != imgs[n + i]).any() for i in same]).cpu()
            bad += [i for i, x in zip(same, d.tolist()) if x]
        return sorted(bad)


def gpu_decoder(kind, sample, chunk):
    """The GPU verification decoder: None with --gpu none or without a GPU (verify on the CPU), else nvJPEG, warmed
    up on `sample` (a list of frames). nvJPEG decodes on the NVJPG engines where the GPU has them (H100, measured; A100
    / B200 / B300 should too, not yet measured); nvImageCodec on the same engines was slower under conversion load
    (H100: 1,612 vs 3,964 frame pairs/s) and is no longer an option."""
    if kind == "none" or not len(sample):
        return None
    import torch

    if not torch.cuda.is_available():
        return None
    d = Decoder(kind)
    # a full chunk (both versions of `chunk` frames), so its buffers reach their size before the budget charges this
    # process's memory
    d((sample * (-(-2 * chunk // len(sample))))[: 2 * chunk])
    torch.cuda.synchronize()
    return d


# ---------------------------------------------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------------------------------------------
class Converter:
    def __init__(self, a, pool, verifier, out_schema):
        self.a, self.pool, self.verifier, self.schema = a, pool, verifier, out_schema
        self.rng = np.random.default_rng(0)
        self.bad_rows = 0

    def batch(self, b, row0):
        """One source batch -> (converted batch, verification job or None)."""
        import pyarrow as pa

        a = self.a
        with PROF("convert: probe source headers"):
            buf, off = frames_of(b.column(a.column))
            info = codec.probe(buf, off)
            lay = codec.layout(info)
        with PROF("convert: to_restart"):
            out, ooff, st = codec.to_restart(buf, off)
        with PROF("convert: probe + segment lengths"):
            oinfo = codec.probe(out, ooff)
            seg, soff, segok = codec.segment_lengths(out, ooff, oinfo)
        t_asm = time.perf_counter()
        good = (st == 0) & (lay != "unsupported") & segok & (oinfo["dri"] == 1)
        n = len(off) - 1
        if good.all():
            frame = pa.LargeBinaryArray.from_buffers(
                pa.large_binary(), n, [None, pa.py_buffer(ooff), pa.py_buffer(out)]
            )
            seg_len = pa.BinaryArray.from_buffers(
                pa.binary(), n, [None, pa.py_buffer((2 * soff).astype(np.int32)), pa.py_buffer(seg)]
            )
        else:
            bad = np.flatnonzero(~good)
            if not a.keep_unsupported:
                i = bad[0]
                raise ConvertError(
                    f"row {row0 + i}: cannot convert (layout {lay[i]}, libjpeg status {st[i]}); "
                    f"{len(bad)} such rows in this batch. --keep-unsupported stores them unchanged"
                )
            self.bad_rows += len(bad)
            frame = pa.array(
                [(out[ooff[i] : ooff[i + 1]] if good[i] else buf[off[i] : off[i + 1]]).tobytes() for i in range(n)],
                pa.large_binary(),
            )
            seg_len = pa.array(
                [seg[soff[i] : soff[i + 1]].tobytes() if good[i] else None for i in range(n)], pa.binary()
            )
        cols = [
            frame if f.name == a.column else seg_len if f.name == "seg_len" else b.column(f.name) for f in self.schema
        ]
        job = None
        idx = np.flatnonzero(good)
        if a.verify < 1:
            idx = idx[self.rng.random(len(idx)) < a.verify]
        if len(idx):
            job = self.verifier.submit(
                [buf[off[i] : off[i + 1]] for i in idx], [out[ooff[i] : ooff[i + 1]] for i in idx]
            )
            job.rows = row0 + idx
        rb = pa.record_batch(cols, schema=self.schema)
        PROF.add("convert: arrow batch + verify submit", time.perf_counter() - t_asm)
        return rb, job

    def stream(self, frags, depth, gate):
        """Converted batches of all fragments in order, as (fragment id, batch), with (fragment id, None) closing each
        fragment. Conversion runs ahead across fragment boundaries, so the threads keep converting while the writer
        commits a fragment and removes it from the source: up to `depth` batches, and as many as `gate` allows
        (each holds CONVERT_COST x its source bytes from when it is read until lance has taken it)."""
        src = queue.Queue()

        def read():  # lance reads in this thread while the pool converts
            try:
                for frag in frags:
                    it = iter(frag.to_batches(batch_size=self.a.batch, batch_readahead=2))
                    while True:
                        with PROF("read: lance scan (reader thread)"):
                            b = next(it, None)
                        with PROF("read: blocked, batches in flight at the memory budget"):
                            cost = 0 if b is None else gate.acquire(CONVERT_COST * b.nbytes)
                        src.put((frag.fragment_id, b, cost))
                        if b is None:
                            break
            finally:
                src.put(None)

        threading.Thread(target=read, daemon=True).start()
        pending, row = deque(), 0
        while True:
            with PROF("write: waiting for source batches"):
                try:  # with batches pending, never wait for the reader: it may be waiting on their memory
                    item = src.get(timeout=0.01) if pending else src.get()
                except queue.Empty:
                    item = "idle"
            if item is not None and item != "idle":
                fid, b, cost = item
                pending.append((fid, None if b is None else self.pool.submit(self.batch, b, row), cost))
                row = 0 if b is None else row + len(b)
            while pending and (
                item is None
                or item == "idle"
                or len(pending) >= depth
                or pending[0][1] is None
                or pending[0][1].done()
            ):
                fid, fut, cost = pending.popleft()
                if fut is None:
                    yield fid, None
                    continue
                with PROF("write: waiting for conversion"):
                    out, job = fut.result()
                t = time.perf_counter()
                if job is not None and not job.done.wait(self.a.stall):
                    while not job.done.wait(0.01):  # stalled on verification for a while: the GPU is behind
                        self.verifier.stalled(self.pool._max_workers)
                if job is not None and job.error is not None:
                    raise ConvertError(
                        f"fragment {fid}: verification failed ({job.error!r}); nothing of this fragment was committed"
                    )
                if job is not None and job.bad:
                    raise ConvertError(
                        f"fragment {fid} rows {job.rows[job.bad][:5]}...: re-coded frame decodes "
                        f"differently ({job.by} decoder); nothing of this fragment was committed"
                    )
                PROF.add("write: waiting for verification", time.perf_counter() - t)
                yield fid, out
                gate.release(cost)  # lance has asked for the next batch: it is done with this one
            if item is None:
                return


def parser():
    ap = argparse.ArgumentParser(
        prog="python -m stable_datasets.framestore.convert", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("store")
    ap.add_argument("--out", help="write the converted store here and keep the source (default: in place)")
    ap.add_argument("--column", default="frame")
    ap.add_argument("--episode-column", default="episode_idx", help="'' if the store has none")
    ap.add_argument("--census-only", action="store_true")
    ap.add_argument("--keep-unsupported", action="store_true")
    ap.add_argument("--verify", type=float, default=1.0, help="fraction of frames decoded and compared (0: none)")
    ap.add_argument(
        "--gpu",
        choices=["nvjpeg", "none"],
        default="nvjpeg",
        help="verification decoder on the GPU: nvJPEG (on the NVJPG engines where the GPU has them); none or no "
        "CUDA device: CPU only",
    )
    ap.add_argument("--cpu-verify", type=int, default=1, help="CPU threads also verify (0: GPU only)")
    ap.add_argument(
        "--stall",
        type=float,
        default=0.25,
        help="seconds the writer waits on verification before CPU threads start verifying too",
    )
    ap.add_argument("--threads", type=int, default=0, help="worker threads (default: --cpu)")
    ap.add_argument("--batch", type=int, default=512, help="rows per converted batch")
    Resources.add_arguments(ap, cpu=0.8, memory=0.8, gpu_memory=0.8, nice=10)
    ap.add_argument("--profile", action="store_true", help="time every stage")
    return ap


def convert_store(store, **options):
    """Convert `store` (see the module docstring); options are the command line's, by name: out=..., memory="16GB",
    gpu="none", census_only=True, ... Raises ConvertError if it stops."""
    a = parser().parse_args([store])
    for k, v in options.items():
        if not hasattr(a, k):
            raise TypeError(f"convert_store() got an unknown option {k!r}")
        setattr(a, k, v)
    run(a)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        run(parser().parse_args(argv))
    except ConvertError as e:
        raise SystemExit(str(e)) from None


def run(a):
    try:
        budget = Budget("convert", Resources.from_args(a))  # before any thread starts: it lowers their priority
        limit_malloc_arenas()  # also before threads: glibc gives each its own heap, which the budget does not see
    except ValueError as e:
        raise ConvertError(str(e))
    threads = a.threads or budget.cpus
    os.environ.setdefault("LANCE_CPU_THREADS", str(threads))  # lance's own pool, before lance starts it
    import lance
    import pyarrow as pa

    PROF.on = a.profile
    if a.verify > 0:  # this tool's threads are the parallelism; torch's own pools would oversubscribe the CPUs
        import torch

        torch.set_num_threads(1)
    pool = ThreadPoolExecutor(threads)
    src_uri = os.path.abspath(a.store)
    tmp = os.path.abspath(a.out) if a.out else f"{src_uri}.converting"
    progress_path = f"{tmp}.progress.json"
    prog = json.load(open(progress_path)) if os.path.exists(progress_path) else None
    if prog and prog.get("finishing"):  # every fragment was converted: only the swap was left
        if prog["in_place"] and os.path.exists(tmp):
            if os.path.exists(src_uri):  # the emptied source, or what an interrupted removal left of it
                shutil.rmtree(src_uri)
            os.rename(tmp, src_uri)
        os.remove(progress_path)
        print(f"DONE {src_uri if prog['in_place'] else tmp} (finished an interrupted run)")
        return
    ds = lance.dataset(src_uri)
    frags = ds.get_fragments()
    ep_col = a.episode_column if a.episode_column in ds.schema.names else None

    if prog is None:
        frame_column(ds, a.column)
        if "seg_len" in ds.schema.names:
            raise ConvertError("the store already has a seg_len column")
    left = [f for f in frags if not prog or f.fragment_id not in prog["done"]]
    sbuf, soff = sample_frames(left[0], a.column) if left else (np.zeros(0, np.uint8), np.zeros(1, np.int64))
    fb, px = frame_size(sbuf, soff)
    sample = codec.frames_of(sbuf, soff)
    page, chunk, gpu = page_bytes(ds, a.column), a.batch, None
    try:  # fixed needs first; the rest bounds the batches in flight
        # lance's scan of a fragment holds ~3.3 pages (measured: 800 MB for 242 MB pages); fragments are read in turn
        budget.reserve(f"lance scan (pages of up to {page / 1e6:.0f} MB)", 4 * page)
        if prog is None:  # per-frame census arrays (~50 B) and their sorts and copies
            budget.reserve("census arrays", 150 * ds.count_rows())
        if not a.census_only and a.verify > 0:
            if a.cpu_verify:  # one decoded pair per thread
                budget.reserve("CPU verification", threads * 2 * 3 * px * 2)
            if a.gpu != "none" and cuda_available():  # both frames of a pair decoded, plus the decoder's buffers
                (chunk,) = budget.allocate(
                    [Stage("GPU verification frame pairs per decode", a.batch, 2 * 3 * px * 1.5, least=8, gpu=True)]
                )
            with budget.measure("GPU decoder", gpu=True):
                gpu = gpu_decoder(a.gpu, sample, chunk)
        budget.charge_process("this process (libraries, CUDA runtime)")  # all it holds by now
        gate = budget.gate("batches in flight", least=2 * CONVERT_COST * a.batch * fb)
    except ResourceError as e:
        raise ConvertError(str(e))
    print(
        f"   frames ~{fb / 1e3:.0f} kB, up to {px / 1e6:.1f} Mpx (sample of {len(sample)}); {threads} threads; "
        f"GPU verification in chunks of {chunk}",
        flush=True,
    )
    if prog is None:
        c = census(ds, a.column, ep_col, pool, gate)
        report(c, ds, frags, [fragment_bytes(ds, f) for f in frags])
        if a.census_only:
            return
        if (c["layout"] == "unsupported").any() and not a.keep_unsupported:
            raise ConvertError("some frames cannot be converted; --keep-unsupported stores them unchanged")
        if os.path.exists(tmp):
            raise ConvertError(f"{tmp} exists but no progress file: remove it or pass another --out")
        prog = {"source": src_uri, "out": tmp, "in_place": not a.out, "done": [], "rows": 0}
    else:
        print(f"RESUME {src_uri}: {len(prog['done'])} fragments already converted", flush=True)

    fields = [
        pa.field(f.name, pa.large_binary(), metadata={"lance-encoding:blob": "true"}) if f.name == a.column else f
        for f in ds.schema
    ]
    out_schema = pa.schema(fields + [pa.field("seg_len", pa.binary())], metadata=ds.schema.metadata)
    if os.path.exists(tmp) and not committed(tmp):  # interrupted before its first commit: holds nothing
        assert prog["rows"] == 0, f"{tmp} has no committed version but {prog['rows']} rows were converted"
        shutil.rmtree(tmp)
    if os.path.exists(tmp):
        d = lance.dataset(tmp)
        if d.count_rows() > prog["rows"]:  # committed, then interrupted before the progress file was written
            nxt = min(f.fragment_id for f in frags if f.fragment_id not in prog["done"])
            k = [f for f in frags if f.fragment_id == nxt][0].count_rows()
            assert d.count_rows() - prog["rows"] == k, "converted rows do not match the next source fragment"
            prog["done"].append(nxt)
            prog["rows"] += k

    def save():
        with open(f"{progress_path}.tmp", "w") as f:
            json.dump(prog, f)
        os.replace(f"{progress_path}.tmp", progress_path)

    def drop(fid):  # remove a converted fragment from the source, files included
        if not prog["in_place"]:
            return
        s = lance.dataset(src_uri)
        if fid in [f.fragment_id for f in s.get_fragments()]:
            s = lance.LanceDataset.commit(
                src_uri, lance.LanceOperation.Delete([], [fid], "false"), read_version=s.version
            )
            s.cleanup_old_versions(older_than=datetime.timedelta(0), delete_unverified=True)

    save()
    for fid in prog["done"]:
        drop(fid)
    todo = [f for f in frags if f.fragment_id not in prog["done"]]
    verifier = Verifier(pool, gpu, cpu=bool(a.cpu_verify), chunk=chunk)
    conv = Converter(a, pool, verifier, out_schema)
    print(
        f"CONVERT {len(todo)} fragments with {threads} threads; verify "
        f"{a.verify:.0%} of frames on "
        + ("GPU (nvjpeg)" + (" + CPU" if a.cpu_verify else "") if gpu is not None else "CPU"),
        flush=True,
    )
    t_all, rows_all, peak = time.time(), 0, 0
    stream = conv.stream(todo, depth=4 * threads, gate=gate)

    failed = []  # lance reports an error raised while it reads the batches as its own OSError: keep the original

    def part(fid):  # one fragment's batches from the shared stream
        while True:
            try:
                f, b = next(stream)
            except Exception as e:
                failed.append(e)
                raise
            assert f == fid, (f, fid)
            if b is None:
                return
            yield b

    for frag in todo:
        nb = fragment_bytes(ds, frag)
        free = shutil.disk_usage(os.path.dirname(tmp)).free
        if nb and free < 1.4 * nb:  # re-coding adds ~9% (4:2:0) to ~33% (4:4:4)
            raise ConvertError(
                f"{free / 1e9:.1f} GB free; fragment {frag.fragment_id} ({nb / 1e9:.1f} GB) needs "
                f"~{1.4 * nb / 1e9:.1f} GB. Free space and run again to resume"
            )
        t0, n0 = time.time(), dict(verifier.count)
        reader = pa.RecordBatchReader.from_batches(out_schema, part(frag.fragment_id))
        with PROF("write: lance write_dataset (main thread, total)"):
            try:
                lance.write_dataset(
                    reader,
                    tmp,
                    schema=out_schema,
                    mode="append" if committed(tmp) else "create",
                    max_rows_per_file=1 << 40,
                )
            except Exception:
                if failed:
                    raise failed[0] from None
                raise
        k = frag.count_rows()
        peak = max(peak, dir_bytes(src_uri) + dir_bytes(tmp)) if prog["in_place"] else peak
        prog["done"].append(frag.fragment_id)
        prog["rows"] += k
        save()
        drop(frag.fragment_id)
        dt = time.time() - t0
        rows_all += k
        g, cp = verifier.count["gpu"] - n0["gpu"], verifier.count["cpu"] - n0["cpu"]
        print(
            f"   fragment {frag.fragment_id}: {k:,} frames in {dt:.1f} s ({k / dt:,.0f} frames/s; verified "
            f"{g:,} on GPU, {cp:,} on CPU)",
            flush=True,
        )
    dt = time.time() - t_all
    d = lance.dataset(tmp)
    assert d.count_rows() == prog["rows"]
    after = dir_bytes(tmp)
    prog["finishing"] = True  # from here a rerun only completes the swap (it must not drop fragments by id again:
    save()  # the converted store's ids are the source's)
    if prog["in_place"]:
        assert not lance.dataset(src_uri).get_fragments(), "source fragments left"
        shutil.rmtree(src_uri)
        os.rename(tmp, src_uri)
    os.remove(progress_path)
    if a.profile:
        PROF.report(rows_all, dt, threads)
    print(
        f"DONE {prog['rows']:,} frames in {dt:.0f} s ({rows_all / max(dt, 1e-9):,.0f} frames/s) -> "
        f"{src_uri if prog['in_place'] else tmp}: {after / 1e9:.2f} GB"
        + (f"; peak on disk {peak / 1e9:.2f} GB" if peak else "")
        + (f"; {conv.bad_rows} rows stored unconverted" if conv.bad_rows else "")
        + f"; {budget.report()}; batches in flight up to {gb(gate.peak)} (estimated)",
        flush=True,
    )


if __name__ == "__main__":
    main()
