"""Build a frame store from videos: each video decoded, sampled to --fps, resized to --short-edge, and its frames
encoded as JPEGs with a restart marker after every MCU as they are written (codec.encode), so the store is in the
layout partial decoding reads without a conversion pass. One episode per video.

    python -m stable_datasets.framestore.build /data/k400/train --out /data/k400.lance --short-edge 384 --fps 15
    python -m stable_datasets.framestore.build videos.txt --out /data/v.lance --label-from-parent     # one path per line
    stable_datasets.framestore.build_store(paths, "/data/k400.lance", short_edge=384, fps=15, columns={"label": labels})

Rows: episode_idx (the video's position in the input), step_idx, frame (a blob column), h, w, seg_len, and any
per-video `columns`. Episodes are written whole, in the order they finish (each one's rows contiguous, which is
all FrameStore needs).

Memory and resuming: workers send finished videos to this process, which appends them to the store in commits of
whole videos (--commit-gb); at most a few videos per worker are in flight, so memory does not grow with the
dataset. An interrupted build resumes where it stopped: run the same command again (the build's settings are kept
in the store and must match).

Decoding: decord where it is installed, resizing inside the decoder (ffmpeg's swscale); else torchcodec on the
CPU, resizing inside the decoder where its version can, else with Pillow (bilinear). Frames are sampled every
round(native fps / fps) frames. Workers: --cpu of the CPUs the container may use (resources.job_cpus, two
always left free), at lower priority (--nice).

Speed (H100 pod, 15 CPUs, Kinetics at 15 fps / 384 px short edge, q90): 2,798 frames/s with decord resizing in
the decoder; 1,633 resizing with Pillow (benchmarks/bench_build.py).
"""

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import codec
from .resources import job_cpus, limit_malloc_arenas, lower_priority
from .store import format_metadata


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
CHUNK = 32  # frames decoded and encoded at a time per worker
BUILD_KEY = b"stable_datasets.framestore.build"


def target(h, w, short_edge):
    """(h, w) resized so the short edge is short_edge (None: unchanged)."""
    if not short_edge:
        return h, w
    return (short_edge, round(w * short_edge / h)) if h <= w else (round(h * short_edge / w), short_edge)


def steps(n, native_fps, fps):
    """The frame indices kept: every round(native_fps / fps)-th (all of them without fps)."""
    every = max(1, round(native_fps / fps)) if fps and native_fps else 1
    return list(range(0, n, every))


class Decord:
    def __init__(self, path, short_edge):
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        h, w = vr[0].shape[:2]
        self.size = target(h, w, short_edge)
        if self.size != (h, w):  # resized in the decoder
            vr = VideoReader(path, ctx=cpu(0), num_threads=1, width=self.size[1], height=self.size[0])
        self.vr, self.n, self.fps = vr, len(vr), vr.get_avg_fps()

    def frames(self, idx):
        return self.vr.get_batch(idx).asnumpy()


class TorchCodec:
    def __init__(self, path, short_edge):
        from torchcodec.decoders import VideoDecoder

        kw = {"device": "cpu", "num_ffmpeg_threads": 1, "seek_mode": "exact", "dimension_order": "NHWC"}
        dec = VideoDecoder(path, **kw)
        m = dec.metadata
        self.size, self.resize = target(m.height, m.width, short_edge), None
        if self.size != (m.height, m.width):
            try:  # resized in the decoder (bilinear, antialiased), where this torchcodec can
                from torchcodec.transforms import Resize

                dec = VideoDecoder(path, transforms=[Resize(self.size)], **kw)
            except (ImportError, TypeError):
                self.resize = self.size
        self.dec, self.n, self.fps = dec, len(dec), m.average_fps

    def frames(self, idx):
        x = self.dec.get_frames_at(indices=idx).data.numpy()
        if self.resize is None:
            return x
        from PIL import Image

        h, w = self.resize
        return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in x])


DECODERS = {"decord": Decord, "torchcodec": TorchCodec}


def video_decoder(name="auto"):
    """The decoder class for name ("auto": decord if installed, else torchcodec)."""
    if name != "auto":
        return DECODERS[name]
    for name in ("decord", "torchcodec"):
        try:
            __import__(name)
            return DECODERS[name]
        except ImportError:
            pass
    raise ImportError("stable_datasets.framestore.build needs a video decoder: pip install torchcodec (or decord)")


def encode_video(task):
    """(i, path, short_edge, fps, quality, decoder, min_frames) -> (i, frames or None, error or None); frames:
    buf, off, seg (uint16 per MCU, all frames), seg_off, h, w."""
    i, path, short_edge, fps, quality, decoder, min_frames = task
    try:
        v = video_decoder(decoder)(path, short_edge)
        idx = steps(v.n, v.fps, fps)
        if len(idx) < min_frames:
            return i, None, f"{len(idx)} frames, fewer than {min_frames}"
        bufs, offs = [], [np.zeros(1, np.int64)]
        at = 0
        for c in range(0, len(idx), CHUNK):
            buf, off = codec.encode(v.frames(idx[c : c + CHUNK]), quality)
            bufs.append(buf)
            offs.append(off[1:] + at)
            at += int(off[-1])
        buf, off = np.concatenate(bufs), np.concatenate(offs)
        info = codec.probe(buf, off)
        seg, seg_off, ok = codec.segment_lengths(buf, off, info)
        if not ok.all():
            return i, None, f"{int((~ok).sum())} encoded frames without one segment per MCU"
        h, w = v.size
        return i, (buf, off, seg, seg_off, h, w), None
    except Exception as e:  # a corrupt or unreadable video: reported, the build goes on
        return i, None, f"{type(e).__name__}: {e}"


def _init_worker(nice):
    os.environ["OMP_NUM_THREADS"] = "1"  # torch (torchcodec) in a worker: one thread each
    limit_malloc_arenas()
    lower_priority(nice)


def committed(path):
    """Whether the lance store at `path` has a committed version (write_dataset makes the directory first)."""
    v = os.path.join(path, "_versions")
    return os.path.isdir(v) and any(f.endswith(".manifest") for f in os.listdir(v))


def schema_for(settings, columns):
    import pyarrow as pa

    extra = [pa.field(k, pa.array(v[:1]).type) for k, v in columns.items()]
    meta = {**format_metadata("frame"), BUILD_KEY: json.dumps(settings, sort_keys=True).encode()}
    return pa.schema(
        [
            pa.field("episode_idx", pa.int32()),
            pa.field("step_idx", pa.int32()),
            pa.field("frame", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
            pa.field("h", pa.int16()),
            pa.field("w", pa.int16()),
            pa.field("seg_len", pa.binary()),
            *extra,
        ],
        metadata=meta,
    )


def record_batch(schema, i, frames, columns):
    import pyarrow as pa

    buf, off, seg, seg_off, h, w = frames
    n = len(off) - 1
    seg = seg.astype("<u2")
    arrays = [
        pa.array(np.full(n, i, np.int32)),
        pa.array(np.arange(n, dtype=np.int32)),
        pa.LargeBinaryArray.from_buffers(pa.large_binary(), n, [None, pa.py_buffer(off), pa.py_buffer(buf)]),
        pa.array(np.full(n, h, np.int16)),
        pa.array(np.full(n, w, np.int16)),
        pa.BinaryArray.from_buffers(
            pa.binary(), n, [None, pa.py_buffer((2 * seg_off).astype(np.int32)), pa.py_buffer(seg)]
        ),
    ]
    arrays += [pa.array([columns[k][i]] * n, schema.field(k).type) for k in columns]
    return pa.record_batch(arrays, schema=schema)


def build_store(
    videos,
    out,
    *,
    short_edge=None,
    fps=None,
    quality=90,
    columns=None,
    cpu=0.8,
    decoder="auto",
    min_frames=1,
    commit_gb=1.0,
    nice=10,
    log=print,
):
    """Build the store at `out` from `videos` (paths); see the module docstring. columns: per-video values
    ({name: list as long as videos}) stored on each of the video's rows. Resumes an interrupted build. Returns
    {"videos", "frames", "bytes", "failed": [(path, why)], "seconds"}."""
    import lance
    import pyarrow as pa

    videos = [str(v) for v in videos]
    columns = dict(columns or {})
    for k, v in columns.items():
        if len(v) != len(videos):
            raise ValueError(f"column {k!r} has {len(v)} values for {len(videos)} videos")
    video_decoder(decoder)  # fail now, not in every worker
    paths = hashlib.sha1("\n".join(videos).encode()).hexdigest()[:16]  # episode_idx is a position in this list
    settings = {"short_edge": short_edge, "fps": fps, "quality": quality, "videos": len(videos), "paths": paths}
    schema = schema_for(settings, columns)

    done = set()
    if committed(out):
        ds = lance.dataset(out)
        old = json.loads((ds.schema.metadata or {}).get(BUILD_KEY, b"{}"))
        if old != settings:
            raise ValueError(f"{out} was built with {old}, not {settings}: pass another --out or the same settings")
        done = set(np.unique(ds.to_table(columns=["episode_idx"]).column(0).to_numpy()).tolist())
        log(f"RESUME {out}: {len(done)} of {len(videos)} videos already built")
    todo = [i for i in range(len(videos)) if i not in done]
    workers = min(job_cpus(cpu), max(1, len(todo)))
    log(f"BUILD {len(todo)} videos with {workers} workers -> {out}")

    from multiprocessing import get_context

    results = queue.Queue()
    slots = threading.Semaphore(2 * workers)  # videos in flight: bounds this process's memory
    stats = {"videos": 0, "frames": 0, "bytes": 0, "failed": [], "seconds": 0.0}
    t0 = time.time()
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(nice,)) as pool:

        def submit():
            for i in todo:
                slots.acquire()
                task = (i, videos[i], short_edge, fps, quality, decoder, min_frames)
                pool.apply_async(encode_video, (task,), callback=results.put, error_callback=results.put)

        threading.Thread(target=submit, daemon=True).start()
        left = len(todo)

        def next_batch():  # the next finished video's rows, or None when every video has come back
            nonlocal left
            while left:
                r = results.get()
                left -= 1
                slots.release()
                if isinstance(r, BaseException):
                    raise r
                i, frames, err = r
                if frames is None:
                    stats["failed"].append((videos[i], err))
                    log(f"skipping {videos[i]}: {err}")
                    continue
                stats["videos"] += 1
                stats["frames"] += len(frames[1]) - 1
                stats["bytes"] += int(frames[1][-1])
                return record_batch(schema, i, frames, columns)
            return None

        first = next_batch()
        while first is not None:

            def commit(b):  # whole videos up to commit_gb, then this commit ends
                nonlocal first
                size = 0
                while b is not None:
                    yield b
                    size += b.nbytes
                    if size >= commit_gb * 1e9:
                        first = next_batch()
                        return
                    b = next_batch()
                first = None

            lance.write_dataset(
                pa.RecordBatchReader.from_batches(schema, commit(first)),
                out,
                schema=schema,
                mode="append" if committed(out) else "create",
                max_rows_per_file=1 << 40,
            )
            el = time.time() - t0
            log(
                f"  {stats['videos'] + len(done)}/{len(videos)} videos, {stats['frames']:,} frames, "
                f"{stats['frames'] / el:,.0f} frames/s, {stats['bytes'] / max(stats['frames'], 1) / 1e3:.1f} KB/frame"
            )
    stats["seconds"] = time.time() - t0
    return stats


def sources(paths):
    """Video paths from directories (searched recursively), lists (.txt, one path per line) and files, sorted
    within each directory."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.suffix == ".txt":
            out += [Path(line.strip()) for line in p.read_text().splitlines() if line.strip()]
        else:
            out.append(p)
    return out


def parser():
    p = argparse.ArgumentParser(
        prog="python -m stable_datasets.framestore.build", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("videos", nargs="+", help="video files, directories (recursive) or .txt lists of paths")
    p.add_argument("--out", required=True, help="the lance store to write (resumed if it exists)")
    p.add_argument("--short-edge", type=int, help="resize so the short edge is this many pixels (default: unchanged)")
    p.add_argument("--fps", type=float, help="keep about this many frames per second (default: every frame)")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    p.add_argument("--label-from-parent", action="store_true", help="a label column: the parent directory's index")
    p.add_argument("--min-frames", type=int, default=1, help="skip videos with fewer kept frames")
    p.add_argument("--decoder", default="auto", choices=["auto", *DECODERS])
    p.add_argument("--cpu", default="0.8", help="share (0.8) or count (12) of the usable CPUs (two always left free)")
    p.add_argument("--commit-gb", type=float, default=1.0, help="append to the store every this many GB")
    p.add_argument("--nice", type=int, default=10)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    videos = sources(a.videos)
    columns = {}
    if a.label_from_parent:
        names = sorted({v.parent.name for v in videos})
        columns["label"] = [names.index(v.parent.name) for v in videos]
    cpu = float(a.cpu) if "." in a.cpu else int(a.cpu)
    s = build_store(
        videos,
        a.out,
        short_edge=a.short_edge,
        fps=a.fps,
        quality=a.quality,
        columns=columns,
        cpu=cpu,
        decoder=a.decoder,
        min_frames=a.min_frames,
        commit_gb=a.commit_gb,
        nice=a.nice,
    )
    print(
        f"DONE {s['videos']} videos, {s['frames']:,} frames, {s['bytes'] / 1e9:.2f} GB in {s['seconds']:.0f} s; "
        f"{len(s['failed'])} skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
