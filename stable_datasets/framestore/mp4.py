"""MP4 video stores: one H.264 MP4 per row (a whole video), re-encoded when the store is built so that every clip
starts on a keyframe and is decoded straight to pixels, with no pre-roll and no JPEG in between.

    store = Mp4Store("/vol/k400_mp4.lance")
    video, start = store.clips(rng, 64, frames=16)      # random clips: a video uniformly, then a keyframe start in it
    clip = store.decode(video[0], start[0], 16)         # (16, H, W, 3) uint8

Encoding (stable_datasets.framestore.build --format mp4): libx264 at --crf (default 18), a closed GOP of --keyint frames (default
4) with no B-frames, so each keyframe is an entry point that needs no earlier frame, -tune fastdecode, 4:2:0 (frames
cropped to even sizes), and the moov box first (+faststart) so a reader finds every frame without reading the rest.
The store keeps frames at the build's fps, so the clip stride is fixed then (frameskip 4 at 30 fps: --fps 7.5).

One coded size (--canvas, default 16:9 at the short edge: 384 x 682): every video is fitted into it, portrait videos
transposed, narrower ones padded on the right and wider ones scaled down to fit and padded below (padding repeats the
edge, so it costs few bytes and leaves no edge for the codec to ring on). The rows' h, w are the valid region at the
top left and `transposed` says whether to transpose it back; readers return each video in its own size and
orientation. Why: NVDEC reuses a decoder only for the same size, and Kinetics-400 has 40 sizes; with one, 16 NVDEC
processes on an H100 decode 1,245 clips/s against ~390 with the sizes mixed (benchmarks/bench_nvdec.py).

Storage, 200 held-out Kinetics-400 videos at 7.5 fps, 384 px short edge (benchmarks/gop_sizes.py), bytes against
the source mp4s: crf 18, keyframe every 4 frames 0.98x (22.0 KB/frame, 40.1 dB PSNR); every 16 0.83x; the JPEG store
at q90 1.99x.

Decoding: torchcodec, on the CPU or on the GPU's video decoders (NVDEC), in approximate seek mode (frame index ->
timestamp from the constant frame rate the build wrote). read="mmap" hands the decoder a copy of the video's bytes
(opening from memory is ~3 ms cheaper per clip than through a Python file object); read="pread" reads only what the
decoder asks for: the header and the clip's GOPs (~1/3 of a 10 s video's bytes per 16-frame clip).

Rows: episode_idx, video (a blob column), frames, fps (the kept rate), h, w (the valid region), transposed, and any
per-video columns. The schema metadata under VIDEO_KEY holds {"format", "codec", "keyint", "crf", "canvas"}.
"""

import io
import json
import os
import subprocess
import tempfile

import numpy as np


VIDEO_KEY, VIDEO_FORMAT = b"stable_datasets.framestore.video", 1


def video_metadata(keyint, crf, canvas=None):
    m = {"format": VIDEO_FORMAT, "codec": "h264", "keyint": keyint, "crf": crf, "canvas": canvas}
    return {VIDEO_KEY: json.dumps(m).encode()}


def default_canvas(short_edge):
    """16:9 at the short edge, even: (384, 682) for 384."""
    return (short_edge // 2 * 2, round(short_edge * 16 / 9) // 2 * 2) if short_edge else None


def fit(frames, canvas):
    """(n, h, w, 3) frames -> (frames on the canvas (n, H, W, 3), h, w of the valid region, transposed): portrait
    frames transposed, frames larger than the canvas scaled down to fit (antialiased bilinear), the rest of the canvas
    the region's edge repeated."""
    transposed = frames.shape[1] > frames.shape[2]
    if transposed:
        frames = frames.transpose(0, 2, 1, 3)
    H, W = canvas
    h, w = frames.shape[1:3]
    if h > H or w > W:
        import torch
        import torch.nn.functional as F

        k = min(H / h, W / w)
        h, w = max(2, int(h * k) // 2 * 2), max(2, int(w * k) // 2 * 2)
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        frames = F.interpolate(x, size=(h, w), mode="bilinear", antialias=True).permute(0, 2, 3, 1).numpy()
    return np.pad(frames, ((0, 0), (0, H - h), (0, W - w), (0, 0)), mode="edge"), h, w, transposed


def video_format(schema):
    """An MP4 store's metadata from its (arrow) schema, or None for a frame store."""
    meta = (schema.metadata or {}).get(VIDEO_KEY)
    if meta is None:
        return None
    m = json.loads(meta)
    if m["format"] > VIDEO_FORMAT:
        raise ValueError(
            f"video store format {m['format']} is newer than this stable_datasets.framestore reads: upgrade stable_datasets.framestore"
        )
    return m


def encode(chunks, h, w, fps, crf=18, keyint=4, preset="medium", gpu=True):
    """MP4 bytes of the frames in `chunks` (an iterable of (n, h, w, 3) uint8 arrays; h and w even), at `fps`: libx264,
    a closed keyframe every `keyint` frames, no B-frames, -tune fastdecode, BT.601 4:2:0, moov first. One thread (the
    builder runs one encoder per CPU). gpu: the subset stable_datasets.framestore.gpu_h264 decodes (baseline profile, one reference
    frame, one slice per macroblock row); every decoder reads it, at 0-11% more bytes."""
    extra = ["-profile:v", "baseline", "-x264-params", f"ref=1:slices={(h + 15) // 16}"] if gpu else []
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "v.mp4")  # +faststart rewrites the file, so not a pipe
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", repr(float(fps)), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-g", str(keyint), "-keyint_min", str(keyint), "-sc_threshold", "0", "-bf", "0", "-flags", "+cgop",
               "-tune", "fastdecode", "-threads", "1", "-pix_fmt", "yuv420p", "-colorspace", "smpte170m",
               "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-color_range", "tv",
               *extra, "-movflags", "+faststart", out]  # fmt: skip
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for c in chunks:
                p.stdin.write(np.ascontiguousarray(c).data)
            p.stdin.close()
        except BrokenPipeError:
            pass
        err = p.stderr.read().decode(errors="replace").strip()
        if p.wait():
            raise RuntimeError(f"ffmpeg failed: {err}")
        with open(out, "rb") as f:
            return f.read()


# slice index (stable_datasets.framestore.gpu_h264's input, built with the store): a header (frames, mb_h, mb_w, chroma qp offset), each
# frame's sample offset in the MP4, then per frame and slice (macroblock row order) a record: payload offset in the
# sample, payload size (escaped), the bit where the slice data starts (after the header, in the unescaped payload),
# qp | intra << 7
SLICE_REC = np.dtype([("off", "<u4"), ("size", "<u2"), ("data_bit", "u1"), ("qp_i", "u1")])


def slice_index(data):
    """The slice index (bytes) of an MP4 written with gpu=True; raises h264.UnsupportedStream if its slices are not
    macroblock rows."""
    from . import h264

    sps_nal, pps_nal, nal_len, samples = h264.mp4_samples(data)
    sps, pps = h264.parse_sps(h264.unescape(sps_nal)), h264.parse_pps(h264.unescape(pps_nal))
    H, W = sps["mb_h"], sps["mb_w"]
    rec = np.zeros((len(samples), H), SLICE_REC)
    for k, (off, size) in enumerate(samples):
        p, rows = off, []
        while p < off + size:
            n = int.from_bytes(data[p : p + nal_len], "big")
            a = p + nal_len
            if data[a] & 31 in (1, 5):
                h = h264.parse_slice_header(h264.unescape(data[a : a + min(n, 96)]), sps, pps)
                rows.append((h["first_mb"], a - off, n, h["data_bit"], h["qp"] | (128 if h["type"] == 2 else 0)))
            p = a + n
        rows.sort()
        if len(rows) != H or any(r[0] != i * W for i, r in enumerate(rows)):
            raise h264.UnsupportedStream("slices are not macroblock rows")
        rec[k] = [r[1:] for r in rows]
    head = np.array([len(samples), H, W, pps["chroma_qp_offset"]], "<i4")
    offs = np.array([o for o, _ in samples], "<u4")
    return head.tobytes() + offs.tobytes() + rec.tobytes()


def read_slice_index(buf):
    """(frames, mb_h, mb_w, chroma qp offset, sample offsets (frames,), records (frames, mb_h)) of a slice index."""
    b = np.frombuffer(buf, np.uint8)
    n, H, W, coff = (int(x) for x in b[:16].view("<i4"))
    offs = b[16 : 16 + 4 * n].view("<u4")
    rec = b[16 + 4 * n : 16 + 4 * n + n * H * SLICE_REC.itemsize].view(SLICE_REC).reshape(n, H)
    return n, H, W, coff, offs, rec


class _Blob(io.RawIOBase):
    """A file over bytes [pos, pos + size) of a memory map or a file descriptor: the decoder reads only what it asks
    for (nread counts it)."""

    def __init__(self, buf, fd, pos, size):
        self.buf, self.fd, self.pos, self.size, self.at = buf, fd, pos, size, 0
        self.nread = 0  # bytes read so far

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.at = min(max(0, (0, self.at, self.size)[whence] + offset), self.size)
        return self.at

    def tell(self):
        return self.at

    def read(self, n=-1):
        n = self.size - self.at if n is None or n < 0 else min(n, self.size - self.at)
        a = self.pos + self.at
        self.at += n
        self.nread += n
        return os.pread(self.fd, n, a) if self.fd is not None else self.buf[a : a + n].tobytes()


class Mp4Store:
    """An MP4 store (stable_datasets.framestore.build --format mp4) opened for reading. read="mmap" (default, or $FRAMESTORE_READ)
    memory-maps the data files; "pread" reads each piece the decoder asks for with one pread (local NVMe, network
    storage). Every video's location, frame count and size is read once into numpy."""

    def __init__(self, path, read=None):
        import lance

        from .store import FrameStore

        ds = lance.dataset(path)
        self.meta = video_format(ds.schema)
        if self.meta is None:
            raise ValueError(f"{path} is not an MP4 store (stable_datasets.framestore.build --format mp4)")
        self.keyint = int(self.meta["keyint"])
        self.read = read or os.environ.get("FRAMESTORE_READ", "mmap")
        if self.read not in ("mmap", "pread"):
            raise ValueError(f"read={self.read!r}: an MP4 store reads with mmap or pread")
        fs = FrameStore(path, index=True, column="video")  # blob descriptors of every video, the files mapped
        self.bufs, self.loc = fs.bufs, fs.index
        self.sloc = None  # the slice index's descriptors (stores built for the GPU decoder)
        if "slices" in ds.schema.names:
            si = FrameStore(path, index=True, column="slices")
            self.sbufs, self.sloc = si.bufs, si.index
        cols = ["frames", "fps"] + (["transposed"] if "transposed" in ds.schema.names else [])
        t = ds.to_table(columns=cols)
        self.n = t.column("frames").to_numpy().astype(np.int64)
        self.fps = t.column("fps").to_numpy()
        tr = t.column("transposed").to_numpy(zero_copy_only=False) if "transposed" in cols else np.zeros(len(t), bool)
        self.transposed = tr.astype(bool)
        self.fds = None
        if self.read == "pread":
            from .index import store_files

            self.fds = [os.open(f, os.O_RDONLY) for f in store_files(path, ds.get_fragments())[0]]

    def __len__(self):
        return len(self.n)

    def clips(self, rng, count, frames=16):
        """(video, start) of `count` random clips, drawn iid: a video uniformly among those with at least `frames`
        frames, then a start uniformly among its keyframes that leave room for the clip."""
        ok = np.flatnonzero(self.n >= frames)
        if not len(ok):
            raise ValueError(f"no video has {frames} frames")
        v = ok[rng.integers(0, len(ok), count)]
        starts = (self.n[v] - frames) // self.keyint + 1
        return v, (rng.random(count) * starts).astype(np.int64) * self.keyint

    def slices(self, video):
        """Video `video`'s slice index (read_slice_index's tuple)."""
        f, pos, size = (int(self.sloc[k][video]) for k in ("fid", "pos", "size"))
        return read_slice_index(self.sbufs[f][pos : pos + size])

    def clip_bytes(self, video, start, frames):
        """(the bytes of frames [start, start + frames) of video `video`: one stretch of its MP4, slice table
        (frames * mb_h, 4) int32: payload offset in that stretch, payload size, data bit, qp | intra << 7)."""
        _, H, _, _, offs, rec = self.slices(video)
        r = rec[start : start + frames]
        begin = int(offs[start] + r[0, 0]["off"])
        last = r[-1, -1]
        end = int(offs[start + frames - 1] + last["off"] + last["size"])
        f, pos, _ = (int(self.loc[k][video]) for k in ("fid", "pos", "size"))
        table = np.empty((frames * H, 4), np.int32)
        table[:, 0] = (offs[start : start + frames, None].astype(np.int64) + r["off"] - begin).reshape(-1)
        table[:, 1] = r["size"].reshape(-1)
        table[:, 2] = r["data_bit"].reshape(-1)
        table[:, 3] = r["qp_i"].reshape(-1)
        return self.bufs[f][pos + begin : pos + end], table

    def file(self, video):
        """Video `video`'s MP4 as a file object (reads go to the map or the file, nothing copied up front)."""
        f, pos, size = (int(self.loc[k][video]) for k in ("fid", "pos", "size"))
        return _Blob(self.bufs[f], None if self.fds is None else self.fds[f], pos, size)

    def source(self, video):
        """What the decoder opens: read="mmap", a copy of the video's bytes (a uint8 tensor); "pread", file()."""
        if self.fds is not None:
            return self.file(video)
        import torch

        f, pos, size = (int(self.loc[k][video]) for k in ("fid", "pos", "size"))
        return torch.from_numpy(np.array(self.bufs[f][pos : pos + size]))

    def decoder(self, video, device="cpu"):
        """A torchcodec VideoDecoder over video `video`: on the CPU (one thread), or device="cuda" on the GPU's video
        decoders (NVDEC; torchcodec keeps a cache of them, so a clip does not create one)."""
        from torchcodec.decoders import VideoDecoder

        if device == "cpu":
            return VideoDecoder(
                self.source(video), seek_mode="approximate", dimension_order="NCHW", num_ffmpeg_threads=1
            )
        from torchcodec.decoders import set_cuda_backend

        with set_cuda_backend("beta"):  # torchcodec's own NVDEC backend (the default from 0.12 on), not ffmpeg's
            return VideoDecoder(self.source(video), seek_mode="approximate", dimension_order="NCHW", device=device)

    def frames(self, video, start, frames=16, size=None, device="cpu"):
        """Frames [start, start + frames) of video `video`: a uint8 tensor (frames, 3, H, W) on `device`, in the
        video's own size and orientation (its valid region of the canvas), resized to size=(H, W) if given
        (antialiased bilinear, as the JPEG paths; on the CPU ~3x cheaper than ffmpeg resizing in the decoder)."""
        x = self.decoder(video, device).get_frames_in_range(int(start), int(start) + frames).data
        x = x[:, :, : int(self.loc["h"][video]), : int(self.loc["w"][video])]
        if self.transposed[video]:
            x = x.transpose(2, 3)
        if size is None or tuple(x.shape[2:]) == tuple(size):
            return x.contiguous()
        import torch.nn.functional as F

        if x.is_cuda:  # antialiased resize on the GPU takes floats
            y = F.interpolate(x.float(), size=tuple(size), mode="bilinear", antialias=True, align_corners=False)
            return y.round_().clamp_(0, 255).to(x.dtype)
        x = x.contiguous(memory_format=__import__("torch").channels_last)  # uint8, channels last: the fast CPU path
        return F.interpolate(x, size=tuple(size), mode="bilinear", antialias=True, align_corners=False)

    def decode(self, video, start, frames=16, size=None):
        """As frames, on the CPU, as uint8 numpy (frames, H, W, 3)."""
        return self.frames(video, start, frames, size).permute(0, 2, 3, 1).numpy()
