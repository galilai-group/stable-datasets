"""Whole-frame clip loader: decoded clips (B, T, H, W, 3) uint8 on the GPU from a frame store, every frame decoded
whole and resized to H x W, for training steps that build their views from full frames (e.g. on the GPU).

    loader = SharedClipLoader("/vol/frames.lance", batch=64, frames=16, stride=2, size=(224, 224))
    clip = next(loader)                         # (64, 16, 224, 224, 3) uint8 CUDA tensor, valid until the next call

  CPU decoders (xW)   read a clip (FrameStore) and decode it with cpu_decode into a page-locked shared host slot
  byte readers (xK)   read a batch's JPEG bytes (FrameStore.copy_frames) into a shared byte slot
  decode server(s)    CUDA (MPS) clients: H2D of CPU-decoded slots, GPU decode (nvJPEG, or the NVJPG engines) and
                      antialiased resize of byte slots, both into a ring of clip slots on the GPU shared with the
                      trainer over CUDA IPC (nvJPEG cannot overlap a busy training step from inside its process: its
                      host side waits on per-image events)

Clips are drawn uniformly over all window starts (every episode long enough to hold one), as LeVJEPA v2's
VariableLengthBatchSampler draws them.
"""

import os
import queue
import threading

import numpy as np
import torch
import torch.multiprocessing as mp


class Windows:
    """Random windows of `frames` rows, every `stride`-th, uniform over all window starts of the episodes (starts,
    lengths) long enough to hold one."""

    def __init__(self, starts, lengths, frames, stride, seed):
        span = (frames - 1) * stride + 1
        ok = lengths >= span
        self.starts, self.n = starts[ok], lengths[ok] - span + 1
        if not len(self.n):
            raise ValueError(f"no episode holds {frames} frames {stride} rows apart")
        self.cum = np.cumsum(self.n)
        self.frames, self.stride = frames, stride
        self.rng = np.random.default_rng(seed)

    def rows(self, count):
        draw = self.rng.integers(self.cum[-1], size=count)
        k = np.searchsorted(self.cum, draw, side="right")
        first = self.starts[k] + draw - np.concatenate(([0], self.cum))[k]
        return first[:, None] + np.arange(self.frames) * self.stride  # (count, frames)


def _open(cfg, seed):
    from .store import FrameStore

    store = FrameStore(cfg["store"], index=cfg["index"], read=cfg["read"], column=cfg["column"])
    return store, Windows(store.ep_start, store.ep_len, cfg["frames"], cfg["stride"], seed)


def _function(spec):
    """ "module:function" (or "/dir/module:function", importing from /dir) -> the function. A string, so it reaches
    the loader's spawned processes."""
    import importlib
    import sys

    if not isinstance(spec, str):
        return spec
    path, _, name = spec.rpartition(":")
    if "/" in path:
        root, path = path.rsplit("/", 1)
        sys.path.insert(0, root)
    return getattr(importlib.import_module(path), name)


def _no_cuda(nice=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if nice:
        os.nice(nice)


def cpu_decoder(cfg, frames, jobs, done, seed):
    """jobs: (slot, pos); writes frames[slot, pos] = (T, H, W, 3) RGB uint8 of one random window, each frame
    decoded and resized by cfg["cpu_decode"](jpeg, H, W)."""
    _no_cuda(cfg["nice"])
    torch.set_num_threads(1)
    decode = _function(cfg["cpu_decode"])
    store, win = _open(cfg, seed)
    out = frames.numpy()
    H, W = cfg["size"]
    while (job := jobs.get()) is not None:
        slot, pos = job
        fid, p, n, _, _ = store.descriptors(win.rows(1)[0])
        for j, (i, a, m) in enumerate(zip(fid, p, n)):
            out[slot, pos, j] = decode(store.frame(i, a, m), H, W)
        done.put(slot)


def byte_reader(cfg, data, offsets, jobs, done, seed):
    """jobs: slot; packs one batch of random windows' JPEGs into data[slot], offsets[slot] (B*T + 1)."""
    _no_cuda(cfg["nice"])
    store, win = _open(cfg, seed)
    buf, offs = data.numpy(), offsets.numpy()
    while (slot := jobs.get()) is not None:
        try:
            offs[slot] = store.copy_frames(win.rows(cfg["batch"]).reshape(-1), buf[slot])[0]
        except ValueError as e:
            raise RuntimeError(f"byte slot too small for a batch ({e}): raise bytes_per_frame") from None
        done.put(slot)


def _gpu_decode_into(dec, data, offs, out, B, T, stream):
    """Decode B*T JPEGs and resize every frame whole to H x W (antialiased bilinear) into out (B, T, H, W, 3)."""
    import torch.nn.functional as F

    H, W = out.shape[2:4]
    o = offs.tolist()
    imgs = dec([data[o[k] : o[k + 1]] for k in range(B * T)], stream)
    with torch.cuda.stream(stream):
        for b in range(B):  # the frames of a clip share one size
            clip = torch.stack(imgs[b * T : (b + 1) * T])
            if clip.shape[-2:] != (H, W):  # a store already at H x W needs no resize
                clip = (
                    F.interpolate(clip.float(), size=(H, W), mode="bilinear", antialias=True, align_corners=False)
                    .round_()
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
            out[b] = clip.permute(0, 2, 3, 1)


def decode_server(cfg, clips, frames, data, offsets, free, ready, stats, stop):
    import warnings

    warnings.filterwarnings("ignore")
    torch.cuda.set_device(0)
    B = cfg["batch"]
    ctx = mp.get_context("spawn")
    cjobs, bjobs, cdone, bdone = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Queue()
    procs = [
        ctx.Process(target=cpu_decoder, args=(cfg, frames, cjobs, cdone, cfg["seed"] + i), daemon=True)
        for i in range(cfg["cpu_workers"])
    ]
    procs += [
        ctx.Process(target=byte_reader, args=(cfg, data, offsets, bjobs, bdone, cfg["seed"] + 500 + i), daemon=True)
        for i in range(cfg["byte_readers"])
    ]
    for p in procs:
        p.start()
    if cfg["cpu_workers"]:
        err = torch.cuda.cudart().cudaHostRegister(frames.data_ptr(), frames.numel(), 0)
        assert int(err) == 0, err

    def take():
        while not stop.is_set():
            try:
                return free.get(timeout=0.5)
            except queue.Empty:
                pass
        return None

    def cpu_path():
        stream, pending = torch.cuda.Stream(), {}
        for s in range(frames.shape[0]):
            pending[s] = B
            for pos in range(B):
                cjobs.put((s, pos))
        while not stop.is_set():
            try:
                slot = cdone.get(timeout=0.5)
            except queue.Empty:
                continue
            pending[slot] -= 1
            if pending[slot]:
                continue
            if (v := take()) is None:
                break
            with torch.cuda.stream(stream):
                clips[v].copy_(frames[slot], non_blocking=True)
            stream.synchronize()
            ready.put(v)
            stats[0] += B
            pending[slot] = B
            for pos in range(B):
                cjobs.put((slot, pos))

    def gpu_path():
        from .decoders import Decoder

        stream, dec = torch.cuda.Stream(), Decoder(cfg["decoder"])
        for s in range(data.shape[0]):
            bjobs.put(s)
        while not stop.is_set():
            try:
                slot = bdone.get(timeout=0.5)
            except queue.Empty:
                continue
            if (v := take()) is None:
                break
            _gpu_decode_into(dec, data[slot], offsets[slot], clips[v], B, cfg["frames"], stream)
            stream.synchronize()
            ready.put(v)
            stats[1] += B
            bjobs.put(slot)

    threads = ([threading.Thread(target=cpu_path, daemon=True)] if cfg["cpu_workers"] else []) + (
        [threading.Thread(target=gpu_path, daemon=True)] if cfg["byte_readers"] else []
    )
    for t in threads:
        t.start()
    stop.wait()
    for p in procs:
        p.kill()
    for p in procs:
        p.join(timeout=5)
    os._exit(0)


class SharedClipLoader:
    """``next(loader)`` -> clip (B, T, H, W, 3) uint8 on the GPU, valid until the next call.

    Args:
        store: frame store (blob-encoded, read by `read`; or not, read by lance take).
        batch, frames, stride, size: clips per batch, frames per clip, rows between two frames, (H, W) of each frame.
        cpu_workers: CPU decode processes (cpu_decode: "module:function" or a function of (jpeg, H, W), default
            decoders.decode_resized); default: the CPUs a job may use (resources.job_cpus: 2 stay free), less one
            for the trainer and one for each GPU server (nvJPEG decodes Huffman codes on its CPU) and byte reader.
        byte_readers: per decode server, feeding its GPU decoder; gpu_servers: decode server processes (MPS
            clients; nvJPEG's host side is serial per process). H100, 23 vCPUs, ViT-S: 2 servers (16 CPU decoders)
            1,505 clips/s, 3 (14) 1,483, 4 (12) 1,450.
        decoder: GPU JPEG decoder, "nvjpeg" (default: fastest under load on every GPU measured) or "hw" (NVJPG engines).
        decoders: instead of gpu_servers, decoder and cpu_workers, a decoder spec (decoders.decoders): "nvjpeg:2,hw:1,
            cpu:16" runs two nvJPEG servers, one on the NVJPG engines and 16 CPU decoders (a kind with no count:
            nvjpeg 2, hw 1, cpu sized from the node).
        index, read: FrameStore's (a per-node index; the read policy).
    """

    def __init__(
        self,
        store,
        batch,
        frames=16,
        stride=1,
        size=(224, 224),
        cpu_workers=None,
        byte_readers=1,
        gpu_servers=2,
        column="frame",
        clip_slots=4,
        host_slots=3,
        byte_slots=3,
        nice=0,
        seed=0,
        decoder="nvjpeg",
        decoders=None,
        cpu_decode="stable_datasets.framestore.decoders:decode_resized",
        index=False,
        read=None,
        bytes_per_frame=96 << 10,
    ):
        from .resources import job_cpus

        self.B = batch
        H, W = size
        kinds = [decoder] * gpu_servers
        if decoders is not None:
            from .decoders import decoders as parse

            d = parse(decoders)
            kinds = ["nvjpeg"] * (2 if d.get("nvjpeg", 0) is None else d.get("nvjpeg", 0))
            kinds += ["hw"] * (1 if d.get("hw", 0) is None else d.get("hw", 0))
            gpu_servers, cpu_workers = len(kinds), d.get("cpu", 0)
        self.kinds = kinds
        if cpu_workers is None:
            cpu_workers = max(0, job_cpus() - 1 - gpu_servers * (1 + byte_readers))
        self.cpu_workers, self.byte_readers = cpu_workers, byte_readers if gpu_servers else 0
        ctx = mp.get_context("spawn")
        self.clips = torch.empty(clip_slots, batch, frames, H, W, 3, dtype=torch.uint8, device="cuda")
        host = torch.empty(host_slots if cpu_workers else 1, batch, frames, H, W, 3, dtype=torch.uint8).share_memory_()
        n = max(1, gpu_servers)
        data = torch.empty(
            n, byte_slots if byte_readers else 1, batch * frames * bytes_per_frame, dtype=torch.uint8
        ).share_memory_()
        offsets = torch.zeros(n, data.shape[1], batch * frames + 1, dtype=torch.int64).share_memory_()
        self._stats = torch.zeros(n, 2, dtype=torch.int64).share_memory_()
        self.free, self.ready, self.stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        for v in range(clip_slots):
            self.free.put(v)
        self.servers = []
        for i in range(n):
            cfg = {
                "store": store,
                "frames": frames,
                "stride": stride,
                "size": (H, W),
                "column": column,
                "batch": batch,
                "nice": nice,
                "decoder": kinds[i] if kinds else decoder,
                "cpu_decode": cpu_decode,
                "index": index,
                "read": read,
                "cpu_workers": cpu_workers if i == 0 else 0,
                "byte_readers": byte_readers if gpu_servers else 0,
                "seed": seed + 1000 * i,
            }
            p = ctx.Process(
                target=decode_server,
                args=(cfg, self.clips, host, data[i], offsets[i], self.free, self.ready, self._stats[i], self.stop),
            )
            p.start()
            self.servers.append(p)
        self.prev = None
        self._release = queue.Queue()
        threading.Thread(target=self._releaser, daemon=True).start()

    def _releaser(self):
        while True:
            ev, v = self._release.get()
            ev.synchronize()
            self.free.put(v)

    def __iter__(self):
        return self

    def __next__(self):
        if self.prev is not None:
            ev = torch.cuda.Event()
            ev.record()
            self._release.put((ev, self.prev))
        while True:
            try:
                v = self.ready.get(timeout=5)
                break
            except queue.Empty:
                dead = [p.exitcode for p in self.servers if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"decode server exited ({dead})")
        self.prev = v
        return self.clips[v]

    @property
    def stats(self):
        return self._stats.sum(0)

    def close(self):
        self.stop.set()
        for p in self.servers:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
