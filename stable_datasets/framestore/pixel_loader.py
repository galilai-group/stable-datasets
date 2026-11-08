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
    """Random windows of `frames` rows, every `stride`-th, drawn iid by FrameStore.windows (an episode uniformly among
    those long enough, then the window's start uniformly in it): the token loader's draw."""

    def __init__(self, store, frames, stride, seed):
        self.store, self.frames, self.stride = store, frames, stride
        self.rng = np.random.default_rng(seed)

    def rows(self, count):
        return self.store.windows(self.rng, count, self.frames, self.stride).reshape(count, self.frames)


def _open(cfg, seed):
    if cfg["mp4"]:  # clips drawn by Mp4Store.clips, in the worker
        from .mp4 import Mp4Store

        return Mp4Store(cfg["store"], read=cfg["read"]), np.random.default_rng(seed)
    from .store import FrameStore

    store = FrameStore(cfg["store"], index=cfg["index"], read=cfg["read"], column=cfg["column"])
    return store, Windows(store, cfg["frames"], cfg["stride"], seed)


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
    from .resources import lower_priority

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lower_priority(nice)  # the consumer and the decode servers win any contention for the CPU


def cpu_decoder(cfg, frames, jobs, done, seed):
    """jobs: (slot, pos); writes frames[slot, pos] = (T, H, W, 3) RGB uint8 of one random window, each frame
    decoded and resized by cfg["cpu_decode"](jpeg, H, W)."""
    _no_cuda(cfg["nice"])
    torch.set_num_threads(1)
    decode = _function(cfg["cpu_decode"])
    store, win = _open(cfg, seed)
    out = frames.numpy()
    H, W = cfg["size"]
    if cfg["mp4"]:  # a clip from its keyframe: decoded and resized by the decoder
        while (job := jobs.get()) is not None:
            v, start = store.clips(win, 1, cfg["frames"])
            out[job] = store.decode(v[0], start[0], cfg["frames"], (H, W))
            done.put(job[0])
        return
    drop = os.environ.get("FRAMESTORE_DROP_PAGES") == "1"  # strictly cold runs: unmap what each clip touched
    while (job := jobs.get()) is not None:
        slot, pos = job
        fid, p, n, _, _ = store.descriptors(win.rows(1)[0])
        for j, (i, a, m) in enumerate(zip(fid, p, n)):
            out[slot, pos, j] = decode(store.frame(i, a, m), H, W)
        if drop:
            store.drop_pages()
        done.put(slot)


def byte_reader(cfg, data, offsets, jobs, done, seed):
    """jobs: slot; packs one batch of random windows' JPEGs into data[slot], offsets[slot] (B*T + 1)."""
    _no_cuda(cfg["nice"])
    store, win = _open(cfg, seed)
    buf, offs = data.numpy(), offsets.numpy()
    drop = os.environ.get("FRAMESTORE_DROP_PAGES") == "1"
    while (slot := jobs.get()) is not None:
        try:
            offs[slot] = store.copy_frames(win.rows(cfg["batch"]).reshape(-1), buf[slot])[0]
        except ValueError as e:
            raise RuntimeError(f"byte slot too small for a batch ({e}): raise bytes_per_frame") from None
        if drop:
            store.drop_pages()
        done.put(slot)


def _gpu_decode_into(dec, data, offs, out, B, T, stream):
    """Decode B*T JPEGs and resize every frame whole to H x W (antialiased bilinear) into out (B, T, H, W, 3)."""
    import torch.nn.functional as F

    H, W = out.shape[2:4]
    o = offs.tolist()
    if dec.kind == "triton":  # one buffer, no per-frame copies
        from .gpu_jpeg import decode_buffer

        imgs = decode_buffer(data[: o[B * T]], offs[: B * T + 1].numpy(), dec.device, stream)
    else:
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
        if dec.kind == "triton":  # the byte slots, page-locked: copied to the GPU whole, asynchronously
            err = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), data.numel(), 0)
            assert int(err) == 0, err
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

    def nvdec_path(k):  # MP4 store: a batch of clips decoded by NVDEC and resized on the GPU, into a clip slot
        from .mp4 import Mp4Store

        store, rng = Mp4Store(cfg["store"], read=cfg["read"]), np.random.default_rng(cfg["seed"] + 7919 * (k + 1))
        T, (H, W), stream = cfg["frames"], cfg["size"], torch.cuda.Stream()
        own = torch.empty(clips.shape[1:], dtype=torch.uint8, device="cuda")  # a slot is taken only to copy into
        while not stop.is_set():
            vid, start = store.clips(rng, B, T)
            with torch.cuda.stream(stream):
                for b in range(B):
                    own[b] = store.frames(vid[b], start[b], T, (H, W), device="cuda").permute(0, 2, 3, 1)
            if (v := take()) is None:
                break
            with torch.cuda.stream(stream):
                clips[v].copy_(own)
            stream.synchronize()
            ready.put(v)
            stats[1] += B

    threads = ([threading.Thread(target=cpu_path, daemon=True)] if cfg["cpu_workers"] else []) + (
        [threading.Thread(target=gpu_path, daemon=True)] if cfg["byte_readers"] else []
    )
    if cfg["decoder"] == "nvdec":
        threads += [threading.Thread(target=nvdec_path, args=(k,), daemon=True) for k in range(cfg["nvdec_threads"])]
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
        store: frame store (blob-encoded, read by `read`; or not, read by lance take), or an MP4 store
            (stable_datasets.framestore.mp4: clips decoded from their keyframe on the CPU, by torchcodec; stride 1).
        batch, frames, stride, size: clips per batch, frames per clip, rows between two frames, (H, W) of each frame.
        cpu_workers: CPU decode processes (cpu_decode: "module:function" or a function of (jpeg, H, W), default
            decoders.decode_resized); default: the CPUs a job may use (resources.job_cpus: 2 stay free). They run
            below the trainer and the decode servers in priority (nice), so no core is held back for those.
        byte_readers: per decode server, feeding its GPU decoder; gpu_servers: decode server processes (MPS
            clients; nvJPEG's host side is serial per process). Loader alone, H100, 23 vCPUs, store in memory:
            "nvjpeg:2" 1,126 clips/s, "nvjpeg:2,cpu" 1,720 (benchmarks/run_matrix.sh).
        decoder: GPU JPEG decoder, "nvjpeg" (on the NVJPG engines where the GPU has them, else the CUDA cores) or
            "triton" (stable_datasets.framestore.gpu_jpeg: all of it in Triton kernels on the CUDA cores; converted stores).
        decoders: instead of gpu_servers, decoder and cpu_workers, a decoder spec (decoders.decoders): "nvjpeg:2,
            cpu:16" runs two nvJPEG servers and 16 CPU decoders (a kind with no count: nvjpeg 2, triton 1, cpu sized
            from the node).
        index, read: FrameStore's (a per-node index; the read policy).
        nvdec_threads: per nvdec server, threads each decoding a batch of MP4 clips on the GPU's video decoders.
        nice: the CPU processes' priority below the consumer's (and I/O priority, resources.lower_priority); 0 to
            run level with it.
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
        nice=10,
        seed=0,
        decoder="nvjpeg",
        decoders=None,
        cpu_decode="stable_datasets.framestore.decoders:decode_resized",
        index=False,
        read=None,
        bytes_per_frame=96 << 10,
        nvdec_threads=4,
    ):
        import lance

        from .mp4 import video_format
        from .resources import job_cpus

        self.B = batch
        H, W = size
        mp4 = video_format(lance.dataset(store).schema) is not None
        if decoder not in ("nvjpeg", "triton"):
            raise ValueError(f"decoder {decoder!r}: the decode servers decode with nvjpeg or triton")
        kinds = [decoder] * gpu_servers
        if decoders is not None:
            from .decoders import decoders as parse

            d = parse(decoders)
            kinds = ["nvjpeg"] * (2 if d.get("nvjpeg", 0) is None else d.get("nvjpeg", 0))
            kinds += ["triton"] * (1 if d.get("triton", 0) is None else d.get("triton", 0))
            kinds += ["nvdec"] * (1 if d.get("nvdec", 0) is None else d.get("nvdec", 0))
            gpu_servers, cpu_workers = len(kinds), d.get("cpu", 0)
        if mp4:  # decoded by torchcodec, on the CPU or NVDEC; its stride was fixed when it was built
            if stride != 1:
                raise ValueError("an MP4 store keeps frames at its build's fps: stride must be 1")
            if decoders is None:
                kinds = []
            elif set(kinds) - {"nvdec"}:
                raise ValueError(f"decoders {decoders!r}: an MP4 store decodes with nvdec and cpu")
            gpu_servers, byte_readers = len(kinds), 0
        elif "nvdec" in kinds:
            raise ValueError(f"decoders {decoders!r}: nvdec decodes MP4 stores, this is a frame store")
        self.kinds = kinds
        if cpu_workers is None:
            cpu_workers = job_cpus()  # niced: the servers and the trainer win any contention
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
                "mp4": mp4,
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
                "nvdec_threads": nvdec_threads,
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
