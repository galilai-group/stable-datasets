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
import time

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


def byte_reader(cfg, data, offsets, tables, jobs, done, seed):
    """jobs: slot; packs one batch of random windows' JPEGs into data[slot], offsets[slot] (B*T + 1). MP4 store: the
    clips' stretches of their MP4s into data[slot], their slice table into tables[slot], and offsets[slot] = bytes,
    slices, then per clip h, w, transposed (gpu_h264.gather)."""
    _no_cuda(cfg["nice"])
    store, win = _open(cfg, seed)
    buf, offs = data.numpy(), offsets.numpy()
    if cfg["mp4"]:
        from .gpu_h264 import gather

        B, T, tab = cfg["batch"], cfg["frames"], tables.numpy()
        while (slot := jobs.get()) is not None:
            v, start = store.clips(win, B, T)
            try:
                raw, table, _ = gather(store, [(v[i], start[i], T) for i in range(B)], out=buf[slot])
            except ValueError as e:
                raise RuntimeError(f"byte slot too small for a batch ({e}): raise bytes_per_frame") from None
            tab[slot, : len(table)] = table
            offs[slot, :2] = len(raw), len(table)
            offs[slot, 2 : 2 + 3 * B] = np.concatenate([store.loc["h"][v], store.loc["w"][v], store.transposed[v]])
            done.put(slot)
        return
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


# a decoder's cost to a GPU-bound consumer, cheapest first: CPU decoding takes no GPU time; NVDEC and nvJPEG run on
# the GPU's fixed-function decoders where it has them (nvJPEG's Huffman decoding on a CPU core where it has none);
# triton decodes on the SMs the consumer computes on
COST = {"cpu": 0, "nvdec": 1, "nvjpeg": 1, "triton": 2}
LANES = 2 + 16  # per server: the CPU path, the byte path, nvdec threads


def decode_server(cfg, clips, frames, data, offsets, tables, free, ready, stats, stop, waiting):
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
        ctx.Process(
            target=byte_reader, args=(cfg, data, offsets, tables, bjobs, bdone, cfg["seed"] + 500 + i), daemon=True
        )
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

    def take_ranked(lane, rank):
        """A clip slot for a batch that `lane` (this server's row of `waiting`) holds or is ready to decode, `rank`
        its decoder's cost to the consumer (COST). cheapest_first: the slot goes to the lane only while no lane of a
        cheaper decoder (any server) waits for one, so a consumer slower than the loader is fed by the cheapest
        decoders and the costlier ones decode only the shortfall."""
        if not cfg["cheapest_first"]:
            return take()
        row = waiting[cfg["server"]]
        row[lane, 1] = rank
        row[lane, 0] = 1  # only this thread writes its lane
        try:
            while not stop.is_set():
                if bool((waiting[..., 0] * (waiting[..., 1] < rank)).any()):
                    time.sleep(0.001)
                    continue
                try:
                    v = free.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not bool((waiting[..., 0] * (waiting[..., 1] < rank)).any()):
                    return v
                free.put(v)
            return None
        finally:
            row[lane, 0] = 0

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
            if (v := take_ranked(0, COST["cpu"])) is None:
                break
            with torch.cuda.stream(stream):
                clips[v].copy_(frames[slot], non_blocking=True)
            stream.synchronize()
            ready.put(v)
            stats[0] += B
            pending[slot] = B
            for pos in range(B):
                cjobs.put((slot, pos))

    def mp4_path():  # MP4 store, decoded by gpu_h264: a batch's bytes and slice table -> RGB clips in a clip slot
        from .gpu_h264 import decode, prepare_device, to_rgb

        stream, T = torch.cuda.Stream(), cfg["frames"]
        err = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), data.numel(), 0)
        assert int(err) == 0, err
        for s in range(data.shape[0]):
            bjobs.put(s)
        while not stop.is_set():
            try:
                slot = bdone.get(timeout=0.5)
            except queue.Empty:
                continue
            if (v := take_ranked(1, COST["triton"])) is None:
                break
            o = offsets[slot].numpy()
            n, S = int(o[0]), int(o[1])
            with torch.cuda.stream(stream):
                batch = prepare_device(data[slot, :n], tables[slot, :S].numpy(), cfg["geom"])
                (Y, U, V), _ = decode(batch)
                to_rgb(Y, U, V, o[2 : 2 + 3 * B].reshape(3, B).T, T, clips[v])
            stream.synchronize()
            ready.put(v)
            stats[1] += B
            bjobs.put(slot)

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
            if (v := take_ranked(1, COST[cfg["decoder"]])) is None:
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
            if (v := take_ranked(2 + k, COST["nvdec"])) is None:
                break
            with torch.cuda.stream(stream):
                clips[v].copy_(own)
            stream.synchronize()
            ready.put(v)
            stats[1] += B

    threads = ([threading.Thread(target=cpu_path, daemon=True)] if cfg["cpu_workers"] else []) + (
        [threading.Thread(target=mp4_path if cfg["mp4"] else gpu_path, daemon=True)] if cfg["byte_readers"] else []
    )
    if cfg["decoder"] == "nvdec":
        threads += [threading.Thread(target=nvdec_path, args=(k,), daemon=True) for k in range(cfg["nvdec_threads"])]
    for t in threads:
        t.start()
    # a decoder process or thread that dies (spawn failed, a decode raised) would leave the consumer waiting forever:
    # exit instead, which next() reports
    while not stop.wait(1.0):
        if any(not p.is_alive() for p in procs) or any(not t.is_alive() for t in threads):
            for p in procs:
                p.kill()
            os._exit(1)
    for p in procs:
        p.kill()
    for p in procs:
        p.join(timeout=5)
    os._exit(0)


def auto_decoders(store, mp4):
    """The decoder spec "auto" resolves to: every decoder this node can run on this store. CPU always; an MP4 store:
    nvdec if torchcodec decodes on this GPU, triton if the store has a slice index and Triton is new enough
    (gpu_h264); a JPEG store: nvjpeg, and triton if the store is converted (restart-coded)."""
    import importlib.util

    parts = ["cpu"]
    has_triton = importlib.util.find_spec("triton") is not None
    if mp4:
        from .mp4 import Mp4Store

        st = Mp4Store(store)
        try:
            st.frames(0, 0, 1, (64, 64), device="cuda")
            parts.append("nvdec:1")
        except Exception:  # no NVDEC here, or torchcodec without CUDA
            pass
        if has_triton and st.sloc is not None:
            from . import gpu_h264

            try:
                gpu_h264._require_triton()
                parts.append("triton:2")
            except (ImportError, RuntimeError):
                pass
    else:
        import lance

        from .store import store_format

        parts.append("nvjpeg:2")
        if has_triton and store_format(lance.dataset(store).schema) is not None:
            parts.append("triton:2")
    return ",".join(parts)


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
            from the node). "auto": every decoder this node and store can use (auto_decoders), cheapest first.
        index, read: FrameStore's (a per-node index; the read policy).
        nvdec_threads: per nvdec server, threads each decoding a batch of MP4 clips on the GPU's video decoders.
        nice: the CPU processes' priority below the consumer's (and I/O priority, resources.lower_priority); 0 to
            run level with it.
        cheapest_first: a decoder takes a clip slot only while no cheaper decoder (COST: cpu, then nvdec and nvjpeg,
            then triton) has a batch waiting for one. All decoders run; a consumer slower than the loader is fed by
            the cheapest, and the costlier decode only the shortfall (a GPU-bound consumer loses SM time to triton,
            none to CPU decoding). False: first come, first served. stats() says whether the consumer still waits.
        gpu_share: under MPS, the share of the GPU's SMs each decode server may use (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE,
            e.g. 0.2); None: no limit.
    """

    def __init__(
        self,
        store,
        batch,
        frames=16,
        stride=1,
        size=(224, 224),
        cpu_workers=None,
        byte_readers=None,
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
        cheapest_first=True,
        gpu_share=None,
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
        if decoders == "auto":
            decoders = auto_decoders(store, mp4)
        self.spec = decoders
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
            elif set(kinds) - {"nvdec", "triton"}:
                raise ValueError(f"decoders {decoders!r}: an MP4 store decodes with triton, nvdec and cpu")
            gpu_servers = len(kinds)
            byte_readers = (byte_readers or 2) if "triton" in kinds else 0
            geom, max_slices = (0, 0, 0), 1
            if "triton" in kinds:
                from .mp4 import Mp4Store

                st = Mp4Store(store)
                if st.sloc is None:
                    raise ValueError(f"{store} has no slice index: rebuild it to decode on the GPU (decoders triton)")
                _, mh, mw, coff, _, _ = st.slices(0)
                geom, max_slices = (mh, mw, coff), batch * frames * mh
        elif "nvdec" in kinds:
            raise ValueError(f"decoders {decoders!r}: nvdec decodes MP4 stores, this is a frame store")
        self.kinds = kinds
        if cpu_workers is None:
            cpu_workers = job_cpus()  # niced: the servers and the trainer win any contention
        byte_readers = 1 if byte_readers is None else byte_readers  # (MP4 stores, GPU decode: 2, set above)
        self.cpu_workers, self.byte_readers = cpu_workers, byte_readers if gpu_servers else 0
        ctx = mp.get_context("spawn")
        self.clips = torch.empty(clip_slots, batch, frames, H, W, 3, dtype=torch.uint8, device="cuda")
        host = torch.empty(host_slots if cpu_workers else 1, batch, frames, H, W, 3, dtype=torch.uint8).share_memory_()
        n = max(1, gpu_servers)
        data = torch.empty(
            n, byte_slots if byte_readers else 1, batch * frames * bytes_per_frame, dtype=torch.uint8
        ).share_memory_()
        offsets = torch.zeros(
            n, data.shape[1], max(batch * frames + 1, 3 * batch + 2), dtype=torch.int64
        ).share_memory_()
        tables = torch.zeros(n, data.shape[1] if mp4 else 1, max_slices if mp4 else 1, 4, dtype=torch.int32)
        tables = tables.share_memory_()
        self._stats = torch.zeros(n, 2, dtype=torch.int64).share_memory_()
        if nvdec_threads > LANES - 2:
            raise ValueError(f"nvdec_threads: at most {LANES - 2}")
        # per server and lane (a thread that takes clip slots): (waits for a slot, its decoder's COST)
        waiting = torch.zeros(n, LANES, 2, dtype=torch.int64).share_memory_()
        self.free, self.ready, self.stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        for v in range(clip_slots):
            self.free.put(v)
        self.servers = []
        for i in range(n):
            cfg = {
                "store": store,
                "mp4": mp4,
                "geom": geom if mp4 else None,
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
                "byte_readers": byte_readers if gpu_servers and (not kinds or kinds[i] != "nvdec") else 0,
                "seed": seed + 1000 * i,
                "server": i,
                "cheapest_first": cheapest_first,
            }
            p = ctx.Process(
                target=decode_server,
                args=(
                    cfg,
                    self.clips,
                    host,
                    data[i],
                    offsets[i],
                    tables[i],
                    self.free,
                    self.ready,
                    self._stats[i],
                    self.stop,
                    waiting,
                ),
            )
            share = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
            if gpu_share is not None:  # read by the server's CUDA context: set while spawning it, not in this process
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, round(100 * gpu_share)))
            try:
                p.start()
            finally:
                if gpu_share is not None:
                    if share is None:
                        os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
                    else:
                        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = share
            self.servers.append(p)
        self.prev = None
        self._starved, self._since, self._n, self._done = 0.0, None, 0, (-1, 0.0)
        self._release = queue.Queue()
        threading.Thread(target=self._releaser, daemon=True).start()

    def _releaser(self):
        """Frees the previous batch's slot once the consumer's GPU work queued before next() is done, and notes when
        that was (stats: GPU starvation)."""
        while True:
            ev, v, n = self._release.get()
            ev.synchronize()
            self._done = (n, time.perf_counter())
            if v is not None:
                self.free.put(v)

    def __iter__(self):
        return self

    def stats(self):
        """Since the first batch: gpu_starved, the share of the time the consumer's GPU sat idle waiting for a batch
        (from the end of the work it queued before next() to the batch's arrival). Above a few percent the consumer
        is data-bound, and more decoders, CPUs or a cheaper store would speed it up; near zero the loader keeps up,
        and with cheapest_first the costlier decoders idle. Time in next() alone would not tell: a consumer that
        queues its GPU work asynchronously (a CUDA graph) waits in next() for slots its own GPU work still holds.
        decoded_by: the share of batches each decoder kind decoded."""
        elapsed = time.perf_counter() - self._since if self._since else 0.0
        by = {}
        for i, (c, g) in enumerate(self._stats.tolist()):
            by["cpu"] = by.get("cpu", 0) + c
            if self.kinds and i < len(self.kinds):
                by[self.kinds[i]] = by.get(self.kinds[i], 0) + g
        total = sum(by.values()) or 1
        return {
            "gpu_starved": round(self._starved / elapsed, 3) if elapsed else None,
            "decoded_by": {k: round(v / total, 3) for k, v in by.items() if v},
        }

    def __next__(self):
        ev, n = torch.cuda.Event(), self._n
        ev.record()
        self._release.put((ev, self.prev, n))
        self._n += 1
        while True:
            try:
                v = self.ready.get(timeout=5)
                t = time.perf_counter()
                if self._since is None:
                    self._since = t  # from the first batch: start-up is not starvation
                elif ev.query():  # the GPU went idle before this batch came
                    done_n, done_t = self._done
                    self._starved += max(0.0, t - done_t) if done_n == n else 0.0
                break
            except queue.Empty:
                dead = [p.exitcode for p in self.servers if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"decode server exited ({dead})")
        self.prev = v
        return self.clips[v]

    def close(self):
        self.stop.set()
        for p in self.servers:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
