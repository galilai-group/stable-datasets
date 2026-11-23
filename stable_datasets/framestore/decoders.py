"""JPEG decoders, and the spec that says which decode resources a loader uses.

Decoder("nvjpeg" | "triton" | "cpu")(frames) (uint8 tensors, numpy arrays or bytes) -> (3, h, w) uint8 tensors
Decoder("cpu", exact=True)                 libjpeg-turbo with per-MCU chroma upsampling, as nvJPEG and splicing need
decoders("nvjpeg:2,cpu:4")                 -> {"nvjpeg": 2, "cpu": 4}: a loader's decode resources and their counts
decode_resized(jpeg, h, w)                 one frame -> (h, w, 3) uint8, CPU decode + antialiased bilinear resize

The loaders decode on the GPU with "nvjpeg" and on the CPU cores with libjpeg-turbo. Where "nvjpeg" runs depends on
the GPU: torchvision's batched nvJPEG decodes on the NVJPG engines where there are any (H100, measured: engines
79-100% busy, SMs 20-66%; A100 / B200 / B300 have engines too and should do the same, not yet measured), else on the
CUDA cores with its Huffman decoding on the calling CPU (A40, L40S, Ada). nvImageCodec on the same engines ("hw",
removed) was slower than nvJPEG on them (H100: 2,256 vs 2,929 clips/s per token loader server; 1,612 vs 3,964 frame
pairs/s verifying a conversion).
"""

from contextlib import nullcontext

import numpy as np


class Decoder:
    """frames (uint8 tensors, numpy arrays or bytes) -> list of (3, h, w) uint8 tensors on the decoder's device.

    "cpu"     libjpeg-turbo via simplejpeg (fancy upsampling), in the calling thread, GIL released
    "nvjpeg"  torchvision decode_jpeg on CUDA: nvJPEG's batched decode, on the NVJPG engines where the GPU has them,
              else on the CUDA cores with Huffman decoding on this process's CPU (per-MCU chroma upsampling, which the
              splice needs); alias "tv"
    "triton"  stable_datasets.framestore.gpu_jpeg: the whole decode in Triton kernels on the CUDA cores, entropy decoding included (one lane per MCU),
              for restart-coded frames (converted stores); libjpeg-turbo's pixels with plain upsampling, as nvJPEG
    """

    def __init__(self, kind="nvjpeg", device="cuda", exact=False):
        self.kind = {"tv": "nvjpeg"}.get(kind, kind)
        if self.kind not in ("nvjpeg", "triton", "cpu"):
            raise ValueError(f"unknown decoder {kind!r}: nvjpeg, triton or cpu")
        self.device, self.exact = device, exact

    def __call__(self, frames, stream=None):
        import torch

        if self.kind == "cpu":
            import simplejpeg

            return [
                torch.from_numpy(
                    simplejpeg.decode_jpeg(_np(f), colorspace="RGB", fastdct=False, fastupsample=self.exact)
                ).permute(2, 0, 1)
                for f in frames
            ]
        if self.kind == "triton":
            from .gpu_jpeg import decode_buffer

            frames = [_np(f) for f in frames]
            off = np.concatenate(([0], np.cumsum([len(f) for f in frames])))
            return decode_buffer(torch.from_numpy(np.concatenate(frames)), off, self.device, stream)
        from torchvision.io import ImageReadMode, decode_jpeg

        frames = [f if isinstance(f, torch.Tensor) else torch.from_numpy(_np(f)) for f in frames]
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            return decode_jpeg(frames, mode=ImageReadMode.RGB, device=self.device)


def decoders(spec):
    """A loader's decoder spec -> {kind: count}: "nvjpeg:2,cpu:4" (a count of "auto" or none: None, the loader's
    choice). Kinds: nvjpeg (torchvision's nvJPEG: the NVJPG engines where the GPU has them, else the CUDA cores), triton
    (stable_datasets.framestore.gpu_jpeg: all of it on the CUDA cores; restart-coded frames), cpu (libjpeg-turbo; MP4 stores: ffmpeg),
    nvdec (MP4 stores: torchcodec on the GPU's video decoders)."""
    out = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        kind, _, n = part.partition(":")
        kind = {"tv": "nvjpeg"}.get(kind, kind)
        if kind in ("hw", "nvjpg", "engines"):
            raise ValueError(f"{spec!r}: the hw decoder was removed; nvjpeg already decodes on the NVJPG engines")
        if kind not in ("nvjpeg", "triton", "cpu", "nvdec"):
            raise ValueError(f"unknown decoder {kind!r} in {spec!r}: nvjpeg, triton, cpu or nvdec")
        out[kind] = None if n in ("", "auto") else int(n)
    return out


def _np(f):
    if isinstance(f, np.ndarray):
        return f
    if isinstance(f, (bytes, bytearray, memoryview)):
        return np.frombuffer(f, np.uint8)
    return f.numpy()


def decode_resized(jpeg, height, width):
    """One frame (bytes or uint8 array) -> uint8 (height, width, 3): libjpeg-turbo decode (torchvision; PIL for what
    it cannot read, e.g. CMYK or PNG) and antialiased bilinear resize. The CPU path of pixel_loader.SharedClipLoader."""
    import io

    import torch
    import torch.nn.functional as F
    from torchvision.io import ImageReadMode, decode_image

    try:
        x = decode_image(torch.frombuffer(bytearray(_np(jpeg)), dtype=torch.uint8), mode=ImageReadMode.RGB)
    except RuntimeError:
        from PIL import Image

        x = torch.from_numpy(np.array(Image.open(io.BytesIO(bytes(_np(jpeg)))).convert("RGB"))).permute(2, 0, 1)
    if tuple(x.shape[1:]) != (height, width):
        x = F.interpolate(
            x[None].contiguous(memory_format=torch.channels_last),
            size=(height, width),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )[0]
    return x.permute(1, 2, 0).numpy()
