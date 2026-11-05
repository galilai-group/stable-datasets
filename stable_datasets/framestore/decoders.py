"""JPEG decoders, and the spec that says which decode resources a loader uses.

Decoder("nvjpeg" | "hw" | "cpu")(frames)   frames (uint8 tensors, numpy arrays or bytes) -> (3, h, w) uint8 tensors
Decoder("cpu", exact=True)                 libjpeg-turbo with per-MCU chroma upsampling, as nvJPEG and splicing need
decoders("nvjpeg:2,cpu:4")                 -> {"nvjpeg": 2, "cpu": 4}: a loader's decode resources and their counts
decode_resized(jpeg, h, w)                 one frame -> (h, w, 3) uint8, CPU decode + antialiased bilinear resize

The loaders decode on the GPU with "nvjpeg" and on the CPU cores with libjpeg-turbo. Where "nvjpeg" runs depends on
the GPU: torchvision's batched nvJPEG decodes on the NVJPG engines where there are any (H100, measured: engines
79-100% busy, SMs 20-66%; A100 / B200 / B300 have engines too and should do the same, not yet measured), else on the
CUDA cores with its Huffman decoding on the calling CPU (A40, L40S, Ada). "hw" (nvImageCodec on the same engines) is
slower than nvJPEG on them (H100: 2,256 vs 2,929 clips/s per token loader server), so the loaders no longer take it;
only the converter's verification can still choose it (convert --gpu hw / auto).
"""

import os
from contextlib import nullcontext

import numpy as np


class Decoder:
    """frames (uint8 tensors, numpy arrays or bytes) -> list of (3, h, w) uint8 tensors on the decoder's device.

    "cpu"     libjpeg-turbo via simplejpeg (fancy upsampling), in the calling thread, GIL released
    "nvjpeg"  torchvision decode_jpeg on CUDA: nvJPEG's batched decode, on the NVJPG engines where the GPU has them,
              else on the CUDA cores with Huffman decoding on this process's CPU (per-MCU chroma upsampling, which the
              splice needs); alias "tv"
    "hw"      nvImageCodec on the NVJPG engines (A100 / H100 / B200, not Ada), one CPU thread
              ($FRAMESTORE_NVIMGCODEC_THREADS); baseline JPEGs only, frames it cannot decode come back as None.
              Whole frames only: its RGB interpolates chroma across MCU borders (unlike nvJPEG), wrong for spliced
              JPEGs. Used by the converter's verification only.
    The decoded images of the last call stay referenced by the decoder (self.keep): the engines' buffers belong to
    nvImageCodec, and a kernel still reading them must not see them freed.
    """

    def __init__(self, kind="nvjpeg", device="cuda", exact=False):
        self.kind = {"tv": "nvjpeg"}.get(kind, kind)
        self.device, self.exact, self.keep = device, exact, []
        if self.kind == "hw":
            from nvidia import nvimgcodec as nv

            self.nv = nv
            # one CPU thread: nvImageCodec otherwise starts a pool per decoder (~4 busy cores per server measured on an
            # H100), which starves the loader's CPU workers
            self.dec = nv.Decoder(
                backends=[nv.Backend(nv.BackendKind.HW_GPU_ONLY)],
                max_num_cpu_threads=int(os.environ.get("FRAMESTORE_NVIMGCODEC_THREADS", "1")),
            )
            self.params = nv.DecodeParams(color_spec=nv.ColorSpec.SRGB, sample_format=nv.SampleFormat.P_RGB)

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
        if self.kind == "nvjpeg":  # fastest under load on every GPU measured (L40S, RTX 6000 Ada, H100)
            from torchvision.io import ImageReadMode, decode_jpeg

            frames = [f if isinstance(f, torch.Tensor) else torch.from_numpy(_np(f)) for f in frames]
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                return decode_jpeg(frames, mode=ImageReadMode.RGB, device=self.device)
        s = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        imgs = self.dec.decode([_np(f).tobytes() for f in frames], params=self.params, cuda_stream=s)
        self.keep = imgs
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            return [None if i is None else torch.as_tensor(i, device=self.device) for i in imgs]


def decoders(spec):
    """A loader's decoder spec -> {kind: count}: "nvjpeg:2,cpu:4" (a count of "auto" or none: None, the loader's
    choice). Kinds: nvjpeg (torchvision's nvJPEG: the NVJPG engines where the GPU has them, else the CUDA cores), cpu
    (libjpeg-turbo)."""
    out = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        kind, _, n = part.partition(":")
        kind = {"tv": "nvjpeg"}.get(kind, kind)
        if kind in ("hw", "nvjpg", "engines"):
            raise ValueError(
                f"{spec!r}: the hw decoder was removed from the loaders; nvjpeg already decodes on the NVJPG engines"
            )
        if kind not in ("nvjpeg", "cpu"):
            raise ValueError(f"unknown decoder {kind!r} in {spec!r}: nvjpeg or cpu")
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
