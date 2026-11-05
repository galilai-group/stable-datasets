"""JPEG decoders, and the spec that says which decode resources a loader uses.

Decoder("nvjpeg" | "hw" | "cpu")(frames)   frames (uint8 tensors, numpy arrays or bytes) -> (3, h, w) uint8 tensors
Decoder("hw").planes(frames, (sx, sy))     the NVJPG engines' raw Y, Cb, Cr planes (chroma not upsampled)
Decoder("cpu", exact=True)                 libjpeg-turbo with per-MCU chroma upsampling, as nvJPEG and splicing need
decoders("nvjpeg:2,hw:1,cpu:4")            -> {"nvjpeg": 2, "hw": 1, "cpu": 4}: decode resources and their counts
decode_resized(jpeg, h, w)                 one frame -> (h, w, 3) uint8, CPU decode + antialiased bilinear resize

A node has up to three kinds of JPEG decoder, and a loader can run them all at once: the NVJPG engines (A100 / H100 /
B200: separate silicon, no SMs), the CUDA cores, and the CPU cores (libjpeg-turbo). Which one "nvjpeg" runs on depends
on the GPU: torchvision's batched nvJPEG decodes on the engines where there are any (H100, measured: engines 79-100%
busy, SMs 20-66%), else on the CUDA cores with its Huffman decoding on the calling CPU (A40, L40S, Ada). So on an H100,
"nvjpeg" and "hw" servers share the same engines: mixing them adds servers, not engines.
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
              ($FRAMESTORE_NVIMGCODEC_THREADS); baseline JPEGs only,
              frames it cannot decode come back as None. Their RGB output interpolates chroma across MCU borders
              (unlike nvJPEG), so spliced JPEGs use planes(), whose chroma a kernel upsamples per MCU.
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

    def planes(self, frames, sub, stream=None):
        """The NVJPG engines' decode of frames with chroma subsampling sub = (sx, sy) (2, 2: 4:2:0; 2, 1: 4:2:2;
        1, 2: 4:4:0; 1, 1: 4:4:4), as (Y, Cb, Cr) 2-D uint8 CUDA tensors per frame (None where they fail), chroma at
        its stored resolution: Y + 1.402 (Cr - 128) etc. with each chroma sample spread over its sx x sy pixels is
        bit for bit nvJPEG's RGB (H100)."""
        import torch

        nv = self.nv
        css = {
            (2, 2): nv.ChromaSubsampling.CSS_420,
            (2, 1): nv.ChromaSubsampling.CSS_422,
            (1, 2): nv.ChromaSubsampling.CSS_440,
            (1, 1): nv.ChromaSubsampling.CSS_444,
        }[tuple(sub)]
        params = nv.DecodeParams(
            color_spec=nv.ColorSpec.SYCC, sample_format=nv.SampleFormat.P_YUV, chroma_subsampling=css
        )
        s = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        imgs = self.dec.decode([_np(f).tobytes() for f in frames], params=params, cuda_stream=s)
        self.keep = imgs
        return [
            None
            if i is None
            else tuple(
                torch.as_tensor(p, device=self.device).reshape(p.shape[0], p.shape[1]) if len(p.shape) >= 2 else None
                for p in i.planes
            )
            for i in imgs
        ]


def decoders(spec):
    """A decoder spec -> {kind: count}: "nvjpeg:2,hw:1,cpu:4" (a count of "auto" or none: None, the loader's choice).
    Kinds: nvjpeg (torchvision's nvJPEG: the NVJPG engines where the GPU has them, else the CUDA cores), hw (the NVJPG
    engines through nvImageCodec), cpu (libjpeg-turbo)."""
    out = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        kind, _, n = part.partition(":")
        kind = {"tv": "nvjpeg", "nvjpg": "hw", "engines": "hw"}.get(kind, kind)
        if kind not in ("nvjpeg", "hw", "cpu"):
            raise ValueError(f"unknown decoder {kind!r} in {spec!r}: nvjpeg, hw or cpu")
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
