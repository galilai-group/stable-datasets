"""Whole-frame JPEG decoders.

Decoder("nvjpeg" | "hw" | "cpu")(frames)   frames (uint8 tensors, numpy arrays or bytes) -> (3, h, w) uint8 tensors
decode_resized(jpeg, h, w)                 one frame -> (h, w, 3) uint8, CPU decode + antialiased bilinear resize
"""

from contextlib import nullcontext

import numpy as np


class Decoder:
    """frames (uint8 tensors, numpy arrays or bytes) -> list of (3, h, w) uint8 tensors on the decoder's device.

    "cpu"     libjpeg-turbo via simplejpeg (fancy upsampling), in the calling thread, GIL released
    "nvjpeg"  torchvision decode_jpeg on CUDA: nvJPEG on the CUDA cores, Huffman decoding on this process's CPU
              (per-MCU chroma upsampling, which the splice needs); alias "tv"
    "hw"      nvImageCodec on the NVJPG engines (A100 / H100 / B200, not Ada): no CPU work; baseline JPEGs only,
              frames it cannot decode come back as None
    """

    def __init__(self, kind="nvjpeg", device="cuda"):
        self.kind = {"tv": "nvjpeg"}.get(kind, kind)
        self.device = device
        if self.kind == "hw":
            from nvidia import nvimgcodec as nv

            self.nv = nv
            self.dec = nv.Decoder(backends=[nv.Backend(nv.BackendKind.HW_GPU_ONLY)])
            self.params = nv.DecodeParams(color_spec=nv.ColorSpec.SRGB, sample_format=nv.SampleFormat.P_RGB)

    def __call__(self, frames, stream=None):
        import torch

        if self.kind == "cpu":
            import simplejpeg

            return [
                torch.from_numpy(simplejpeg.decode_jpeg(_np(f), colorspace="RGB")).permute(2, 0, 1) for f in frames
            ]
        if self.kind == "nvjpeg":  # fastest under load on every GPU measured (L40S, RTX 6000 Ada, H100)
            from torchvision.io import ImageReadMode, decode_jpeg

            frames = [f if isinstance(f, torch.Tensor) else torch.from_numpy(_np(f)) for f in frames]
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                return decode_jpeg(frames, mode=ImageReadMode.RGB, device=self.device)
        s = stream.cuda_stream if stream is not None else torch.cuda.current_stream().cuda_stream
        imgs = self.dec.decode([_np(f).tobytes() for f in frames], params=self.params, cuda_stream=s)
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            return [None if i is None else torch.as_tensor(i, device=self.device) for i in imgs]


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
