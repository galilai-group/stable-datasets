"""Single-kernel WFDB-212 unpack/sign-extension/calibration on CUDA via Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(
    packed,
    scale,
    baseline,
    output,
    count: tl.constexpr,
    physical: tl.constexpr,
    packed_input: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    frame, channel = index // 2, index % 2
    if packed_input:
        low = tl.load(packed + frame * 3, valid, 0).to(tl.int32)
        middle = tl.load(packed + frame * 3 + 1, valid, 0).to(tl.int32)
        high = tl.load(packed + frame * 3 + 2, valid, 0).to(tl.int32)
        unsigned = tl.where(channel == 0, low | ((middle & 15) << 8), high | ((middle >> 4) << 8))
        digital = (unsigned ^ 2048) - 2048
    else:
        digital = tl.load(packed + index, valid, 0).to(tl.int32)
    if physical:
        value = (digital.to(tl.float32) - tl.load(baseline + channel)) * tl.load(scale + channel)
        value = tl.where(digital == -2048, float("nan"), value)
    else:
        value = digital
    tl.store(output + index, value, valid)


def decode_cuda(
    packed: torch.Tensor, scale: torch.Tensor, baseline: torch.Tensor, *, physical: bool = True
) -> torch.Tensor:
    """Decode contiguous CUDA (..., 3) bytes; return (..., 2) volts or exact int16.

    Channel metadata must already reside on the same GPU. Launch, output
    allocation and calibration are included; compilation is warmed before timing.
    """
    if packed.dtype != torch.uint8 or packed.ndim < 2 or packed.shape[-1] != 3 or not packed.is_contiguous():
        raise ValueError("Expected contiguous uint8 packets (..., 3)")
    if not packed.is_cuda or any(
        value.device != packed.device or value.dtype != torch.float32 or value.shape != (2,)
        for value in (scale, baseline)
    ):
        raise ValueError("Require CUDA packets and two float32 calibration values on the same device")
    output = torch.empty(
        (*packed.shape[:-1], 2), device=packed.device, dtype=torch.float32 if physical else torch.int16
    )
    _decode[(triton.cdiv(output.numel(), 256),)](
        packed, scale, baseline, output, output.numel(), physical, True, 256, enable_fp_fusion=False
    )
    return output


def calibrate_cuda(digital: torch.Tensor, scale: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Fused hybrid control: already CPU-unpacked int16 samples -> GPU volts/NaNs."""
    if not digital.is_cuda or digital.dtype != torch.int16 or not digital.is_contiguous() or digital.shape[-1] != 2:
        raise ValueError("Require contiguous CUDA int16 (..., 2)")
    if any(
        value.device != digital.device or value.dtype != torch.float32 or value.shape != (2,)
        for value in (scale, baseline)
    ):
        raise ValueError("Calibration must be two float32 values on the same device")
    output = torch.empty_like(digital, dtype=torch.float32)
    _decode[(triton.cdiv(output.numel(), 256),)](
        digital, scale, baseline, output, output.numel(), True, False, 256, enable_fp_fusion=False
    )
    return output
