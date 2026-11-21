"""Affine signal calibration in the units specified by source metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    import torch

VOLTAGE_UNIT_SCALES = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6}


def calibration_parameters(digital_min, digital_max, physical_min, physical_max):
    """Return float32 per-channel gain/offset in the supplied physical units.

    These are EDF-style affine parameters, not unit normalization or a full
    EDF implementation. Physical extrema may be reversed (negative gain).
    """
    bounds = [np.asarray(x, dtype=np.float64) for x in (digital_min, digital_max, physical_min, physical_max)]
    if any(x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() for x in bounds):
        raise ValueError("calibration bounds must be finite nonempty vectors")
    if any(x.shape != bounds[0].shape for x in bounds):
        raise ValueError("calibration vectors must have the same shape")
    dmin, dmax, pmin, pmax = bounds
    if np.any(dmax <= dmin) or np.any(pmax == pmin):
        raise ValueError("calibration ranges must be nonzero with digital_max > digital_min")
    gain = (pmax - pmin) / (dmax - dmin)
    offset = pmin - dmin * gain
    with np.errstate(over="ignore"):
        gain, offset = gain.astype(np.float32), offset.astype(np.float32)
    if not np.isfinite(gain).all() or not np.isfinite(offset).all() or np.any(gain == 0):
        raise ValueError("calibration cannot be represented in float32")
    return gain, offset


def calibrate(samples: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Calibrate an int16 (..., channels) CPU or CUDA tensor to float32."""
    import torch

    if samples.dtype != torch.int16 or samples.ndim < 2:
        raise ValueError("samples must be int16 with a final channel dimension")
    if gain.ndim != 1 or gain.shape != offset.shape or gain.numel() != samples.shape[-1]:
        raise ValueError("gain and offset must match the channel count")
    if gain.dtype != torch.float32 or offset.dtype != torch.float32:
        raise ValueError("gain and offset must be float32")
    if samples.device != gain.device or samples.device != offset.device:
        raise ValueError("samples, gain and offset must use the same device")
    return samples.to(torch.float32) * gain + offset


def calibrate_cpu(digital: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Calibrate to float32 using one allocation and in-place arithmetic."""
    physical = digital.astype(np.float32)
    physical *= gain
    physical += offset
    return physical
