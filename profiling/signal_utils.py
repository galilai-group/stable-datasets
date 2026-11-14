"""Shared numerical and measurement helpers for signal-format experiments.

Only NumPy and PyTorch are required; importing these helpers never loads a
source reader, Arrow prototype or processed-cache implementation. Physical
signal comparisons use volts. Existing prototype modules re-export the helpers
so their import paths and command-line interfaces remain compatible.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


# The absolute tolerance is 0.1 nV, below the reference recording's 1 uV step.
SIGNAL_RELATIVE_TOLERANCE = 2e-6
SIGNAL_ABSOLUTE_TOLERANCE_VOLTS = 1e-10
VOLTAGE_UNIT_SCALES = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6}


def checksum(path: Path) -> str:
    """Return a streaming SHA256 digest without loading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    if samples.dtype != torch.int16 or samples.ndim < 2:
        raise ValueError("samples must be int16 with a final channel dimension")
    if gain.ndim != 1 or gain.shape != offset.shape or gain.numel() != samples.shape[-1]:
        raise ValueError("gain and offset must match the channel count")
    if gain.dtype != torch.float32 or offset.dtype != torch.float32:
        raise ValueError("gain and offset must be float32")
    if samples.device != gain.device or samples.device != offset.device:
        raise ValueError("samples, gain and offset must use the same device")
    return samples.to(torch.float32) * gain + offset


def measure_stage(
    operation: Callable[[], object], device: torch.device, warmup: int, repeats: int
) -> dict[str, float]:
    """Measure completed-call median/p95 latency in milliseconds.

    Warmups are excluded. CUDA synchronization brackets each timed call;
    allocations inside the operation are included, but result disposal is not.
    This measures batch latency rather than overlapping pipeline throughput.
    """
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
        del result
    return {"median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95))}


def require_matching_signals(actual: NDArray, expected: NDArray) -> None:
    """Abort on sample/shape differences beyond the experiment's volt tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=SIGNAL_RELATIVE_TOLERANCE, atol=SIGNAL_ABSOLUTE_TOLERANCE_VOLTS)
