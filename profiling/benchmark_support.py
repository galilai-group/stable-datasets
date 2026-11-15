"""Completed-work timing, source hashing and CUDA transfer for benchmarks."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch


def checksum(path: Path) -> str:
    """Return a streaming SHA256 digest without loading the whole file into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def transfer_to_cuda(samples: np.ndarray) -> torch.Tensor:
    """Include per-batch pinning and H2D transfer; the caller synchronizes."""
    return torch.from_numpy(samples).pin_memory().to("cuda", non_blocking=True)
