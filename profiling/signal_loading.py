"""Experimental signal-window extraction and CPU/CUDA stage benchmarks.

Run ``python -m profiling.signal_loading --device cpu``. This prototype does
not parse EDF, read storage, resample signals, or change library defaults.
"""

from __future__ import annotations

import argparse
import json
import operator
import platform
import time
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import torch


def extract_windows(column, requests, length: int, channels: int) -> np.ndarray:
    """Copy (row, start) windows from nested Arrow lists into writable float32.

    Supports ragged recording lengths, sliced arrays and multiple chunks.
    Requested windows must have a fixed channel count and contain no nulls.
    Selection order and repeated requests are preserved. No Python signal
    scalars are materialized. Empty requests return (0, length, channels).
    """
    length, channels = operator.index(length), operator.index(channels)
    if length < 1 or channels < 1:
        raise ValueError("length and channels must be positive")
    if isinstance(column, pa.Array):
        column = pa.chunked_array([column])
    if not isinstance(column, pa.ChunkedArray):
        raise TypeError("column must be an Arrow Array or ChunkedArray")

    def is_list(t):
        return pa.types.is_list(t) or pa.types.is_large_list(t)

    if not is_list(column.type) or not is_list(column.type.value_type):
        raise TypeError("expected list<list<numeric>>")
    dtype = column.type.value_type.value_type
    if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
        raise TypeError("signal values must be numeric")
    requests = [(operator.index(row), operator.index(start)) for row, start in requests]
    out = np.empty((len(requests), length, channels), dtype=np.float32)
    ends = np.cumsum([len(chunk) for chunk in column.chunks])
    for i, (row, start) in enumerate(requests):
        if row < 0 or row >= len(column) or start < 0:
            raise IndexError("row/start outside column")
        ci = int(np.searchsorted(ends, row, side="right"))
        local = row - (int(ends[ci - 1]) if ci else 0)
        chunk = column.chunk(ci)
        if not chunk[local].is_valid:
            raise ValueError("recording is null")
        offsets = chunk.offsets
        lo, hi = offsets[local].as_py(), offsets[local + 1].as_py()
        if start + length > hi - lo:
            raise IndexError("window extends beyond recording")
        times = chunk.values.slice(lo + start, length)
        if times.null_count:
            raise ValueError("window contains null time points")
        inner = times.offsets.to_numpy(zero_copy_only=False)
        if np.any(np.diff(inner) != channels):
            raise ValueError("window has an unexpected channel count")
        values = times.values.slice(int(inner[0]), length * channels)
        if values.null_count:
            raise ValueError("window contains null samples")
        out[i] = values.to_numpy(zero_copy_only=False).reshape(length, channels)
    return out


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


def main():
    """Benchmark equivalent synthetic windows; print machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.length, args.channels, args.repeats) < 1 or args.warmup < 0:
        parser.error("sizes/repeats must be positive; warmup must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the NVIDIA driver and CUDA-enabled PyTorch")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    from stable_datasets.formatting import get_formatter
    from stable_datasets.schema import Features, Sequence, Value

    rng = np.random.default_rng(0)
    shape = (args.batch_size, args.length, args.channels)
    digital = rng.integers(-32768, 32768, size=shape, dtype=np.int16)
    c = args.channels
    gain, offset = calibration_parameters(
        np.full(c, -32768), np.full(c, 32767), np.linspace(-200, -100, c), np.linspace(100, 300, c)
    )
    raw = torch.from_numpy(digital)
    g, o = torch.from_numpy(gain), torch.from_numpy(offset)
    physical = calibrate(raw, g, o).numpy()
    features = Features({"series": Sequence(Sequence(Value("float32")))})
    # Setup is excluded from timings. Two chunks deliberately exercise chunk boundaries.
    split = max(1, args.batch_size // 2)
    chunks = [pa.array(physical[:split].tolist(), type=features["series"].to_arrow_type())]
    if split < args.batch_size:
        chunks.append(pa.array(physical[split:].tolist(), type=features["series"].to_arrow_type()))
    column = pa.chunked_array(chunks)
    table = pa.table({"series": column})
    requests = [(int(i), 0) for i in rng.permutation(args.batch_size)]
    indices = [r for r, _ in requests]
    selected = table.take(indices)
    formatter = get_formatter("torch", features)

    def baseline():
        return torch.from_numpy(np.asarray([r["series"] for r in formatter.format_batch(selected)], dtype=np.float32))

    def numeric():
        return torch.from_numpy(extract_windows(column, requests, args.length, c))

    np.testing.assert_array_equal(numeric().numpy(), baseline().numpy())
    reference = digital.astype(np.float64) * gain.astype(np.float64) + offset.astype(np.float64)
    np.testing.assert_allclose(physical, reference, rtol=2e-6, atol=1e-4)
    stages = {
        "cached_python_lists_to_cpu_tensor": baseline,
        "cached_numeric_windows_to_cpu_tensor": numeric,
        "int16_cpu_calibration": lambda: calibrate(raw, g, o),
    }
    setup = {}
    if device.type == "cuda":
        # Both transfer paths use the same selected windows and prepared pinned inputs.
        start = time.perf_counter()
        pinned_raw = torch.from_numpy(digital[indices].copy()).pin_memory()
        pinned_float = torch.from_numpy(physical[indices].copy()).pin_memory()
        setup["prepare_both_pinned_inputs_ms"] = (time.perf_counter() - start) * 1000
        dg, do = g.to(device), o.to(device)

        def transfer_and_calibrate():
            return calibrate(pinned_raw.to(device, non_blocking=True), dg, do)

        actual = transfer_and_calibrate().cpu().numpy()
        np.testing.assert_allclose(actual, reference[indices], rtol=2e-6, atol=1e-4)
        stages["prepared_float32_transfer"] = lambda: pinned_float.to(device, non_blocking=True)
        stages["prepared_int16_transfer_and_gpu_calibration"] = transfer_and_calibrate
        # Include CPU calibration + pinning + transfer for a complete batch-level comparison.
        stages["cpu_calibrate_pin_transfer"] = lambda: calibrate(raw, g, o).pin_memory().to(device, non_blocking=True)
        stages["int16_pin_transfer_gpu_calibrate"] = lambda: calibrate(
            raw.pin_memory().to(device, non_blocking=True), dg, do
        )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    results = {name: measure_stage(fn, device, args.warmup, args.repeats) for name, fn in stages.items()}
    report = {
        "scope": "synthetic in-memory stages; no EDF parsing, disk I/O, resampling or training",
        "shape": shape,
        "device": str(device),
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pyarrow": pa.__version__, "torch": torch.__version__},
        "torch_cpu_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "float32_batch_bytes": int(physical.nbytes),
        "int16_batch_bytes": int(digital.nbytes),
        "correctness": "CPU paths match; CUDA path checked when requested",
        "setup": setup,
        "stages": results,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory}
        report["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
