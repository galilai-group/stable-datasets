"""Native WFDB-212 source -> GPU volts: library, compiled CPU, hybrid and GPU.

Diagnostic completed-batch latency with warm OS pages, not training throughput.
All source gathering, allocation, pinning and dynamic H2D data transfers are
timed. Headers, constant calibration and JIT compilation are prepared outside.
"""

import argparse
import json
import platform
from pathlib import Path

import numba
import numpy as np
import torch
import triton
import wfdb

from profiling.benchmark_support import checksum, measure_stage, transfer_to_cuda
from profiling.research_environment import collect_environment
from profiling.wfdb212 import Wfdb212Source, unpack_cpu
from profiling.wfdb212_cpu import decode_cpu_parallel, decode_cpu_serial
from profiling.wfdb212_cuda import calibrate_cuda, decode_cuda


def require_equal(actual: np.ndarray, reference: np.ndarray) -> None:
    """Require matching shape, missing-value locations and calibrated volts."""
    if actual.shape != reference.shape:
        raise ValueError("Output shape differs from reference")
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-10, equal_nan=True)


def build_operations(source: Wfdb212Source, starts: np.ndarray, length: int) -> dict:
    """Six matched endpoints, all returning GPU float32 volts with NaN semantics."""
    scale = torch.from_numpy(source.scale).cuda()
    baseline = torch.from_numpy(source.baseline).cuda()

    def library():
        return transfer_to_cuda(source.reference(starts, length, physical=True).astype(np.float32))

    def numpy_cpu():
        return transfer_to_cuda(source.physical(unpack_cpu(source.gather(starts, length))))

    def serial_cpu():
        return transfer_to_cuda(decode_cpu_serial(source.gather(starts, length), source.scale, source.baseline))

    def parallel_cpu():
        return transfer_to_cuda(decode_cpu_parallel(source.gather(starts, length), source.scale, source.baseline))

    def hybrid():
        return calibrate_cuda(transfer_to_cuda(unpack_cpu(source.gather(starts, length))), scale, baseline)

    def gpu():
        return decode_cuda(transfer_to_cuda(source.gather(starts, length)), scale, baseline)

    return {
        "wfdb_library": library,
        "numpy_cpu": numpy_cpu,
        "compiled_cpu_serial": serial_cpu,
        "compiled_cpu_parallel": parallel_cpu,
        "cpu_unpack_gpu_calibrate": hybrid,
        "packed_gpu": gpu,
    }


def measure_operations(operations: dict, rounds: int, repeats: int) -> dict:
    """Rotate a seeded order; six rounds balance all six positions."""
    order = np.random.default_rng(0).permutation(list(operations)).tolist()
    trials = {name: [] for name in operations}
    for round_index in range(rounds):
        rotated = order[round_index % len(order) :] + order[: round_index % len(order)]
        for position, name in enumerate(rotated):
            trials[name].append(
                {
                    **measure_stage(operations[name], torch.device("cuda"), 3, repeats),
                    "round": round_index,
                    "position": position,
                }
            )
    return {
        name: {"median_ms": float(np.median([trial["median_ms"] for trial in values])), "trials": values}
        for name, values in trials.items()
    }


def run_benchmark(record: Path, batch_size: int, length: int, *, rounds: int = 6, repeats: int = 30) -> dict:
    """Validate all exact integers and all physical endpoints before measurement."""
    torch.set_num_threads(1)
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    source = Wfdb212Source(record)
    paths = [record.with_suffix(".hea"), source.data_path]
    hashes = {path.name: checksum(path) for path in paths}
    if batch_size < 3 or not 1 <= length <= source.header.sig_len:
        raise ValueError("Require three or more windows fitting the recording")
    starts = np.random.default_rng(0).integers(0, source.header.sig_len - length + 1, size=batch_size)
    starts[:3] = [0, source.header.sig_len - length, min(1, source.header.sig_len - length)]
    packed = source.gather(starts, length)
    digital = unpack_cpu(packed)
    np.testing.assert_array_equal(digital, source.reference(starts, length, physical=False))
    expected = source.reference(starts, length, physical=True)
    operations = build_operations(source, starts, length)
    errors = {}
    # These checks also compile every JIT specialization outside timed regions.
    for name, operation in operations.items():
        actual = operation().cpu().numpy()
        require_equal(actual, expected)
        errors[name] = float(np.nanmax(np.abs(actual - expected))) if np.isfinite(expected).any() else None
    scale = torch.from_numpy(source.scale).cuda()
    baseline = torch.from_numpy(source.baseline).cuda()
    resident = transfer_to_cuda(packed)
    np.testing.assert_array_equal(decode_cuda(resident, scale, baseline, physical=False).cpu().numpy(), digital)
    results = measure_operations(operations, rounds, repeats)
    diagnostics = {
        "gather_packed_cpu": measure_stage(lambda: source.gather(starts, length), torch.device("cpu"), 3, repeats),
        "prepared_compiled_cpu_serial": measure_stage(
            lambda: decode_cpu_serial(packed, source.scale, source.baseline), torch.device("cpu"), 3, repeats
        ),
        "prepared_gpu_decode": measure_stage(
            lambda: decode_cuda(resident, scale, baseline), torch.device("cuda"), 3, repeats
        ),
    }
    if hashes != {path.name: checksum(path) for path in paths}:
        raise ValueError("Source changed during benchmark")
    return {
        "scope": "Native two-channel WFDB-212 to GPU volts, warm OS cache; no training or cold-storage claim",
        "source_sha256": hashes,
        "source_unchanged": True,
        "shape": [batch_size, length, 2],
        "sample_rate_hz": source.header.fs,
        "sample_count": source.header.sig_len,
        "channels": source.header.sig_name,
        "units": source.header.units,
        "adc_gain": source.header.adc_gain,
        "baseline": source.header.baseline,
        "window_starts": starts.tolist(),
        "packed_bytes_per_batch": packed.nbytes,
        "int16_bytes_per_batch": digital.nbytes,
        "float32_bytes_per_batch": digital.size * 4,
        "selected_missing_samples": int(np.count_nonzero(digital == -2048)),
        "checks": {"digital_bit_exact": True, "physical_values_and_nan_masks": True, "errors_volts": errors},
        "rounds": rounds,
        "repeats": repeats,
        "warmup": 3,
        "torch_threads": 1,
        "compiled_cpu_parallel_threads": numba.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0),
        "environment": collect_environment(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wfdb": wfdb.__version__,
            "numba": numba.__version__,
            "triton": triton.__version__,
        },
        "code_sha256": {
            name: checksum(Path(__file__).with_name(name))
            for name in (
                "wfdb212.py",
                "wfdb212_cpu.py",
                "wfdb212_cuda.py",
                "benchmark_wfdb212.py",
                "benchmark_support.py",
                "research_environment.py",
            )
        },
        "results": results,
        "separate_prepared_input_diagnostics": diagnostics,
    }


def main() -> None:
    """Run a small explicit source-format experiment without replacing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="Local WFDB record path without extension")
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("batch-size", 32), ("length", 1024), ("rounds", 6), ("repeats", 30)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    if min(args.rounds, args.repeats) < 1 or args.batch_size < 3 or args.length < 1:
        parser.error("Require batch-size >= 3 and positive sizes/rounds/repeats")
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; run with GPU access")
    report = run_benchmark(args.record, args.batch_size, args.length, rounds=args.rounds, repeats=args.repeats)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({name: result["median_ms"] for name, result in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
