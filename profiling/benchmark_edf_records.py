"""Compare source-inclusive EDF window assembly on CPU and GPU.

All paths read the same original file with a warm OS cache. Plan creation,
source access, allocation, pinning and H2D transfers occur inside complete batch
measurements. Reader setup/calibration placement are reused outside timing.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import edfio
import mne
import numpy as np
import torch

from profiling.benchmark_support import checksum, measure_stage, transfer_to_cuda
from profiling.edf_record_loading import BatchedEdfRecords, GpuRecordAssembler
from profiling.edf_reference import read_mne_windows, require_matching_signals, validate_source
from profiling.edf_source import NativeEdfMetadata, NativeEdfReader
from profiling.research_environment import collect_environment
from profiling.signal_calibration import calibrate, calibrate_cpu


def choose_windows(metadata: NativeEdfMetadata, batch_size: int, length: int) -> np.ndarray:
    """Include first/last windows and an EDF-record boundary among seeded requests."""
    last_start = metadata.sample_count - length
    if batch_size < 3 or last_start < 0:
        raise ValueError("Require at least three windows fitting the recording")
    starts = np.random.default_rng(0).integers(0, last_start + 1, size=batch_size)
    starts[:3] = [
        0,
        last_start,
        min(last_start, max(0, round(metadata.record_duration_s * metadata.sample_rate_hz) - 1)),
    ]
    return starts


@dataclass(frozen=True)
class RecordBenchmarkConfig:
    """Fixed batch shape and balanced latency-trial settings."""

    source: Path
    output: Path
    batch_size: int = 32
    length: int = 1024
    repeats: int = 30
    rounds: int = 7


def build_gpu_operations(
    source: NativeEdfReader, records: BatchedEdfRecords, starts: np.ndarray, length: int
) -> dict[str, Callable[[], torch.Tensor]]:
    """Bind seven complete source-to-GPU paths with shared requested windows."""
    metadata = source.metadata
    gain = torch.from_numpy(metadata.gain).cuda()
    offset = torch.from_numpy(metadata.offset).cuda()
    assembler = GpuRecordAssembler(records, torch.device("cuda"))

    def library_cpu_calibration():
        return transfer_to_cuda(calibrate_cpu(source.read_windows(starts, length), metadata.gain, metadata.offset))

    def library_gpu_calibration():
        return calibrate(transfer_to_cuda(source.read_windows(starts, length)), gain, offset)

    def cpu_gather_cpu_calibration():
        digital = records.read_cpu(records.plan(starts, length))
        return transfer_to_cuda(calibrate_cpu(digital, metadata.gain, metadata.offset))

    def cpu_gather_gpu_calibration():
        digital = records.read_cpu(records.plan(starts, length))
        return calibrate(transfer_to_cuda(digital), gain, offset)

    def cpu_records_cpu_calibration():
        digital = records.read_cpu_from_records(records.plan(starts, length))
        return transfer_to_cuda(calibrate_cpu(digital, metadata.gain, metadata.offset))

    def cpu_records_gpu_calibration():
        digital = records.read_cpu_from_records(records.plan(starts, length))
        return calibrate(transfer_to_cuda(digital), gain, offset)

    def gpu_records_gpu_assembly():
        return assembler.assemble(records.plan(starts, length))

    return {
        "library_cpu_calibration": library_cpu_calibration,
        "library_gpu_calibration": library_gpu_calibration,
        "cpu_gather_cpu_calibration": cpu_gather_cpu_calibration,
        "cpu_gather_gpu_calibration": cpu_gather_gpu_calibration,
        "cpu_records_cpu_calibration": cpu_records_cpu_calibration,
        "cpu_records_gpu_calibration": cpu_records_gpu_calibration,
        "gpu_records_gpu_assembly": gpu_records_gpu_assembly,
    }


def measure_gpu_operations(operations: dict[str, Callable[[], torch.Tensor]], config: RecordBenchmarkConfig) -> dict:
    """Rotate a fixed seeded order so seven rounds balance all seven positions."""
    order = np.random.default_rng(0).permutation(list(operations)).tolist()
    results = {name: [] for name in operations}
    for round_index in range(config.rounds):
        rotated = order[round_index % len(order) :] + order[: round_index % len(order)]
        for position, name in enumerate(rotated):
            results[name].append(
                {
                    **measure_stage(operations[name], torch.device("cuda"), 3, config.repeats),
                    "round": round_index,
                    "position": position,
                }
            )
    return {
        name: {
            "median_of_round_medians_ms": float(np.median([trial["median_ms"] for trial in trials])),
            "trials": trials,
        }
        for name, trials in results.items()
    }


def validate_raw_paths(
    source: NativeEdfReader, records: BatchedEdfRecords, starts: np.ndarray, length: int, expected: np.ndarray
) -> dict:
    """Check both CPU layouts exactly and physical values against independent MNE."""
    plan = records.plan(starts, length)
    library = source.read_windows(starts, length)
    direct = records.read_cpu(plan)
    selected = records.read_cpu_from_records(plan)
    np.testing.assert_array_equal(direct, library)
    np.testing.assert_array_equal(selected, library)
    physical = calibrate_cpu(direct, source.metadata.gain, source.metadata.offset)
    require_matching_signals(physical, expected)
    return {"raw_samples_match_library": True, "cpu_max_abs_error_volts": float(np.max(np.abs(physical - expected)))}


def run_benchmark(config: RecordBenchmarkConfig) -> dict:
    """Validate raw offsets and every CUDA path before measuring batch latencies."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; run with GPU access")
    torch.set_num_threads(1)
    source_hash = checksum(config.source)
    source = NativeEdfReader(config.source)
    records = BatchedEdfRecords(config.source, source)
    starts = choose_windows(source.metadata, config.batch_size, config.length)
    with mne.io.read_raw_edf(str(config.source), preload=False, verbose="ERROR") as reference:
        checks = validate_source(source, reference, starts, config.length)
        expected = read_mne_windows(reference, starts, config.length)
    checks.update(validate_raw_paths(source, records, starts, config.length, expected))
    operations = build_gpu_operations(source, records, starts, config.length)
    gpu_errors = {}
    for name, operation in operations.items():
        actual = operation().cpu().numpy()
        require_matching_signals(actual, expected)
        gpu_errors[name] = float(np.max(np.abs(actual - expected)))
    results = measure_gpu_operations(operations, config)
    if checksum(config.source) != source_hash:
        raise ValueError("Source changed during measurement")
    return build_report(config, source, records, starts, checks, gpu_errors, results, source_hash)


def build_report(
    config: RecordBenchmarkConfig,
    source: NativeEdfReader,
    records: BatchedEdfRecords,
    starts: np.ndarray,
    checks: dict,
    gpu_errors: dict,
    results: dict,
    source_hash: str,
) -> dict:
    """Record layouts, transferred payload sizes, versions and complete trial results."""
    return {
        "scope": "Unchanged native EDF, warm OS cache, completed-batch GPU latency; no training/GDS/cold disk claim",
        "source": {"name": config.source.name, "sha256": source_hash},
        "source_unchanged": True,
        "shape": [config.batch_size, config.length, len(source.metadata.channel_names)],
        "window_starts": starts.tolist(),
        "channel_names": list(source.metadata.channel_names),
        "record_layout": {
            "record_count": records.layout.record_count,
            "words_per_record": records.layout.words_per_record,
            "samples_per_channel_record": records.layout.samples_per_channel_record,
            "channel_offsets": records.layout.channel_offsets.tolist(),
        },
        "payload": records.payload_statistics(records.plan(starts, config.length)),
        "checks": checks,
        "gpu_max_abs_errors_volts": gpu_errors,
        "rounds": config.rounds,
        "repeats": config.repeats,
        "warmup": 3,
        "torch_cpu_threads": 1,
        "gpu": torch.cuda.get_device_name(0),
        "environment": collect_environment(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mne": mne.__version__,
            "edfio": edfio.__version__,
        },
        "code_sha256": {
            name: checksum(Path(__file__).with_name(name))
            for name in (
                "benchmark_edf_records.py",
                "edf_record_loading.py",
                "edf_source.py",
                "edf_reference.py",
                "signal_calibration.py",
                "benchmark_support.py",
                "research_environment.py",
            )
        },
        "results": results,
    }


def main() -> None:
    """Parse the diagnostic protocol and preserve all previous artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("batch-size", 32), ("length", 1024), ("repeats", 30), ("rounds", 7)):
        parser.add_argument(f"--{name}", type=int, default=default)
    arguments = parser.parse_args()
    if arguments.batch_size < 3 or min(arguments.length, arguments.repeats, arguments.rounds) < 1:
        parser.error("Require batch-size >= 3 and positive sizes/repeats/rounds")
    if arguments.output.exists():
        parser.error("Output exists; choose a new path")
    config = RecordBenchmarkConfig(**vars(arguments))
    report = run_benchmark(config)
    with config.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {name: result["median_of_round_medians_ms"] for name, result in report["results"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
