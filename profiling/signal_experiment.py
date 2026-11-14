"""Run a fixed signal-training protocol with telemetry and separate CUDA traces.

No power/clock settings or other processes are changed. All output goes to a new
directory. Individual training reports retain trial order and wall-clock bounds;
raw nvidia-smi samples make changing GPU conditions visible rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from profiling.signal_cache import SignalCache
from profiling.signal_training import (
    SignalRegressor,
    TrainingConfig,
    build_schedule,
    run_comparison,
    run_trial,
    training_step,
)
from profiling.signal_utils import checksum


TELEMETRY_FIELDS = (
    "timestamp",
    "index",
    "pstate",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "utilization.gpu",
)


@dataclass(frozen=True)
class ExperimentProtocol:
    """Fixed experiment settings chosen before observing performance results."""

    cache_directory: Path
    output_dir: Path
    steps: int = 1000
    repeats: int = 10
    rounds: int = 2
    warmup: int = 50
    cpu_threads: int = 4
    profile_steps: int = 20
    order_seed_offset: int = 0
    precondition_seconds: float = 0.0


def precondition_gpu(protocol: ExperimentProtocol) -> None:
    """Apply a fixed-duration model workload before measuring a block.

    This establishes repeatable preparation, not a guarantee of thermal steady
    state. Synchronizing each update bounds queued work. Timed trials construct
    fresh models and reset their seeds afterward.
    """
    if protocol.precondition_seconds <= 0:
        return
    torch.set_num_threads(protocol.cpu_threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = SignalCache.open(protocol.cache_directory)
    samples = torch.from_numpy(np.array(cache.physical[:1024], copy=True)).unsqueeze(0).repeat(32, 1, 1).cuda()
    model = SignalRegressor(cache.physical.shape[1]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    while True:
        loss = training_step(model, optimizer, samples)
        torch.cuda.synchronize()
        if time.perf_counter() - started >= protocol.precondition_seconds:
            break
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite preconditioning loss")


def write_json(path: Path, value: dict) -> None:
    """Publish a new JSON artifact without replacing existing results."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def host_snapshot() -> dict:
    """Record power/governor/load state without attempting to control the host."""
    power = {}
    for path in Path("/sys/class/power_supply").glob("*/*"):
        if path.name in ("online", "status"):
            power[str(path)] = path.read_text().strip()
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=True)
    return {
        "unix_s": time.time(),
        "host_timezone": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "power_supply": power,
        "cpu_governor": governor.read_text().strip() if governor.exists() else None,
        "nvidia_smi": gpu.stdout,
    }


class GpuTelemetry:
    """Sample supported NVML fields at 500 ms via one owned nvidia-smi process."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def __enter__(self):
        self.output = (self.output_dir / "gpu-telemetry.csv").open("x")
        self.errors = (self.output_dir / "gpu-telemetry.stderr").open("x")
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "500",
                ],
                stdout=self.output,
                stderr=self.errors,
                text=True,
            )
        except BaseException:
            self.output.close()
            self.errors.close()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.output.close()
        self.errors.close()


def summarize_telemetry(path: Path) -> dict:
    """Summarize raw GPU-0 samples; these include setup and are not per-trial utilization."""
    with path.open() as stream:
        rows = [row for row in csv.reader(stream) if len(row) == len(TELEMETRY_FIELDS) and row[1].strip() == "0"]
    summary = {"sample_count": len(rows), "interval_ms": 500, "fields": list(TELEMETRY_FIELDS)}
    for column, name in (
        (3, "temperature_c"),
        (4, "sm_clock_mhz"),
        (5, "memory_clock_mhz"),
        (6, "gpu_utilization_percent"),
    ):
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except ValueError:
                continue
        summary[name] = (
            {"min": min(values), "median": float(np.median(values)), "max": max(values)} if values else None
        )
    return summary


def summarize_round(report: dict) -> dict:
    """Check every final loss and describe ratios without significance claims.

    Ratios pair float32/int16 trials by repetition block, not by simultaneous
    execution. Their min/max is descriptive, not a confidence interval.
    """
    results = report["results"]
    expected_loss = results["resident"]["trials"][0]["final_loss"]
    for result in results.values():
        np.testing.assert_allclose(
            [trial["final_loss"] for trial in result["trials"]], expected_loss, rtol=2e-5, atol=1e-7
        )
    float_trials = results["float32_overlap"]["trials"]
    integer_trials = results["int16_overlap"]["trials"]
    ratios = [
        integer["steps_per_second"] / physical["steps_per_second"]
        for physical, integer in zip(float_trials, integer_trials)
    ]
    return {
        "median_steps_per_second": {name: result["median_steps_per_second"] for name, result in results.items()},
        "overlap_ratio_of_medians": results["int16_overlap"]["median_steps_per_second"]
        / results["float32_overlap"]["median_steps_per_second"],
        "paired_block_ratios": ratios,
        "paired_ratio_median": float(np.median(ratios)),
        "paired_ratio_min": min(ratios),
        "paired_ratio_max": max(ratios),
        "integer_faster_blocks": sum(ratio > 1 for ratio in ratios),
        "blocks": len(ratios),
        "all_final_losses_match": True,
    }


def run_measurements(protocol: ExperimentProtocol) -> list[dict]:
    """Run complete rounds with the same data seed and distinct balanced order seeds."""
    summaries = []
    with GpuTelemetry(protocol.output_dir):
        precondition_gpu(protocol)
        for round_index in range(protocol.rounds):
            print(f"Measurement round {round_index + 1}/{protocol.rounds}", flush=True)
            config = TrainingConfig(
                cache_directory=protocol.cache_directory,
                output=protocol.output_dir / f"round-{round_index + 1}.json",
                steps=protocol.steps,
                repeats=protocol.repeats,
                warmup=protocol.warmup,
                cpu_threads=protocol.cpu_threads,
                order_seed=100 + protocol.order_seed_offset + round_index,
            )
            report = run_comparison(config)
            summary = summarize_round(report)
            write_json(config.output, report)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    return summaries


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a sorted union of trace intervals in microseconds."""
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def overlap_duration(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    """Compute intersection length for two sorted interval unions."""
    first = second = 0
    duration = 0.0
    while first < len(left) and second < len(right):
        duration += max(0.0, min(left[first][1], right[second][1]) - max(left[first][0], right[second][0]))
        if left[first][1] < right[second][1]:
            first += 1
        else:
            second += 1
    return duration


def summarize_trace(path: Path) -> dict:
    """Measure observed H2D/compute-kernel overlap inside the marked sequence.

    Kernels on H2D streams (including calibration) are excluded from model
    kernels. This requires separate transfer/compute streams, as used here.
    CPU ranges show wall durations including API waits; they are not exclusive
    CPU usage or directly additive to GPU timings.
    """
    events = json.loads(path.read_text())["traceEvents"]
    measured = [
        event for event in events if event.get("name") == "signal.measured_sequence" and event.get("ph") == "X"
    ]
    if len(measured) != 1:
        raise ValueError("Expected one marked profiler sequence")
    start, stop = measured[0]["ts"], measured[0]["ts"] + measured[0]["dur"]
    selected = [
        event
        for event in events
        if event.get("ph") == "X" and start <= event["ts"] and event["ts"] + event.get("dur", 0) <= stop
    ]
    copies = [event for event in selected if event.get("cat") == "gpu_memcpy" and "HtoD" in event["name"]]
    copy_streams = {event.get("args", {}).get("stream") for event in copies}
    kernels = [event for event in selected if event.get("cat") == "kernel"]
    compute = [event for event in kernels if event.get("args", {}).get("stream") not in copy_streams]
    transfers = merge_intervals([(event["ts"], event["ts"] + event["dur"]) for event in copies])
    computation = merge_intervals([(event["ts"], event["ts"] + event["dur"]) for event in compute])
    transfer_us = sum(stop - start for start, stop in transfers)
    overlap_us = overlap_duration(transfers, computation)
    cpu_ranges = {}
    for event in selected:
        name = event.get("name", "")
        if name.startswith("signal."):
            cpu_ranges[name] = cpu_ranges.get(name, 0.0) + event["dur"] / 1000
    return {
        "gpu_events_available": bool(copies and compute),
        "h2d_events": len(copies),
        "model_kernel_events": len(compute),
        "measured_sequence_ms": measured[0]["dur"] / 1000,
        "h2d_union_ms": transfer_us / 1000,
        "h2d_overlapping_model_ms": overlap_us / 1000,
        "fraction_h2d_overlapping_model": overlap_us / transfer_us if transfer_us else None,
        "cpu_range_total_ms": cpu_ranges,
    }


def run_profiles(protocol: ExperimentProtocol) -> dict:
    """Capture short separate traces; never use instrumented throughput as a benchmark."""
    cache = SignalCache.open(protocol.cache_directory)
    config = TrainingConfig(
        protocol.cache_directory,
        protocol.output_dir / "unused.json",
        steps=protocol.profile_steps,
        warmup=min(5, protocol.profile_steps),
        repeats=1,
        cpu_threads=protocol.cpu_threads,
    )
    schedule = build_schedule(cache, config)
    summaries = {}
    for representation in ("float32", "int16"):
        for overlap in (False, True):
            name = f"{representation}_{'overlap' if overlap else 'serial'}"
            print(f"Separate profile: {name}", flush=True)
            path = protocol.output_dir / f"trace-{name}.json"
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profiler:
                run_trial(cache, config, schedule, representation, overlap=overlap, profile_ranges=True)
            profiler.export_chrome_trace(str(path))
            summaries[name] = summarize_trace(path)
    return summaries


def run_experiment(protocol: ExperimentProtocol) -> None:
    """Persist the protocol first, then measurements, diagnostics and a summary."""
    protocol.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        protocol.output_dir / "protocol.json",
        {
            **asdict(protocol),
            "cache_directory": str(protocol.cache_directory),
            "output_dir": str(protocol.output_dir),
            "runner_sha256": checksum(Path(__file__)),
            "before": host_snapshot(),
            "limitations": "Warm single-record cache; no clock/power controls; no clinical model; profiled timings excluded",
        },
    )
    rounds = run_measurements(protocol)
    after = host_snapshot()
    profiles = run_profiles(protocol)
    write_json(
        protocol.output_dir / "summary.json",
        {
            "rounds": rounds,
            "telemetry": summarize_telemetry(protocol.output_dir / "gpu-telemetry.csv"),
            "after_measurements": after,
            "profiles": profiles,
        },
    )


def main() -> None:
    """Parse a fixed protocol and preserve all prior result directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (
        ("steps", 1000),
        ("repeats", 10),
        ("rounds", 2),
        ("warmup", 50),
        ("cpu-threads", 4),
        ("profile-steps", 20),
        ("order-seed-offset", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--precondition-seconds", type=float, default=0.0)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.repeats, arguments.rounds, arguments.cpu_threads, arguments.profile_steps) < 1:
        parser.error("Sizes/counts must be positive")
    if not 0 <= arguments.warmup <= arguments.steps:
        parser.error("Warmup must be between zero and steps")
    if not np.isfinite(arguments.precondition_seconds) or arguments.precondition_seconds < 0:
        parser.error("Preconditioning seconds must be finite and nonnegative")
    run_experiment(ExperimentProtocol(**vars(arguments)))


if __name__ == "__main__":
    main()
