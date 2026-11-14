"""Compare paired signal caches with actual forward/backward/optimizer work.

This is a warm-cache systems experiment, not a seizure detection model. A small
Conv1D regressor predicts per-channel window means. All paths share windows,
initial weights and SGD settings. Resident GPU inputs isolate model throughput;
cache paths include gathering, transfer and optional calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from profiling.signal_cache import SignalCache
from profiling.signal_utils import calibrate, checksum


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed workload dimensions and deterministic repetition settings."""

    cache_directory: Path
    output: Path
    batch_size: int = 32
    length: int = 1024
    steps: int = 30
    repeats: int = 3
    warmup: int = 5
    cpu_threads: int = 1
    seed: int = 0
    order_seed: int = 0


def profile_region(name: str, enabled: bool):
    """Add named CPU/GPU correlation ranges only for separate profiler runs."""
    return torch.profiler.record_function(name) if enabled else nullcontext()


class SignalRegressor(nn.Module):
    """Predict channel means from (batch, time, channels) signals in volts.

    Shared conversion to mV keeps loss magnitudes practical; it is included in
    every path's training work. Auxiliary channels are retained as in the source
    builder. No clinical labels or accuracy claims are involved.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 64, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, channels),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return (batch, channels) predictions in mV."""
        return self.encoder(samples.transpose(1, 2).contiguous() * 1000)


class PinnedSignalLoader:
    """Gather into two reusable pinned buffers and transfer on a dedicated stream.

    A completion event protects host-buffer reuse. Compute events protect GPU
    buffer reuse. Current-stream waits allow transfer/calibration of the next
    batch to overlap model work. The serialized baseline synchronizes each step
    using exactly the same loader and allocations.
    """

    def __init__(
        self, cache: SignalCache, representation: str, batch_size: int, length: int, *, profile_ranges: bool = False
    ):
        if representation not in ("float32", "int16"):
            raise ValueError("Unknown representation")
        self.cache = cache
        self.representation = representation
        self.profile_ranges = profile_ranges
        self.stream = torch.cuda.Stream()
        shape = (batch_size, length, cache.physical.shape[1])
        dtype = torch.float32 if representation == "float32" else torch.int16
        self.host = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(2)]
        self.device = [torch.empty(shape, dtype=dtype, device="cuda") for _ in range(2)]
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.consumed = [torch.cuda.Event() for _ in range(2)]
        self.used = [False, False]
        self.gain = torch.from_numpy(cache.gain).to("cuda")
        self.offset = torch.from_numpy(cache.offset).to("cuda")
        # New streams must see initialization on the allocating/default stream.
        self.stream.wait_stream(torch.cuda.current_stream())

    def stage(self, starts: np.ndarray, slot: int) -> torch.Tensor:
        """Gather and enqueue one batch; return a tensor ready on the caller stream."""
        with profile_region("signal.host_buffer_wait", self.profile_ranges):
            if self.used[slot]:
                self.ready[slot].synchronize()
        with profile_region("signal.gather", self.profile_ranges):
            self.cache.gather(self.representation, starts, self.host[slot].numpy())
        with torch.cuda.stream(self.stream):
            if self.used[slot]:
                self.stream.wait_event(self.consumed[slot])
            with profile_region("signal.h2d", self.profile_ranges):
                self.device[slot].copy_(self.host[slot], non_blocking=True)
            samples = self.device[slot]
            if self.representation == "int16":
                with profile_region("signal.calibration", self.profile_ranges):
                    samples = calibrate(samples, self.gain, self.offset)
            self.ready[slot].record(self.stream)
        consumer = torch.cuda.current_stream()
        consumer.wait_event(self.ready[slot])
        samples.record_stream(consumer)
        self.used[slot] = True
        return samples

    def release(self, slot: int) -> None:
        """Mark completion after backward/optimizer operations have been queued."""
        self.consumed[slot].record(torch.cuda.current_stream())


def training_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, samples: torch.Tensor, *, profile_ranges: bool = False
) -> torch.Tensor:
    """Run one mean-regression update without synchronizing for scalar logging."""
    optimizer.zero_grad(set_to_none=True)
    with profile_region("signal.forward_and_target", profile_ranges):
        prediction = model(samples)
        target = samples.mean(dim=1) * 1000
        loss = nn.functional.mse_loss(prediction, target)
    with profile_region("signal.backward", profile_ranges):
        loss.backward()
    with profile_region("signal.optimizer", profile_ranges):
        optimizer.step()
    return loss.detach()


def build_schedule(cache: SignalCache, config: TrainingConfig) -> np.ndarray:
    """Select deterministic shuffled windows, including recording edges."""
    last_start = len(cache.physical) - config.length
    if last_start < 0:
        raise ValueError("Window length exceeds recording")
    starts = np.random.default_rng(config.seed).integers(0, last_start + 1, size=(config.steps, config.batch_size))
    starts.flat[0] = 0
    starts.flat[-1] = last_start
    return starts


def validate_training_paths(cache: SignalCache, config: TrainingConfig, schedule: np.ndarray) -> None:
    """Check input values, loss and updated model parameters before timing.

    All cached samples were CPU-validated on open. Here the first scheduled batch
    additionally exercises reusable buffers and the actual CUDA calibration/model.
    """
    outcomes = []
    for representation in ("float32", "int16"):
        torch.manual_seed(config.seed)
        model = SignalRegressor(cache.physical.shape[1]).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = PinnedSignalLoader(cache, representation, config.batch_size, config.length)
        samples = loader.stage(schedule[0], 0)
        input_values = samples.detach().cpu().clone()
        loss = training_step(model, optimizer, samples)
        loader.release(0)
        torch.cuda.synchronize()
        outcomes.append(
            (input_values, loss.cpu(), [parameter.detach().cpu().clone() for parameter in model.parameters()])
        )
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=2e-6, atol=1e-10)
    torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=2e-5, atol=1e-7)
    for float_parameter, integer_parameter in zip(outcomes[0][2], outcomes[1][2]):
        torch.testing.assert_close(float_parameter, integer_parameter, rtol=2e-5, atol=1e-7)


def run_trial(
    cache: SignalCache,
    config: TrainingConfig,
    schedule: np.ndarray,
    representation: str,
    *,
    overlap: bool,
    resident: list[torch.Tensor] | None = None,
    profile_ranges: bool = False,
) -> dict[str, float]:
    """Time a complete synchronized sequence including cache work and SGD.

    Model, optimizer, stream and pinned buffers are setup outside timing. Warmup
    updates use the same prefix for every path. Only the timed region contributes
    to steps/second; final loss inspection happens afterward.
    """
    torch.manual_seed(config.seed)
    model = SignalRegressor(cache.physical.shape[1]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loader = (
        None
        if resident is not None
        else PinnedSignalLoader(cache, representation, config.batch_size, config.length, profile_ranges=profile_ranges)
    )

    def execute(step_count: int) -> torch.Tensor:
        for index in range(step_count):
            slot = index % 2
            samples = resident[index] if resident is not None else loader.stage(schedule[index], slot)
            loss = training_step(model, optimizer, samples, profile_ranges=profile_ranges)
            if loader is not None:
                loader.release(slot)
            if not overlap:
                torch.cuda.synchronize()
        return loss

    if config.warmup:
        execute(config.warmup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_at_unix_s = time.time()
    with profile_region("signal.measured_sequence", profile_ranges):
        started = time.perf_counter()
        loss = execute(config.steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    ended_at_unix_s = time.time()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    return {
        "steps_per_second": config.steps / elapsed,
        "elapsed_seconds": elapsed,
        "final_loss": loss.item(),
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "started_at_unix_s": started_at_unix_s,
        "ended_at_unix_s": ended_at_unix_s,
    }


def prepare_resident_batches(cache: SignalCache, config: TrainingConfig, schedule: np.ndarray) -> list[torch.Tensor]:
    """Place the complete float32 window schedule on GPU outside timed regions.

    These tensors stay allocated during all paths so device memory pressure is
    comparable. They are a model-throughput reference, not a feasible loader for
    arbitrarily large datasets. Storage scales with steps * batch * length.
    """
    buffer = np.empty((config.batch_size, config.length, cache.physical.shape[1]), dtype=np.float32)
    resident = []
    for starts in schedule:
        cache.gather("float32", starts, buffer)
        resident.append(torch.from_numpy(buffer).to("cuda", copy=True))
    torch.cuda.synchronize()
    return resident


def measure_training_paths(
    cache: SignalCache, config: TrainingConfig, schedule: np.ndarray, resident: list[torch.Tensor]
) -> dict[str, list[dict[str, float]]]:
    """Rotate path order across trials to reduce systematic ordering bias."""
    paths = ["resident", "float32_serial", "int16_serial", "float32_overlap", "int16_overlap"]
    results = {name: [] for name in paths}
    base_order = np.random.default_rng(config.order_seed).permutation(paths).tolist()
    for repeat in range(config.repeats):
        trial_order = base_order[repeat % len(paths) :] + base_order[: repeat % len(paths)]
        for position, name in enumerate(trial_order):
            trial = run_trial(
                cache,
                config,
                schedule,
                "int16" if name.startswith("int16") else "float32",
                overlap=name == "resident" or name.endswith("overlap"),
                resident=resident if name == "resident" else None,
            )
            trial.update({"repeat_index": repeat, "position": position})
            results[name].append(trial)
    return results


def build_training_report(
    cache: SignalCache, config: TrainingConfig, schedule: np.ndarray, results: dict[str, list[dict[str, float]]]
) -> dict:
    """Record workload identity, code digests and every trial without timing I/O."""
    return {
        "scope": "One warm memory-mapped recording and a small mean-regression model; no clinical accuracy claim",
        "config": {**asdict(config), "cache_directory": str(config.cache_directory), "output": str(config.output)},
        "manifest_sha256": checksum(config.cache_directory / "manifest.json"),
        "code_sha256": {
            "signal_training.py": checksum(Path(__file__)),
            "signal_cache.py": checksum(Path(__file__).with_name("signal_cache.py")),
            "signal_loading.py": checksum(Path(__file__).with_name("signal_loading.py")),
            "signal_utils.py": checksum(Path(__file__).with_name("signal_utils.py")),
        },
        "model": {
            "name": "SignalRegressor",
            "hidden_channels": 64,
            "optimizer": "SGD",
            "learning_rate": 1e-3,
            "target": "per-channel window means in mV",
        },
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "shape": [config.batch_size, config.length, cache.physical.shape[1]],
        "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "correctness": "Full cache CPU calibration, first-batch CUDA input/loss/updated parameters passed",
        "results": {
            name: {
                "median_steps_per_second": float(np.median([trial["steps_per_second"] for trial in trials])),
                "trials": trials,
            }
            for name, trials in results.items()
        },
    }


def run_comparison(config: TrainingConfig) -> dict:
    """Coordinate validated resident/serialized/overlap trials on one CUDA device."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; run outside a GPU-restricted sandbox")
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = SignalCache.open(config.cache_directory)
    schedule = build_schedule(cache, config)
    validate_training_paths(cache, config, schedule)
    resident = prepare_resident_batches(cache, config, schedule)
    results = measure_training_paths(cache, config, schedule, resident)
    return build_training_report(cache, config, schedule, results)


def parse_arguments() -> TrainingConfig:
    """Parse bounded workload settings and preserve existing benchmark outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (
        ("batch-size", 32),
        ("length", 1024),
        ("steps", 30),
        ("repeats", 3),
        ("warmup", 5),
        ("cpu-threads", 1),
        ("seed", 0),
        ("order-seed", 0),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    arguments = parser.parse_args()
    if min(arguments.batch_size, arguments.length, arguments.steps, arguments.repeats, arguments.cpu_threads) < 1:
        parser.error("Sizes, repeats and CPU threads must be positive")
    if not 0 <= arguments.warmup <= arguments.steps:
        parser.error("Warmup must be between zero and steps")
    if arguments.output.exists():
        parser.error("Output exists; choose a new filename")
    return TrainingConfig(**vars(arguments))


def main() -> None:
    """Run correctness checks and training measurements, then publish JSON."""
    config = parse_arguments()
    report = run_comparison(config)
    with config.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps({name: result["median_steps_per_second"] for name, result in report["results"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
