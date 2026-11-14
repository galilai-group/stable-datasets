"""Deterministic protocol ordering and trace interpretation checks."""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from profiling import signal_experiment, signal_training
from profiling.signal_experiment import merge_intervals, overlap_duration, summarize_round, summarize_trace
from profiling.signal_training import TrainingConfig


def test_interval_overlap_counts_union_not_double_counted_kernels():
    left = merge_intervals([(0, 5), (3, 7), (10, 11)])
    right = merge_intervals([(2, 4), (6, 10.5)])
    assert left == [(0, 7), (10, 11)]
    assert overlap_duration(left, right) == 3.5
    assert overlap_duration([], right) == 0


def test_profiler_summary_excludes_warmup_and_calibration_from_model_overlap(tmp_path):
    def event(name, category, start, duration, stream=None):
        return {"name": name, "cat": category, "ph": "X", "ts": start, "dur": duration, "args": {"stream": stream}}

    events = [
        event("signal.measured_sequence", "user_annotation", 100, 100),
        event("Memcpy HtoD", "gpu_memcpy", 10, 10, 1),  # Warmup excluded.
        event("Memcpy HtoD", "gpu_memcpy", 110, 10, 1),
        event("calibrate", "kernel", 120, 5, 1),  # Not model computation.
        event("model", "kernel", 115, 20, 7),
        event("signal.gather", "user_annotation", 100, 5),
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    summary = summarize_trace(path)
    assert summary["gpu_events_available"]
    assert summary["h2d_events"] == 1
    assert summary["model_kernel_events"] == 1
    assert summary["fraction_h2d_overlapping_model"] == 0.5


def test_round_summary_rejects_training_divergence():
    trial = {"steps_per_second": 10.0, "final_loss": 1.0}
    report = {
        "results": {
            name: {"trials": [dict(trial)], "median_steps_per_second": 10.0}
            for name in ("resident", "float32_overlap", "int16_overlap")
        }
    }
    summary = summarize_round(report)
    assert summary["paired_block_ratios"] == [1.0]
    report["results"]["int16_overlap"]["trials"][0]["final_loss"] = 2.0
    with pytest.raises(AssertionError):
        summarize_round(report)


def test_trial_order_is_deterministic_and_balanced(monkeypatch):
    """Each path occupies each position twice in ten repetitions."""
    monkeypatch.setattr(signal_training, "run_trial", lambda *args, **kwargs: {})
    config = TrainingConfig(Path("unused"), Path("unused.json"), repeats=10, order_seed=42)
    results = signal_training.measure_training_paths(None, config, None, [])
    assert results == signal_training.measure_training_paths(None, config, None, [])
    for trials in results.values():
        assert [sum(trial["position"] == position for trial in trials) for position in range(5)] == [2] * 5


def test_measurement_blocks_use_distinct_order_seeds_and_preconditioning(tmp_path, monkeypatch):
    configurations = []
    preconditions = []
    protocol = signal_experiment.ExperimentProtocol(
        tmp_path, tmp_path, rounds=2, order_seed_offset=10, precondition_seconds=10
    )
    monkeypatch.setattr(signal_experiment, "GpuTelemetry", lambda directory: nullcontext())
    monkeypatch.setattr(signal_experiment, "precondition_gpu", preconditions.append)
    monkeypatch.setattr(signal_experiment, "run_comparison", lambda config: configurations.append(config) or {})
    monkeypatch.setattr(signal_experiment, "summarize_round", lambda report: {})
    monkeypatch.setattr(signal_experiment, "write_json", lambda path, report: None)
    signal_experiment.run_measurements(protocol)
    assert preconditions == [protocol]
    assert [config.order_seed for config in configurations] == [110, 111]
    assert [config.seed for config in configurations] == [0, 0]
