"""Small CNN diagnostic for neonatal window loading, workers and data stalls.

This is a single-recording systems experiment with real seizure annotations,
not a patient-generalization evaluation or the later EEG-JEPA adaptation.
Storage and label policy live in separate modules. GPU assembly is unchanged.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from profiling.benchmark_support import checksum
from profiling.neonatal_targets import load_expert_labels, window_targets
from profiling.research_environment import collect_environment
from profiling.training_windows import CPU_PATHS, GPU_PATH, WindowBatches, configure_reader_threads


class SmallEegCnn(nn.Module):
    """Classify EEG channels after identical fixed volts-to-scale conversion."""

    def __init__(self, channels: tuple[str, ...]):
        super().__init__()
        indices = [i for i, name in enumerate(channels) if name.startswith("EEG")]
        if not indices:
            raise ValueError("Require explicitly named EEG channels")
        self.register_buffer("eeg_indices", torch.tensor(indices, dtype=torch.long))
        self.network = nn.Sequential(
            nn.Conv1d(len(indices), 16, 7, stride=4, padding=3),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, values):
        return self.network(values.index_select(2, self.eeg_indices).transpose(1, 2) * 1e4)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _new_model(channels, device):
    torch.manual_seed(0)
    model = SmallEegCnn(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    return model, optimizer


class TrialFailure(RuntimeError):
    """Keep failed-step context and partial observations separate from valid trials."""

    def __init__(self, context, error):
        self.context = context
        super().__init__(
            f"{context['path']} workers={context['workers']} failed during {context['phase']} "
            f"at batch {context['batch_number']}/{context['total_batches']} "
            f"after {len(context['completed_steps'])} completed steps: {error}"
        )


def train_trial(dataset: WindowBatches, workers: int, device: torch.device) -> dict:
    """Include worker startup and synchronized data-ready and training steps.

    Worker processes prefetch during compute. Synchronizing at step boundaries
    makes measured data-ready gaps visible, but does not measure a fully
    asynchronous training pipeline. Reader shutdown is outside the interval.
    """
    if workers < 0 or (dataset.path == GPU_PATH and (workers or device.type != "cuda")):
        raise ValueError("Existing GPU assembly requires CUDA and zero workers")
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=configure_reader_threads,
        multiprocessing_context="spawn" if workers else None,
        prefetch_factor=2 if workers else None,
        timeout=60 if workers else 0,
        generator=torch.Generator().manual_seed(0),
    )
    model, optimizer = _new_model(dataset.metadata.channels, device)
    assembler = None
    if dataset.path == GPU_PATH:
        from profiling.edf_record_loading import BatchedEdfRecords, GpuRecordAssembler
        from profiling.edf_source import NativeEdfReader

        assembler = GpuRecordAssembler(BatchedEdfRecords(dataset.source, NativeEdfReader(dataset.source)), device)
    _synchronize(device)
    started = perf_counter()
    iterator = None
    iterator_seconds = None
    phase = "starting_iterator"
    batch_number = 0
    steps = []
    try:
        iterator = iter(loader)
        iterator_seconds = perf_counter() - started
        for index in range(len(dataset)):
            batch_number = index + 1
            phase = "waiting_for_batch"
            ready_started = perf_counter()
            batch = next(iterator)
            wait_seconds = perf_counter() - ready_started
            phase = "preparing_device_batch"
            if assembler is None:
                values = batch["values"].to(device, non_blocking=True)
            else:
                values = assembler.assemble(assembler.source.plan(batch["starts"].numpy(), dataset.length))
            targets = batch["targets"].to(device, non_blocking=True)
            _synchronize(device)
            ready_seconds = perf_counter() - ready_started
            phase = "training_step"
            compute_started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(values), targets)
            loss.backward()
            optimizer.step()
            _synchronize(device)
            compute_seconds = perf_counter() - compute_started
            steps.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "elapsed_seconds": perf_counter() - started,
                    "loader_wait_seconds": wait_seconds,
                    "data_ready_seconds": ready_seconds,
                    "compute_seconds": compute_seconds,
                }
            )
        elapsed = perf_counter() - started
    except Exception as error:
        raise TrialFailure(
            {
                "path": dataset.path,
                "workers": workers,
                "phase": phase,
                "batch_number": batch_number,
                "total_batches": len(dataset),
                "elapsed_seconds": perf_counter() - started,
                "iterator_startup_seconds": iterator_seconds,
                "completed_steps": steps,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            error,
        ) from error
    finally:
        if iterator is not None:
            del iterator
    if not all(np.isfinite(row["loss"]) for row in steps):
        raise ValueError("Nonfinite training loss")
    return {
        "path": dataset.path,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "windows_per_second": dataset.starts.size / elapsed,
        "iterator_startup_seconds": iterator_seconds,
        "data_ready_fraction": sum(row["data_ready_seconds"] for row in steps) / elapsed,
        "steps": steps,
    }


def run(
    source,
    workspace,
    annotations,
    *,
    subject_id,
    batch_size,
    steps,
    seconds,
    workers,
    trials,
    device,
    paths=None,
    progress=None,
):
    """Validate the first batch against MNE, then compare matched learning runs."""
    import mne

    from profiling.edf_reference import read_mne_windows

    if min(batch_size, steps, trials) < 1 or not workers or any(w < 0 for w in workers):
        raise ValueError("Require positive sizes and nonnegative worker counts")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Require positive finite window duration")
    if source.name != f"eeg{subject_id}.edf":
        raise ValueError("EDF filename must agree with the annotated subject ID")
    prep = json.loads((workspace / "preparation.json").read_text())
    source_hash = checksum(source)
    annotation_hash = checksum(annotations)
    if source_hash != prep["source_sha256"]:
        raise ValueError("EDF differs from the prepared source")
    configure_reader_threads()
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Require exactly one visible CUDA GPU")
        if os.environ.get("SLURM_JOB_ID") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Missing Slurm GPU assignment")
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching CUDA")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    meta = prep["recording"]
    rate = float(meta["rate_hz"])
    length = round(seconds * rate)
    if rate != int(rate) or seconds != int(seconds) or length > meta["n_samples"]:
        raise ValueError("Diagnostic windows require whole seconds at an integral rate")
    experts = load_expert_labels(annotations, subject_id)
    if experts.shape[1] < int(np.ceil(meta["n_samples"] / rate)):
        raise ValueError("Annotations do not cover the recording")
    latest = meta["n_samples"] // int(rate) - int(seconds)
    starts = np.random.default_rng(0).integers(0, latest + 1, size=(steps, batch_size)) * int(rate)
    labels = window_targets(experts, starts, length, rate)
    available = (*CPU_PATHS, GPU_PATH) if target.type == "cuda" else CPU_PATHS
    selected = list(dict.fromkeys(available if paths is None else paths))
    if not selected or any(name not in available for name in selected):
        raise ValueError("Require loading paths available on the selected device")
    cases = [(name, count) for name in selected for count in ([0] if name == GPU_PATH else sorted(set(workers)))]
    with mne.io.read_raw_edf(str(source), preload=False, verbose="ERROR") as raw:
        expected = read_mne_windows(raw, starts[0], length)
    datasets = {name: WindowBatches(source, workspace, starts, labels, length, name) for name, _ in cases}
    errors = {}
    for name, dataset in datasets.items():
        if name == GPU_PATH:
            from profiling.edf_record_loading import BatchedEdfRecords, GpuRecordAssembler
            from profiling.edf_source import NativeEdfReader

            reader = BatchedEdfRecords(source, NativeEdfReader(source))
            actual = GpuRecordAssembler(reader, target).assemble(reader.plan(starts[0], length)).cpu().numpy()
        else:
            actual = dataset[0]["values"]
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10)
        errors[name] = float(np.max(np.abs(actual - expected)))
    # Warm model forward/backward kernels using an independent model, then reset
    # weights and AdamW state for every measured trial.
    warm_model, warm_optimizer = _new_model(tuple(meta["channels"]), target)
    warm_values = torch.from_numpy(expected.astype(np.float32)).to(target)
    warm_targets = torch.from_numpy(labels[0]).to(target)
    for _ in range(2):
        warm_optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(warm_model(warm_values), warm_targets).backward()
        warm_optimizer.step()
    _synchronize(target)
    parameters = sum(p.numel() for p in warm_model.parameters())
    del warm_model, warm_optimizer, warm_values, warm_targets
    metadata = {
        "scope": "Single-recording CNN data-stall diagnostic; synchronized steps; no held-out accuracy claim",
        "cache_condition": "No eviction or residency verification; first-batch validation and repeated schedules affect caching",
        "label_policy": "Majority of three experts per second; any positive second makes the whole-second window positive",
        "subject_id": subject_id,
        "device": str(target),
        "planned_trials": trials,
        "cases": [{"path": name, "workers": count} for name, count in cases],
        "source_sha256": source_hash,
        "annotations_sha256": annotation_hash,
        "model": "Two Conv1d/GroupNorm/GELU blocks and global pooling; EEG channels only; volts multiplied by 1e4",
        "parameters": parameters,
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01},
        "seed": 0,
        "model_warmup_steps": 2,
        "worker_prefetch_factor": 2,
        "worker_timeout_seconds": 60,
        "batch_size": batch_size,
        "seconds": seconds,
        "starts": starts.tolist(),
        "targets": labels.tolist(),
        "positive_windows": int(labels.sum()),
        "recording": meta,
        "first_batch_max_abs_errors_volts": errors,
        "loss_equivalence_tolerance": {"rtol": 2e-4, "atol": 2e-5},
        "timing": "Includes worker startup and data-ready/compute steps; setup, MNE, two model warmups and reader shutdown excluded",
        "gpu_worker_policy": "Existing GPU assembler runs in main process with zero workers; CPU paths use requested worker counts",
        "environment": collect_environment(),
        "allocation": {
            key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "code_sha256": {
            name: checksum(Path(__file__).resolve().parents[1] / name)
            for name in (
                "profiling/benchmark_signal_training.py",
                "profiling/neonatal_targets.py",
                "profiling/training_windows.py",
                "profiling/edf_source.py",
                "profiling/edf_record_loading.py",
                "profiling/edf_reference.py",
                "profiling/signal_calibration.py",
                "profiling/signal_windows.py",
                "profiling/lance_signal_windows.py",
                "profiling/current_signal_cache.py",
                "profiling/benchmark_support.py",
                "profiling/research_environment.py",
                "stable_datasets/cache.py",
                "stable_datasets/dataset.py",
                "stable_datasets/formatting.py",
                "stable_datasets/features/base.py",
                "stable_datasets/backends/arrow_shards.py",
            )
        },
    }

    def emit(event, **values):
        if progress is not None:
            progress({"event": event, **values})

    emit("run_started", metadata=metadata)
    reports = []
    reference_losses = None
    for trial in range(trials):
        order = cases[trial % len(cases) :] + cases[: trial % len(cases)]
        for name, count in order:
            case = {"trial": trial, "path": name, "workers": count}
            emit("trial_started", **case)
            print(f"starting trial={trial} path={name} workers={count}", flush=True)
            result = None
            try:
                result = train_trial(datasets[name], count, target)
                losses = [row["loss"] for row in result["steps"]]
                if reference_losses is None:
                    reference_losses = losses
                np.testing.assert_allclose(losses, reference_losses, rtol=2e-4, atol=2e-5)
            except Exception as error:
                emit(
                    "trial_failed",
                    **case,
                    error_type=type(error).__name__,
                    error=str(error),
                    context=getattr(error, "context", None),
                    unvalidated_result=result,
                )
                raise
            result.update(trial=trial)
            reports.append(result)
            emit("trial_completed", validated=True, result=result)
            print(
                f"trial={trial} path={name} workers={count} seconds={result['elapsed_seconds']:.3f} "
                f"data_ready={100 * result['data_ready_fraction']:.1f}%",
                flush=True,
            )
    if checksum(source) != source_hash or checksum(annotations) != annotation_hash:
        raise ValueError("Experiment inputs changed during training")
    for name, digest in metadata["code_sha256"].items():
        if checksum(Path(__file__).resolve().parents[1] / name) != digest:
            raise ValueError("Experiment code changed during training")
    return {**metadata, "trials": reports}


def main():
    """Keep diagnostic configuration and artifact writing outside storage code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--paths", nargs="+", choices=(*CPU_PATHS, GPU_PATH))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Research results must be outside the repository")
    if output.exists():
        raise FileExistsError(output)
    journal = output.with_suffix(".progress.jsonl")
    with journal.open("x") as stream:

        def persist(event):
            entry = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        persist(
            {
                "event": "invocation",
                "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in args.items()},
            }
        )
        try:
            report = run(**args, progress=persist)
            with output.open("x") as destination:
                json.dump(report, destination, indent=2)
                destination.flush()
                os.fsync(destination.fileno())
        except Exception as error:
            persist({"event": "run_failed", "error_type": type(error).__name__, "error": str(error)})
            raise
        persist({"event": "run_completed", "output": str(output)})


if __name__ == "__main__":
    main()
