"""Prepare and compare neonatal window layouts with matched physical endpoints.

Preparation, correctness checks and report writing are outside timing. Run
artifacts live outside the repository. Only warm-source latency is implemented;
reopening files is not represented as cold storage. No training/label policy.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import lance
import numpy as np
import torch

from profiling.benchmark_support import checksum, measure_stage, transfer_to_cuda
from profiling.current_signal_cache import CurrentSignalCache, write_current_cache
from profiling.lance_signal_windows import LanceSignalWindows, write_signal_store
from profiling.research_environment import collect_environment
from profiling.signal_windows import Recording, Window, physical_windows


def _external(path: Path):
    if path.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Experiment data, results and figures must be outside the repository")


def prepare(source: Path, workspace: Path, chunk_samples: int):
    """Build both Lance layouts and a production-format baseline from one EDF.

    This is one recording's signal-column experiment, not a full builder run.
    Embedded annotations are retained in metadata, not converted to labels.
    """
    from profiling.edf_source import NativeEdfReader

    _external(workspace)
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    workspace.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    before = checksum(source)
    native = NativeEdfReader(source)
    m = native.metadata
    meta = Recording(
        m.sample_count, m.sample_rate_hz, m.channel_names, tuple(m.gain.tolist()), tuple(m.offset.tolist())
    )
    samples = native.read_windows(np.array([0]), m.sample_count)[0]
    write_signal_store(workspace / "recordings.lance", [meta], [samples])
    write_signal_store(workspace / "chunks.lance", [meta], [samples], chunk_samples=chunk_samples)
    physical = physical_windows([meta], [Window(0, 0, m.sample_count)], [samples])
    write_current_cache(workspace / "current", physical)
    if checksum(source) != before:
        raise ValueError("Source changed during preparation")
    report = {
        "source_name": source.name,
        "source_sha256": before,
        "recording": meta.to_dict(),
        "chunk_samples": chunk_samples,
        "lance_version": lance.__version__,
        "preparation_seconds": perf_counter() - started,
        "annotations": [
            {"onset_s": a.onset, "duration_s": a.duration, "text": a.text} for a in native.recording.annotations
        ],
        "scope": "One-recording production signal-column baseline; no external neonatal MAT labels or classifier targets",
    }
    (workspace / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")


def run(
    source: Path,
    workspace: Path,
    *,
    batch_sizes: list[int],
    seconds: float,
    device: str,
    repeats: int,
    storage_label: str,
):
    """Validate every path against MNE, then time balanced warm-source batches."""
    import mne

    from profiling.edf_record_loading import BatchedEdfRecords, GpuRecordAssembler
    from profiling.edf_reference import read_mne_windows
    from profiling.edf_source import NativeEdfReader
    from profiling.signal_calibration import calibrate_cpu

    if not batch_sizes or any(b < 1 for b in batch_sizes) or repeats < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Require positive batch sizes, repetitions and finite window seconds")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.set_num_threads(1)
    prep = json.loads((workspace / "preparation.json").read_text())
    digest = checksum(source)
    if digest != prep["source_sha256"]:
        raise ValueError("EDF differs from the prepared source")
    meta = Recording.from_dict(prep["recording"])
    length = round(seconds * meta.rate_hz)
    if length < 1 or length > meta.n_samples or not np.isclose(length / meta.rate_hz, seconds):
        raise ValueError("Duration must be sample-aligned and fit the recording")
    current = CurrentSignalCache(workspace / "current", [meta])
    blobs, chunks = (LanceSignalWindows(workspace / name) for name in ("recordings.lance", "chunks.lance"))
    if blobs.recordings != [meta] or chunks.recordings != [meta]:
        raise ValueError("Prepared layout metadata disagree")
    native = NativeEdfReader(source)
    records = BatchedEdfRecords(source, native)
    target = torch.device(device)
    assembler = GpuRecordAssembler(records, target) if device == "cuda" else None
    reports = []
    with mne.io.read_raw_edf(str(source), preload=False, verbose="ERROR") as reference:
        for batch_size in batch_sizes:
            starts = np.random.default_rng(0).integers(0, meta.n_samples - length + 1, size=batch_size)
            requests = [Window(0, int(start), length) for start in starts]
            expected = read_mne_windows(reference, starts, length)

            def memmap():
                values = records.read_cpu_from_records(records.plan(starts, length))
                return calibrate_cpu(values, native.metadata.gain, native.metadata.offset)

            def blob():
                return np.stack(physical_windows([meta], requests, blobs.read(requests)))

            def chunk():
                return np.stack(physical_windows([meta], requests, chunks.read(requests)))

            cpu = {
                "current_cache": lambda: np.stack(current.read(requests)),
                "edf_memmap": memmap,
                "lance_blob_ranges": blob,
                "lance_chunk_rows": chunk,
            }
            operations = {
                name: (lambda fn=fn: transfer_to_cuda(fn())) if device == "cuda" else fn for name, fn in cpu.items()
            }
            if assembler is not None:
                operations["edf_gpu_assembly"] = lambda: assembler.assemble(records.plan(starts, length))
            errors = {}
            for name, operation in operations.items():
                actual = operation()
                if isinstance(actual, torch.Tensor):
                    actual = actual.cpu().numpy()
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10)
                errors[name] = float(np.max(np.abs(actual - expected)))
            trials = {name: [] for name in operations}
            order = np.random.default_rng(0).permutation(list(operations)).tolist()
            for round_index in range(len(order)):
                for name in order[round_index:] + order[:round_index]:
                    trials[name].append(measure_stage(operations[name], target, 2, repeats))
            results = {}
            for name, values in trials.items():
                latency = float(np.median([v["median_ms"] for v in values]))
                results[name] = {
                    "median_ms": latency,
                    "windows_per_second": batch_size * 1000 / latency,
                    "rounds": values,
                }
            reports.append(
                {
                    "batch_size": batch_size,
                    "window_starts": starts.tolist(),
                    "results": results,
                    "max_abs_errors_volts": errors,
                    "blob_requested_bytes": blobs.payload_bytes(requests),
                    "chunk_selected_bytes": chunks.payload_bytes(requests),
                    "current_materialized_scalars": current.materialized_samples(requests),
                }
            )
    if checksum(source) != digest:
        raise ValueError("Source changed during measurement")
    return {
        "scope": "Warm-source completed-batch rate; no cold/NFS verification, overlap, workers or training claim",
        "baseline": "Production Arrow writer/backend/formatter, signal column only; one materialization per recording per batch",
        "endpoint": "float32 GPU tensor in volts" if device == "cuda" else "float32 NumPy batch in volts",
        "storage_label_user_supplied": storage_label,
        "source_name": source.name,
        "source_sha256": digest,
        "environment": collect_environment(),
        "lance_version": lance.__version__,
        "seconds": seconds,
        "samples_per_window": length,
        "recording": meta.to_dict(),
        "repeats": repeats,
        "warmups_per_path_round": 2,
        "reports": reports,
        "code_sha256": {
            name: checksum(Path(__file__).resolve().parents[1] / name)
            for name in (
                "profiling/benchmark_signal_windows.py",
                "profiling/current_signal_cache.py",
                "profiling/lance_signal_windows.py",
                "profiling/signal_windows.py",
                "profiling/edf_source.py",
                "profiling/edf_reference.py",
                "profiling/edf_record_loading.py",
                "profiling/signal_calibration.py",
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


def plot(report: Path, output: Path):
    """Render a standalone throughput comparison; never mix CPU/GPU endpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _external(output)
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(report.read_text())
    figure, axis = plt.subplots(figsize=(7, 4))
    for name in data["reports"][0]["results"]:
        axis.plot(
            [r["batch_size"] for r in data["reports"]],
            [r["results"][name]["windows_per_second"] for r in data["reports"]],
            marker="o",
            label=name,
        )
    axis.set(xlabel="Batch size", ylabel="Windows / second", title=f"Warm source — {data['endpoint']}")
    axis.set_yscale("log")
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    """Keep preparation, measurement and plotting explicit and reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("workspace", type=Path)
    prep.add_argument("--chunk-samples", type=int, default=1024)
    bench = sub.add_parser("run")
    bench.add_argument("source", type=Path)
    bench.add_argument("workspace", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    bench.add_argument("--seconds", type=float, default=4)
    bench.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--storage-label", default="unspecified")
    draw = sub.add_parser("plot")
    draw.add_argument("report", type=Path)
    draw.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    elif command == "run":
        output = args.pop("output")
        _external(output)
        if output.exists():
            raise FileExistsError(output)
        report = run(**args)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2)
    else:
        plot(**args)


if __name__ == "__main__":
    main()
