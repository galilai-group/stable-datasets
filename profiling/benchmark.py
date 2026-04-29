#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import csv
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


try:
    from filelock import FileLock
except ImportError:
    FileLock = None

np = None
torch = None
DataLoader = None
Dataset = object
PeakMemoryMonitor = None
StageProfile = None
dump_cprofile = None
dump_stage_profile = None
timed = None


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ImageNet-1K backend benchmark configuration.")
    parser.add_argument("--dataset", default="ImageNet-1K", choices=("ImageNet-1K",))
    parser.add_argument("--backend", required=True, choices=("hf", "pyarrow", "lance"))
    parser.add_argument("--regime", required=True, choices=("training", "sparse"))
    parser.add_argument("--decode", required=True, choices=("on", "off"))
    parser.add_argument("--experiment-name", default="manual")
    parser.add_argument("--parallelism-level", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lance-cpu-threads", type=int, default=None)
    parser.add_argument("--lance-io-threads", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--measured-batches", type=int, default=200)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--subset-fraction", type=float, default=0.05)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--memory-poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--multiprocessing-context", default="fork", choices=("fork", "spawn", "forkserver"))
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--processed-cache-dir", default=None)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument("--profile-delay-batches", type=int, default=0)
    parser.add_argument("--profile-output-dir", default="profiling/results/profiles")
    parser.add_argument("--profile-with-cprofile", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.backend == "lance":
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative.")
        if args.lance_cpu_threads is not None and args.lance_cpu_threads <= 0:
            raise ValueError("--lance-cpu-threads must be positive when provided.")
        if args.lance_io_threads is not None and args.lance_io_threads <= 0:
            raise ValueError("--lance-io-threads must be positive when provided.")
        if args.num_workers > 0 and args.multiprocessing_context != "spawn":
            raise ValueError(
                "Lance with --num-workers > 0 must use --multiprocessing-context spawn because Lance is not fork-safe."
            )
    elif args.lance_cpu_threads is not None or args.lance_io_threads is not None:
        raise ValueError("--lance-cpu-threads/--lance-io-threads may only be provided when --backend lance.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative.")
    if args.measured_batches <= 0:
        raise ValueError("--measured-batches must be positive.")
    if not (0.0 < args.subset_fraction <= 1.0):
        raise ValueError("--subset-fraction must be in (0, 1].")
    if args.profile_batches < 0:
        raise ValueError("--profile-batches must be non-negative.")
    if args.profile_delay_batches < 0:
        raise ValueError("--profile-delay-batches must be non-negative.")
    if args.profile_batches > 0 and args.num_workers != 0:
        raise ValueError("Profiling mode currently requires --num-workers 0 for interpretable traces.")
    if args.profile_delay_batches > 0 and args.profile_batches == 0:
        raise ValueError("--profile-delay-batches requires --profile-batches > 0.")


def configure_environment(args: argparse.Namespace) -> None:
    if args.backend == "lance":
        import os

        if args.lance_cpu_threads is not None:
            os.environ["LANCE_CPU_THREADS"] = str(args.lance_cpu_threads)
        if args.lance_io_threads is not None:
            os.environ["LANCE_IO_THREADS"] = str(args.lance_io_threads)


def load_runtime_dependencies() -> None:
    global np, torch, DataLoader, Dataset, PeakMemoryMonitor, StageProfile, dump_cprofile, dump_stage_profile, timed

    import numpy as _np
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import Dataset as _Dataset
    from utils import PeakMemoryMonitor as _PeakMemoryMonitor
    from utils import StageProfile as _StageProfile
    from utils import dump_cprofile as _dump_cprofile
    from utils import dump_stage_profile as _dump_stage_profile
    from utils import timed as _timed

    np = _np
    torch = _torch
    DataLoader = _DataLoader
    Dataset = _Dataset
    PeakMemoryMonitor = _PeakMemoryMonitor
    StageProfile = _StageProfile
    dump_cprofile = _dump_cprofile
    dump_stage_profile = _dump_stage_profile
    timed = _timed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_runtime_dependencies_loaded() -> None:
    if np is None or torch is None:
        load_runtime_dependencies()


def pil_to_tensor(image) -> torch.Tensor:
    ensure_runtime_dependencies_loaded()
    image = image.convert("RGB").resize((224, 224))
    array = np.array(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array).to(dtype=torch.float32) / 255.0


def extract_raw_image_payload(image_value):
    if isinstance(image_value, bytes):
        return image_value
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return image_value["bytes"]
        path = image_value.get("path")
        if path:
            return Path(path).read_bytes()
    return image_value


def normalize_sample(row: dict, decode_mode: str) -> dict:
    ensure_runtime_dependencies_loaded()
    label = int(row["label"])
    image_value = row["image"]
    if decode_mode == "on":
        image_value = pil_to_tensor(image_value)
    else:
        image_value = extract_raw_image_payload(image_value)
    return {"image": image_value, "label": label}


def instrument_stable_dataset_for_profiling(dataset, stage_profile) -> None:
    if stage_profile is None:
        return
    backend = getattr(dataset, "_backend", None)
    formatter = getattr(dataset, "_formatter", None)
    if backend is None or formatter is None:
        return

    if not hasattr(backend, "_profiling_original_take"):
        backend._profiling_original_take = backend.take

        def timed_take(indices, *, _original=backend.take):
            started = time.perf_counter()
            table = _original(indices)
            stage_profile.record_backend_take(time.perf_counter() - started)
            return table

        backend.take = timed_take

    if not hasattr(formatter, "_profiling_original_format_batch"):
        formatter._profiling_original_format_batch = formatter.format_batch

        def timed_format_batch(table, *, _original=formatter.format_batch):
            started = time.perf_counter()
            rows = _original(table)
            stage_profile.record_format_batch(time.perf_counter() - started)
            return rows

        formatter.format_batch = timed_format_batch


class TransformDataset(Dataset):
    def __init__(self, dataset, decode_mode: str, stage_profile=None):
        self.dataset = dataset
        self.decode_mode = decode_mode
        self.stage_profile = stage_profile

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.stage_profile is None or not self.stage_profile.enabled:
            return normalize_sample(self.dataset[index], self.decode_mode)

        started = time.perf_counter()
        row = self.dataset[index]
        self.stage_profile.record_source_getitem(time.perf_counter() - started)

        started = time.perf_counter()
        sample = normalize_sample(row, self.decode_mode)
        self.stage_profile.record_normalize(time.perf_counter() - started)
        return sample

    def __getitems__(self, indices: list[int]):
        if self.stage_profile is None or not self.stage_profile.enabled:
            if hasattr(self.dataset, "__getitems__"):
                rows = self.dataset.__getitems__(indices)
            else:
                rows = [self.dataset[i] for i in indices]
            return [normalize_sample(row, self.decode_mode) for row in rows]

        started = time.perf_counter()
        if hasattr(self.dataset, "__getitems__"):
            rows = self.dataset.__getitems__(indices)
            self.stage_profile.record_source_getitems(time.perf_counter() - started)
        else:
            rows = []
            for index in indices:
                item_started = time.perf_counter()
                rows.append(self.dataset[index])
                self.stage_profile.record_source_getitem(time.perf_counter() - item_started)

        samples = []
        for row in rows:
            normalize_started = time.perf_counter()
            samples.append(normalize_sample(row, self.decode_mode))
            self.stage_profile.record_normalize(time.perf_counter() - normalize_started)
        return samples


class IndexedViewDataset(Dataset):
    def __init__(self, dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def __getitems__(self, indices: list[int]):
        mapped = [int(self.indices[i]) for i in indices]
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(mapped)
        return [self.dataset[i] for i in mapped]


def benchmark_collate(batch: list[dict]) -> dict:
    images = [sample["image"] for sample in batch]
    labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.int64)
    if images and isinstance(images[0], torch.Tensor):
        image_batch = torch.stack(images)
    else:
        image_batch = images
    return {"image": image_batch, "label": labels}


@dataclass
class TimedCollate:
    stage_profile: object

    def __call__(self, batch: list[dict]) -> dict:
        with timed(self.stage_profile.record_collate):
            return benchmark_collate(batch)


def load_stable_dataset(args: argparse.Namespace):
    import stable_datasets as sds

    dataset_kwargs = {"split": "train"}
    if args.download_dir is not None:
        dataset_kwargs["download_dir"] = args.download_dir
    if args.processed_cache_dir is not None:
        dataset_kwargs["processed_cache_dir"] = args.processed_cache_dir

    dataset_kwargs["storage_format"] = "lance" if args.backend == "lance" else "arrow"

    dataset = sds.images.ImageNet1K(**dataset_kwargs)
    if args.decode == "off":
        dataset = dataset.set_decode(False)
    return dataset


def load_hf_dataset(args: argparse.Namespace):
    import os

    import datasets as hf_datasets

    load_kwargs = {"split": "train"}
    if os.environ.get("HF_DATASETS_CACHE"):
        load_kwargs["cache_dir"] = os.environ["HF_DATASETS_CACHE"]

    dataset = hf_datasets.load_dataset("ILSVRC/imagenet-1k", **load_kwargs)
    if args.decode == "off":
        dataset = dataset.cast_column("image", hf_datasets.Image(decode=False))
    return dataset


def load_dataset(args: argparse.Namespace, stage_profile=None):
    if args.dataset != "ImageNet-1K":
        raise ValueError(f"Unsupported dataset: {args.dataset!r}")
    if args.backend == "hf":
        base = load_hf_dataset(args)
    else:
        base = load_stable_dataset(args)
        instrument_stable_dataset_for_profiling(base, stage_profile)

    transformed = TransformDataset(base, decode_mode=args.decode, stage_profile=stage_profile)
    if args.regime == "sparse":
        subset_size = max(1, int(len(transformed) * args.subset_fraction))
        rng = np.random.default_rng(args.seed)
        subset_indices = rng.choice(len(transformed), size=subset_size, replace=False)
        transformed = IndexedViewDataset(transformed, subset_indices)
    return transformed


def build_dataloader(dataset, args: argparse.Namespace, *, stage_profile=None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    collate_fn = benchmark_collate
    if stage_profile is not None and stage_profile.enabled:
        collate_fn = TimedCollate(stage_profile)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "generator": generator,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["multiprocessing_context"] = args.multiprocessing_context
    return DataLoader(**loader_kwargs)


def batch_size_of(batch: dict) -> int:
    images = batch["image"]
    if isinstance(images, torch.Tensor):
        return int(images.shape[0])
    return len(images)


def consume_n_batches(loader: DataLoader, n_batches: int, *, stage_profile=None) -> tuple[int, float, float | None]:
    if n_batches == 0:
        return 0, 0.0, None

    iterator = iter(loader)
    total_images = 0
    first_batch_sec = None
    started = time.perf_counter()

    for _ in range(n_batches):
        while True:
            try:
                next_started = time.perf_counter()
                batch = next(iterator)
                next_elapsed = time.perf_counter() - next_started
                if first_batch_sec is None:
                    first_batch_sec = next_elapsed
                if stage_profile is not None:
                    stage_profile.record_loader_next(next_elapsed, batch_size_of(batch))
                break
            except StopIteration:
                iterator = iter(loader)
        total_images += batch_size_of(batch)
        del batch

    elapsed_sec = time.perf_counter() - started
    return total_images, elapsed_sec, first_batch_sec


@dataclass
class BenchmarkResult:
    timestamp_utc: str
    dataset: str
    backend: str
    regime: str
    decode_mode: str
    experiment_name: str
    parallelism_level: int
    num_workers: int
    lance_cpu_threads: int | None
    lance_io_threads: int | None
    batch_size: int
    warmup_batches: int
    measured_batches: int
    total_images: int
    elapsed_sec: float
    images_per_sec: float
    peak_total_host_uss_bytes: int
    time_to_first_batch_sec: float | None
    subset_fraction: float
    seed: int
    replicate_id: int


def profile_artifact_stem(args: argparse.Namespace) -> str:
    parallelism = args.parallelism_level
    if parallelism is None:
        parallelism = args.lance_cpu_threads if args.backend == "lance" else args.num_workers
    return (
        f"{args.dataset.lower().replace('-', '').replace(' ', '')}_"
        f"{args.backend}_{args.regime}_decode-{args.decode}_"
        f"{args.experiment_name}_p{parallelism}_r{args.replicate}"
    )


def maybe_run_profile_phase(loader, args: argparse.Namespace, *, stage_profile) -> None:
    if args.profile_batches <= 0:
        return

    output_dir = Path(args.profile_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = profile_artifact_stem(args)
    metadata = {
        "dataset": args.dataset,
        "backend": args.backend,
        "regime": args.regime,
        "decode_mode": args.decode,
        "experiment_name": args.experiment_name,
        "batch_size": args.batch_size,
        "profile_batches": args.profile_batches,
        "profile_delay_batches": args.profile_delay_batches,
        "num_workers": args.num_workers,
        "lance_cpu_threads": args.lance_cpu_threads,
        "lance_io_threads": args.lance_io_threads,
    }

    if args.profile_delay_batches > 0:
        consume_n_batches(loader, args.profile_delay_batches)
        stage_profile.add_note(f"Skipped {args.profile_delay_batches} post-warmup batches before profiling.")

    if args.profile_with_cprofile:
        profiler = cProfile.Profile()
        stage_profile.enable()
        profiler.enable()
        try:
            consume_n_batches(loader, args.profile_batches, stage_profile=stage_profile)
        finally:
            profiler.disable()
            stage_profile.disable()
        dump_cprofile(
            profiler,
            pstats_path=output_dir / f"{stem}.pstats",
            text_path=output_dir / f"{stem}.cprofile.txt",
        )
        stage_profile.add_note("cProfile enabled for profile phase.")
    else:
        stage_profile.enable()
        try:
            consume_n_batches(loader, args.profile_batches, stage_profile=stage_profile)
        finally:
            stage_profile.disable()

    dump_stage_profile(stage_profile, output_dir / f"{stem}.stages.json", metadata=metadata)


def append_result(result: BenchmarkResult, results_path: str) -> None:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    lock = FileLock(str(path) + ".lock") if FileLock is not None else nullcontext()

    with lock:
        file_exists = path.exists()
        if file_exists:
            with path.open("r", newline="") as handle:
                reader = csv.reader(handle)
                existing_header = next(reader, None)
            expected_header = list(row.keys())
            if existing_header is not None and existing_header != expected_header:
                raise ValueError(
                    f"Results schema mismatch for {path}. Existing header has "
                    f"{len(existing_header)} columns but this run would write "
                    f"{len(expected_header)}. Use a separate --results-path."
                )
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_environment(args)
    load_runtime_dependencies()
    seed_everything(args.seed)

    stage_profile = StageProfile() if args.profile_batches > 0 else None
    dataset = load_dataset(args, stage_profile=stage_profile)
    loader = build_dataloader(dataset, args, stage_profile=stage_profile)

    _, _, time_to_first_batch_sec = consume_n_batches(loader, args.warmup_batches)
    maybe_run_profile_phase(loader, args, stage_profile=stage_profile)

    monitor = PeakMemoryMonitor(poll_interval_sec=args.memory_poll_interval_sec)
    monitor.start()
    measured_started = time.perf_counter()
    total_images, _, _ = consume_n_batches(loader, args.measured_batches)
    elapsed_sec = time.perf_counter() - measured_started
    peak_total_host_uss_bytes = monitor.stop()

    result = BenchmarkResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dataset=args.dataset,
        backend=args.backend,
        regime=args.regime,
        decode_mode=args.decode,
        experiment_name=args.experiment_name,
        parallelism_level=(
            args.parallelism_level
            if args.parallelism_level is not None
            else (args.lance_cpu_threads if args.backend == "lance" else args.num_workers)
        ),
        num_workers=args.num_workers,
        lance_cpu_threads=args.lance_cpu_threads,
        lance_io_threads=args.lance_io_threads,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
        total_images=total_images,
        elapsed_sec=elapsed_sec,
        images_per_sec=(total_images / elapsed_sec) if elapsed_sec > 0 else 0.0,
        peak_total_host_uss_bytes=peak_total_host_uss_bytes,
        time_to_first_batch_sec=time_to_first_batch_sec,
        subset_fraction=args.subset_fraction,
        seed=args.seed,
        replicate_id=args.replicate,
    )
    append_result(result, args.results_path)

    print(
        f"{result.backend} {result.regime} decode={result.decode_mode} "
        f"images_per_sec={result.images_per_sec:.3f} "
        f"peak_total_host_uss_bytes={result.peak_total_host_uss_bytes}"
    )


if __name__ == "__main__":
    main()
