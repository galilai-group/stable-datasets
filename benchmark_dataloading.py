#!/usr/bin/env python3
"""
benchmark_dataloading.py

Compare dataset preparation (from cache) and 1-epoch read times across:
  - stable-datasets  (current: HF datasets.GeneratorBasedBuilder + Arrow)
  - stable-pyarrow   (migration branch: raw PyArrow IPC, no HF dependency)
  - HuggingFace Datasets (load_dataset from Hub)
  - Torchvision (where available, otherwise '--')

Metrics:
  prep_time  — Wall-clock time to get a usable dataset object from warm cache.
               This is the time a user waits between calling the constructor and
               being able to index into the dataset.  Download time is excluded
               (run --download first to populate every backend's cache).

               What each backend actually does during "prep":
                 stable-datasets : BaseDatasetBuilder.__new__  →  download_and_prepare()
                                   (cache hit = no-op) + as_dataset() (memory-maps Arrow).
                 stable-pyarrow  : Same flow but with custom Arrow IPC cache; no HF overhead.
                 HF Datasets     : datasets.load_dataset()  →  resolves Hub metadata, checks
                                   fingerprint, memory-maps Arrow cache.
                 Torchvision     : Constructor loads raw files into RAM (e.g. unpickles numpy
                                   arrays for CIFAR; reads image directories for Food101).

  read_time  — Wall-clock time to iterate every sample in one epoch via a
               torch DataLoader (batch_size, num_workers, collate_fn=list).
               All backends return native Python objects (dicts with PIL images
               for Arrow-backed; (PIL, int) tuples for Torchvision).  No tensor
               conversion is applied to keep the comparison fair — we are
               measuring data I/O and decode, not tensor allocation.

Why stable-datasets can be faster than raw HF load_dataset despite using HF
under the hood:
  - stable-datasets downloads from the original data source once and builds
    its own Arrow cache locally.  Subsequent loads just memory-map the file.
  - HF load_dataset has extra overhead: Hub metadata resolution (even on warm
    cache), config/split fingerprinting, and more complex cache bookkeeping.

Why Torchvision read times can appear faster:
  - Datasets like CIFAR / MNIST load the *entire* dataset into RAM as numpy
    arrays during __init__ (counted as prep).  Iteration is then pure in-memory
    indexing + PIL.Image.fromarray.
  - Arrow-backed datasets (stable / HF) memory-map the file but still decode
    each image from stored PNG/bytes on every __getitem__ call.
  So Torchvision front-loads work into prep; Arrow-backed backends defer it to
  read.  Neither is "unfair", but you should look at prep + read combined for
  the full picture.

Usage:
    # 1. Populate caches for all backends:
    python benchmark_dataloading.py --download

    # 2. Benchmark from warm cache:
    python benchmark_dataloading.py

    # Specific datasets:
    python benchmark_dataloading.py --datasets CIFAR10 FashionMNIST

    # Customize DataLoader:
    python benchmark_dataloading.py --batch-size 256 --num-workers 8
"""

import argparse
import gc
import importlib
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader

# ── Constants ────────────────────────────────────────────────────────────────

TORCHVISION_ROOT = os.path.expanduser("~/.cache/torchvision")
PYARROW_WORKTREE = os.path.join(os.path.dirname(__file__), ".claude", "worktrees", "pyarrow-migration")
# ImageNet expects a pre-extracted ILSVRC directory for Torchvision.
# Set IMAGENET_ROOT env var to point to the directory containing train/ and val/.
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", "/data/imagenet")

# ── Dataset registry ─────────────────────────────────────────────────────────
#
# Each entry maps a dataset name to its backend configurations:
#   "stable":  (module_path, class_name) | None    — None if no builder exists
#   "pyarrow": (module_path, class_name)           — same module names, loaded from worktree
#   "hf":      (hub_path, config_name) | None      — config_name can be None
#   "tv":      (class_name, split_style) | None    — split_style: "train_flag" or "split_arg"

DATASETS = {
    "CIFAR10": {
        "stable": ("stable_datasets.images.cifar10", "CIFAR10"),
        "hf": ("cifar10", None),
        "tv": ("CIFAR10", "train_flag"),
    },
    "CIFAR100": {
        "stable": ("stable_datasets.images.cifar100", "CIFAR100"),
        "hf": ("cifar100", None),
        "tv": ("CIFAR100", "train_flag"),
    },
    "FashionMNIST": {
        "stable": ("stable_datasets.images.fashion_mnist", "FashionMNIST"),
        "hf": ("fashion_mnist", None),
        "tv": ("FashionMNIST", "train_flag"),
    },
    "KMNIST": {
        "stable": ("stable_datasets.images.k_mnist", "KMNIST"),
        "hf": ("kmnist", None),
        "tv": ("KMNIST", "train_flag"),
    },
    "SVHN": {
        "stable": ("stable_datasets.images.svhn", "SVHN"),
        "hf": ("svhn", "cropped_digits"),
        "tv": ("SVHN", "split_arg"),
    },
    "STL10": {
        "stable": ("stable_datasets.images.stl10", "STL10"),
        "hf": ("tanganke/stl10", None),
        "tv": ("STL10", "split_arg"),
    },
    "Food101": {
        "stable": ("stable_datasets.images.food101", "Food101"),
        "hf": ("food101", None),
        "tv": ("Food101", "split_arg"),
    },
    "DTD": {
        "stable": ("stable_datasets.images.dtd", "DTD"),
        "hf": ("dtd", None),
        "tv": ("DTD", "split_arg"),
    },
    "Flowers102": {
        "stable": ("stable_datasets.images.flowers102", "Flowers102"),
        "hf": ("nelorth/oxford-flowers", None),
        "tv": ("Flowers102", "split_arg"),
    },
    "FGVCAircraft": {
        "stable": ("stable_datasets.images.fgvc_aircraft", "FGVCAircraft"),
        "hf": None,
        "tv": ("FGVCAircraft", "split_arg"),
    },
    "Country211": {
        "stable": ("stable_datasets.images.country211", "Country211"),
        "hf": None,
        "tv": ("Country211", "split_arg"),
    },
    "Beans": {
        "stable": ("stable_datasets.images.beans", "Beans"),
        "hf": ("beans", None),
        "tv": None,
    },
    "ImageNet-1K": {
        "stable": None,
        "hf": ("ILSVRC/imagenet-1k", None),
        "tv": ("ImageNet", "imagenet_dir"),
    },
}

ALL_BACKENDS = ("stable", "pyarrow", "hf", "tv")
BACKEND_LABELS = {
    "stable": "stable-datasets",
    "pyarrow": "stable-pyarrow",
    "hf": "HF Datasets",
    "tv": "Torchvision",
}


# ── Collate function (module-level for pickling with num_workers > 0) ────────

def _list_collate(batch):
    """Return batch as a list; avoids tensor-stacking issues with variable-size images."""
    return batch


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_stable(module_path, cls_name, split):
    """Load a stable-datasets dataset (current branch, HF-backed)."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls(split=split)


def load_pyarrow(module_path, cls_name, split):
    """Load a stable-datasets dataset from the pyarrow-migration worktree."""
    # Temporarily prepend the worktree to sys.path so its stable_datasets takes priority.
    # We must also evict any cached stable_datasets modules to avoid mixing code.
    worktree = PYARROW_WORKTREE

    # Save and evict current stable_datasets modules
    saved_modules = {}
    for key in list(sys.modules):
        if key == "stable_datasets" or key.startswith("stable_datasets."):
            saved_modules[key] = sys.modules.pop(key)

    sys.path.insert(0, worktree)
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        ds = cls(split=split)
        return ds
    finally:
        sys.path.remove(worktree)
        # Evict worktree modules so the main branch is clean for subsequent calls
        for key in list(sys.modules):
            if key == "stable_datasets" or key.startswith("stable_datasets."):
                del sys.modules[key]
        # Restore original modules
        sys.modules.update(saved_modules)


def load_hf(hub_path, config_name, split):
    """Load a HuggingFace Hub dataset."""
    import datasets as hf_datasets

    kwargs = {}
    if config_name is not None:
        kwargs["name"] = config_name
    return hf_datasets.load_dataset(hub_path, split=split, **kwargs)


def load_torchvision(cls_name, split_style, split, download=False):
    """Load a torchvision dataset."""
    import torchvision.datasets as tv_datasets

    cls = getattr(tv_datasets, cls_name)

    if split_style == "imagenet_dir":
        # ImageNet expects a pre-extracted root with train/ and val/ subdirs.
        # download=True is not supported; data must be prepared externally.
        return cls(IMAGENET_ROOT, split="val" if split == "test" else split)

    os.makedirs(TORCHVISION_ROOT, exist_ok=True)
    if split_style == "train_flag":
        return cls(TORCHVISION_ROOT, train=(split == "train"), download=download)
    else:  # split_arg
        return cls(TORCHVISION_ROOT, split=split, download=download)


# ── Timing helpers ───────────────────────────────────────────────────────────

def time_preparation(load_fn):
    """Time dataset preparation (construction from cache). Returns (dataset, seconds)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = time.perf_counter()
    ds = load_fn()
    elapsed = time.perf_counter() - start
    return ds, elapsed


def time_epoch_read(ds, batch_size, num_workers):
    """Time one full epoch of DataLoader iteration. Returns seconds."""
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_list_collate,
        pin_memory=False,
    )

    gc.collect()
    start = time.perf_counter()
    for _ in loader:
        pass
    elapsed = time.perf_counter() - start
    return elapsed


def _format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{s:.1f}s"


# ── Per-backend benchmark ────────────────────────────────────────────────────

def _make_result(prep=None, read=None, length=None, error=None):
    return {"prep": prep, "read": read, "len": length, "error": error}


def _bench_one(label, load_fn, batch_size, num_workers, download_only):
    """Benchmark a single backend. Returns a result dict."""
    try:
        print(f"  {label}: loading...", end="", flush=True)
        ds, prep = time_preparation(load_fn)
        n = len(ds)
        print(f" {n} samples, prep={_format_time(prep)}", end="", flush=True)
        if download_only:
            print(" (cached)")
            return _make_result(prep=prep, length=n)
        read = time_epoch_read(ds, batch_size, num_workers)
        print(f", read={_format_time(read)}")
        del ds
        return _make_result(prep=prep, read=read, length=n)
    except Exception as e:
        print(f" ERROR: {e}")
        return _make_result(error=str(e))


def benchmark_dataset(name, cfg, split, batch_size, num_workers, download_only=False):
    """Benchmark a single dataset across all backends. Returns dict keyed by backend."""
    results = {}
    stable_cfg = cfg.get("stable")

    # ── stable-datasets (current, HF-backed) ──
    if stable_cfg is None:
        results["stable"] = _make_result(error="n/a")
    else:
        results["stable"] = _bench_one(
            f"[{name}] stable-datasets",
            lambda: load_stable(stable_cfg[0], stable_cfg[1], split),
            batch_size, num_workers, download_only,
        )
    gc.collect()

    # ── stable-pyarrow (migration worktree) ──
    if stable_cfg is None:
        results["pyarrow"] = _make_result(error="n/a")
    elif os.path.isdir(PYARROW_WORKTREE):
        results["pyarrow"] = _bench_one(
            f"[{name}] stable-pyarrow",
            lambda: load_pyarrow(stable_cfg[0], stable_cfg[1], split),
            batch_size, num_workers, download_only,
        )
    else:
        results["pyarrow"] = _make_result(error="n/a")
    gc.collect()

    # ── HuggingFace Datasets ──
    hf_cfg = cfg.get("hf")
    if hf_cfg is None:
        results["hf"] = _make_result(error="n/a")
    else:
        results["hf"] = _bench_one(
            f"[{name}] HF Datasets",
            lambda: load_hf(hf_cfg[0], hf_cfg[1], split),
            batch_size, num_workers, download_only,
        )
    gc.collect()

    # ── Torchvision ──
    tv_cfg = cfg.get("tv")
    if tv_cfg is None:
        results["tv"] = _make_result(error="n/a")
    else:
        results["tv"] = _bench_one(
            f"[{name}] Torchvision",
            lambda: load_torchvision(tv_cfg[0], tv_cfg[1], split, download=download_only),
            batch_size, num_workers, download_only,
        )
    gc.collect()

    return results


# ── Output ───────────────────────────────────────────────────────────────────

def print_table(all_results, metric, title):
    """Print a formatted comparison table for a given metric ('prep' or 'read')."""
    backends = list(ALL_BACKENDS)
    col_width = 18
    header_parts = [f"{'Dataset':<20}"]
    for b in backends:
        header_parts.append(f"{BACKEND_LABELS[b]:>{col_width}}")
    header = " ".join(header_parts)
    sep = "-" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)

    for name, res in all_results.items():
        parts = [f"{name:<20}"]
        for b in backends:
            r = res[b]
            if r["error"] == "n/a":
                parts.append(f"{'--':>{col_width}}")
            elif r["error"] is not None:
                parts.append(f"{'error':>{col_width}}")
            elif r[metric] is None:
                parts.append(f"{'--':>{col_width}}")
            else:
                parts.append(f"{_format_time(r[metric]):>{col_width}}")
        print(" ".join(parts))

    print(sep)

    # Also print prep+read combined if we have both
    if metric == "read":
        print(f"\n{title.replace('1-Epoch Read', 'Total (prep + read)')}")
        print(sep)
        print(header)
        print(sep)
        for name, res in all_results.items():
            parts = [f"{name:<20}"]
            for b in backends:
                r = res[b]
                if r["error"] == "n/a" or r["error"] is not None:
                    parts.append(f"{'--':>{col_width}}")
                elif r["prep"] is not None and r["read"] is not None:
                    parts.append(f"{_format_time(r['prep'] + r['read']):>{col_width}}")
                else:
                    parts.append(f"{'--':>{col_width}}")
            print(" ".join(parts))
        print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark stable-datasets vs HF Datasets vs Torchvision"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to benchmark (default: all). Choices: {', '.join(DATASETS)}",
    )
    parser.add_argument("--split", default="train", help="Split to benchmark (default: train)")
    parser.add_argument("--batch-size", type=int, default=128, help="DataLoader batch size (default: 128)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/cache data for all backends (run this first, then benchmark without this flag)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())
    invalid = [n for n in ds_names if n not in DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    pyarrow_available = os.path.isdir(PYARROW_WORKTREE)

    mode = "download" if args.download else "benchmark"
    print(f"Mode: {mode} | split={args.split} | batch_size={args.batch_size} | num_workers={args.num_workers}")
    print(f"Datasets: {ds_names}")
    print(f"pyarrow-migration worktree: {'found' if pyarrow_available else 'NOT FOUND (column will show --)'}\n")

    all_results = {}
    for name in ds_names:
        print(f"{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        all_results[name] = benchmark_dataset(
            name,
            DATASETS[name],
            args.split,
            args.batch_size,
            args.num_workers,
            download_only=args.download,
        )
        print()

    # ── Print summary tables ─────────────────────────────────────
    print_table(all_results, "prep", "Preparation Time (dataset load from cache)")

    if not args.download:
        print_table(
            all_results,
            "read",
            f"1-Epoch Read Time (batch_size={args.batch_size}, num_workers={args.num_workers})",
        )

    # ── Optionally save JSON ─────────────────────────────────────
    if args.output:
        import json

        serializable = {}
        for name, res in all_results.items():
            serializable[name] = {}
            for backend in ALL_BACKENDS:
                serializable[name][backend] = {
                    "prep_s": res[backend]["prep"],
                    "read_s": res[backend]["read"],
                    "num_samples": res[backend]["len"],
                    "error": res[backend]["error"],
                }
        with open(args.output, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "results": serializable,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
