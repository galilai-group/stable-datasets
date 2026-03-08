#!/usr/bin/env python3
"""Comprehensive benchmark with multiple runs per dataset for ± ranges.

Runs each backend in a subprocess to avoid module-swap issues.
"""

import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

PYARROW_WORKTREE = os.path.join(os.path.dirname(__file__), ".claude", "worktrees", "pyarrow-migration")
TORCHVISION_ROOT = os.path.expanduser("~/.cache/torchvision")
# ImageNet requires a pre-extracted ILSVRC directory for Torchvision.
# Set this env var or pass --imagenet-root to override.
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", "/data/imagenet")

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
    "Beans": {
        "stable": ("stable_datasets.images.beans", "Beans"),
        "hf": ("beans", None),
        "tv": None,
    },
    "Flowers102": {
        "stable": ("stable_datasets.images.flowers102", "Flowers102"),
        "hf": ("nelorth/oxford-flowers", None),
        "tv": ("Flowers102", "split_arg"),
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

SPLIT = "train"
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_RUNS = 5

# ── Subprocess worker script ────────────────────────────────────────────────

WORKER_SCRIPT = r'''
import gc, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader

def _list_collate(batch):
    return batch

backend = sys.argv[1]
module_path = sys.argv[2]
cls_name = sys.argv[3]
split = sys.argv[4]
batch_size = int(sys.argv[5])
num_workers = int(sys.argv[6])

# Extra args for specific backends
extra = sys.argv[7:]

try:
    gc.collect()
    t0 = time.perf_counter()

    if backend == "stable":
        import importlib
        mod = importlib.import_module(module_path)
        ds = getattr(mod, cls_name)(split=split)

    elif backend == "pyarrow":
        worktree = extra[0]
        sys.path.insert(0, worktree)
        import importlib
        mod = importlib.import_module(module_path)
        ds = getattr(mod, cls_name)(split=split)

    elif backend == "hf":
        import datasets as hf_datasets
        hub_path = extra[0]
        config_name = extra[1] if len(extra) > 1 and extra[1] != "None" else None
        kw = {"name": config_name} if config_name else {}
        ds = hf_datasets.load_dataset(hub_path, split=split, **kw)

    elif backend == "tv":
        import torchvision.datasets as tv_datasets
        tv_root = extra[0]
        split_style = extra[1]
        os.makedirs(tv_root, exist_ok=True)
        cls = getattr(tv_datasets, cls_name)
        if split_style == "train_flag":
            ds = cls(tv_root, train=(split == "train"), download=False)
        elif split_style == "imagenet_dir":
            # ImageNet expects a root dir containing train/ and val/ subdirectories
            ds = cls(tv_root, split="val" if split == "test" else split)
        else:
            ds = cls(tv_root, split=split, download=False)

    prep = time.perf_counter() - t0
    n = len(ds)

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_list_collate)
    gc.collect()
    t0 = time.perf_counter()
    for _ in loader:
        pass
    read = time.perf_counter() - t0

    print(json.dumps({"prep": prep, "read": read, "n": n}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
'''


def run_one(backend, module_path, cls_name, cfg):
    """Run a single benchmark in a subprocess. Returns {"prep", "read", "n"} or {"error"}."""
    cmd = [sys.executable, "-c", WORKER_SCRIPT, backend, module_path, cls_name,
           SPLIT, str(BATCH_SIZE), str(NUM_WORKERS)]

    if backend == "pyarrow":
        cmd.append(PYARROW_WORKTREE)
    elif backend == "hf":
        hf_cfg = cfg["hf"]
        cmd.extend([hf_cfg[0], str(hf_cfg[1])])
    elif backend == "tv":
        tv_cfg = cfg["tv"]
        root = IMAGENET_ROOT if tv_cfg[1] == "imagenet_dir" else TORCHVISION_ROOT
        cmd.extend([root, tv_cfg[1]])

    # ImageNet-1K epoch is ~1.28M samples; give it more time
    timeout = 1800 if "ImageNet" in cls_name else 300
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({timeout}s)"}
    stdout = result.stdout.strip()
    if not stdout:
        return {"error": result.stderr[:200] if result.stderr else "no output"}
    try:
        return json.loads(stdout.split("\n")[-1])
    except json.JSONDecodeError:
        return {"error": stdout[:200]}


def fmt(values):
    """Format as 'median (min-max)'."""
    if not values:
        return "--"
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    def _t(s):
        return f"{s*1000:.0f}ms" if s < 1.0 else f"{s:.2f}s"
    return f"{_t(med)} ({_t(lo)}-{_t(hi)})"


def main():
    import argparse
    global IMAGENET_ROOT, BATCH_SIZE, NUM_WORKERS, NUM_RUNS, SPLIT

    parser = argparse.ArgumentParser(description="Comprehensive dataloading benchmark")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to benchmark (default: all). Choices: {', '.join(DATASETS)}")
    parser.add_argument("--imagenet-root", default=IMAGENET_ROOT,
                        help=f"Root directory for ImageNet (default: {IMAGENET_ROOT} or $IMAGENET_ROOT)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()

    IMAGENET_ROOT = args.imagenet_root
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_RUNS = args.num_runs
    SPLIT = args.split

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())
    invalid = [n for n in ds_names if n not in DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    print(f"Comprehensive benchmark: {NUM_RUNS} runs, batch_size={BATCH_SIZE}, "
          f"num_workers={NUM_WORKERS}, split={SPLIT}")
    print(f"Datasets: {ds_names}")
    if "ImageNet-1K" in ds_names:
        print(f"ImageNet root: {IMAGENET_ROOT}")
    print()

    # results[dataset][backend] = {"prep": [...], "read": [...]}
    results = {}

    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        results[ds_name] = {}
        print(f"{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        for backend in ALL_BACKENDS:
            label = BACKEND_LABELS[backend]

            # Check if backend is available
            if backend in ("stable", "pyarrow") and cfg.get("stable") is None:
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue
            if backend == "hf" and cfg.get("hf") is None:
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue
            if backend == "tv" and cfg.get("tv") is None:
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue

            stable_cfg = cfg.get("stable")
            if stable_cfg:
                module_path, cls_name = stable_cfg
            else:
                # No stable builder; use a placeholder for HF/TV-only datasets
                module_path, cls_name = "n/a", "n/a"
            if backend == "tv":
                cls_name = cfg["tv"][0]

            preps, reads = [], []
            for run in range(NUM_RUNS):
                r = run_one(backend, module_path, cls_name, cfg)
                if "error" in r:
                    print(f"  [{label}] run {run+1}/{NUM_RUNS}: ERROR - {r['error'][:100]}", flush=True)
                    break
                preps.append(r["prep"])
                reads.append(r["read"])
                print(f"  [{label}] run {run+1}/{NUM_RUNS}: "
                      f"prep={r['prep']*1000:.0f}ms read={r['read']:.2f}s ({r['n']} samples)", flush=True)

            if preps:
                results[ds_name][backend] = {"prep": preps, "read": reads}
            else:
                results[ds_name][backend] = None
        print()

    # Print tables
    col_w = 24
    backends = list(ALL_BACKENDS)

    for metric, title in [("prep", "Preparation Time"), ("read", "1-Epoch Read Time")]:
        header = f"{'Dataset':<14} " + " ".join(f"{BACKEND_LABELS[b]:>{col_w}}" for b in backends)
        sep = "-" * len(header)
        print(f"\n{title} (median (min-max), {NUM_RUNS} runs, batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})")
        print(sep)
        print(header)
        print(sep)
        for ds_name in ds_names:
            parts = [f"{ds_name:<14}"]
            for b in backends:
                r = results[ds_name].get(b)
                if r is None:
                    parts.append(f"{'--':>{col_w}}")
                else:
                    parts.append(f"{fmt(r[metric]):>{col_w}}")
            print(" ".join(parts))
        print(sep)

    # Total table
    header = f"{'Dataset':<14} " + " ".join(f"{BACKEND_LABELS[b]:>{col_w}}" for b in backends)
    sep = "-" * len(header)
    print(f"\nTotal (prep + read) (median (min-max), {NUM_RUNS} runs)")
    print(sep)
    print(header)
    print(sep)
    for ds_name in DATASETS:
        parts = [f"{ds_name:<14}"]
        for b in backends:
            r = results[ds_name].get(b)
            if r is None:
                parts.append(f"{'--':>{col_w}}")
            else:
                totals = [p + rd for p, rd in zip(r["prep"], r["read"])]
                parts.append(f"{fmt(totals):>{col_w}}")
        print(" ".join(parts))
    print(sep)


if __name__ == "__main__":
    main()
