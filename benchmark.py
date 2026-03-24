#!/usr/bin/env python3
"""Benchmark stable-datasets vs HuggingFace Datasets vs Torchvision.

Runs each backend in a subprocess to get clean memory/timing measurements.
Outputs terminal tables + LaTeX tables to profile_tables.tex (or --latex-output).

Usage:
    python profile.py                          # all datasets, 5 runs
    python profile.py --datasets CIFAR10 Beans # specific datasets
    python profile.py --num-runs 3             # fewer runs
    python profile.py -o results.json          # save raw JSON
"""

import json
import os
import statistics
import subprocess
import sys

DEFAULT_CACHE_DIR = "/users/sboughan/scratch/.stable-datasets"
TORCHVISION_ROOT = os.path.expanduser("~/.cache/torchvision")
IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", "/data/imagenet")

# Datasets whose caches should be cleaned between backend runs to avoid
# hitting quota.  ImageNet alone is ~150 GB per backend.
LARGE_DATASETS = {"ImageNet-1K"}

# Dataset registry.  Each entry maps backend -> config.
#   stable: (module_path, class_name) or None
#   hf:     (hub_path, config_name_or_None) or None
#   tv:     (class_name, split_style) or None
#       split_style: "train_flag" = train=True/False, "split_arg" = split="train",
#                    "imagenet_dir" = root dir with train/val subdirs
DATASETS = {
    "CIFAR-10": {
        "stable": ("stable_datasets.images.cifar10", "CIFAR10"),
        "hf": ("cifar10", None),
        "tv": ("CIFAR10", "train_flag"),
    },
    "CIFAR-100": {
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
    "STL-10": {
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
        "stable": ("stable_datasets.images.imagenet_1k", "ImageNet1K"),
        "hf": ("ILSVRC/imagenet-1k", None),
        "tv": ("ImageNet", "imagenet_dir"),
    },
}

BACKENDS = ("stable", "hf", "tv")
BACKEND_LABELS = {
    "stable": "stable-datasets",
    "hf": "HF Datasets",
    "tv": "Torchvision",
}

# LaTeX column header for each backend
LATEX_HEADERS = {
    "stable": r"\textbf{stable-datasets}",
    "hf": r"\textbf{HF Datasets}",
    "tv": r"\textbf{Torchvision}",
}


# -- Subprocess worker script ------------------------------------------------
# This runs in a fresh process so each backend gets clean imports and memory.

WORKER_SCRIPT = r'''
import gc, json, os, resource, sys, time, warnings
warnings.filterwarnings("ignore")

# Cache dir is passed via env by the parent process:
#   STABLE_DATASETS_CACHE_DIR  -> stable-datasets
#   HF_HOME                   -> huggingface datasets

import torch
from torch.utils.data import DataLoader

def _list_collate(batch):
    return batch

backend   = sys.argv[1]
mod_path  = sys.argv[2]
cls_name  = sys.argv[3]
split     = sys.argv[4]
batch_size  = int(sys.argv[5])
num_workers = int(sys.argv[6])
num_epochs  = int(sys.argv[7])
extra = sys.argv[8:]

try:
    gc.collect()
    t0 = time.perf_counter()

    if backend == "stable":
        import importlib
        mod = importlib.import_module(mod_path)
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
        tv_download = os.environ.get("TV_DOWNLOAD", "0") == "1"
        if split_style == "train_flag":
            ds = cls(tv_root, train=(split == "train"), download=tv_download)
        elif split_style == "imagenet_dir":
            ds = cls(tv_root, split="val" if split == "test" else split)
        else:
            ds = cls(tv_root, split=split, download=tv_download)

    prep = time.perf_counter() - t0
    n = len(ds)

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_list_collate)

    # Run multiple epochs to get mean +/- std
    epoch_times = []
    for epoch in range(num_epochs):
        gc.collect()
        t0 = time.perf_counter()
        for _ in loader:
            pass
        epoch_times.append(time.perf_counter() - t0)

    # Peak RSS in bytes (ru_maxrss is in KB on Linux)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    print(json.dumps({"prep": prep, "epoch_times": epoch_times, "n": n, "peak_rss": peak_rss}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
'''


def run_one(backend, cfg, args):
    """Run a single benchmark in a subprocess."""
    stable_cfg = cfg.get("stable")
    mod_path = stable_cfg[0] if stable_cfg else "n/a"
    cls_name = stable_cfg[1] if stable_cfg else "n/a"

    if backend == "tv":
        cls_name = cfg["tv"][0]

    cmd = [
        sys.executable, "-c", WORKER_SCRIPT,
        backend, mod_path, cls_name,
        args.split, str(args.batch_size), str(args.num_workers),
        str(args.num_epochs),
    ]

    if backend == "hf":
        hf_cfg = cfg["hf"]
        cmd.extend([hf_cfg[0], str(hf_cfg[1])])
    elif backend == "tv":
        tv_cfg = cfg["tv"]
        tv_root = args.imagenet_root if tv_cfg[1] == "imagenet_dir" else os.path.join(args.cache_dir, "torchvision")
        cmd.extend([tv_root, tv_cfg[1]])

    timeout = 7200 if "imagenet" in cls_name.lower() else 600

    # Set cache dirs so downloads land on scratch, not home
    env = os.environ.copy()
    env["STABLE_DATASETS_CACHE_DIR"] = args.cache_dir
    env["HF_HOME"] = os.path.join(args.cache_dir, "huggingface")
    env["TV_DOWNLOAD"] = "1"  # allow torchvision to download on first run

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({timeout}s)"}

    stdout = result.stdout.strip()
    if not stdout:
        return {"error": result.stderr[:200] if result.stderr else "no output"}
    try:
        return json.loads(stdout.split("\n")[-1])
    except json.JSONDecodeError:
        return {"error": stdout[:200]}


def cleanup_cache(backend, args):
    """Remove cached data for a backend to free disk for the next one."""
    import shutil

    if backend == "stable":
        target = os.path.join(args.cache_dir, "processed")
    elif backend == "hf":
        target = os.path.join(args.cache_dir, "huggingface")
    else:
        return  # torchvision uses pre-extracted data, nothing to clean

    if os.path.exists(target):
        size_gb = sum(
            f.stat().st_size for f in __import__("pathlib").Path(target).rglob("*") if f.is_file()
        ) / (1024 ** 3)
        print(f"  [cleanup] removing {target} ({size_gb:.1f} GB)", flush=True)
        shutil.rmtree(target, ignore_errors=True)


# -- Formatting ---------------------------------------------------------------


def fmt_time(seconds):
    """Format seconds for terminal display."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_median_range(values):
    """Format as 'median (min-max)' for terminal."""
    if not values:
        return "--"
    med = statistics.median(values)
    return f"{fmt_time(med)} ({fmt_time(min(values))}-{fmt_time(max(values))})"


def fmt_mean_std(values):
    """Format as 'mean +/- std' for terminal."""
    if not values:
        return "--"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{fmt_time(m)} +/- {fmt_time(s)}"


def fmt_latex_time(seconds):
    r"""Format seconds for LaTeX (e.g. '3.65\,s' or '698\,ms')."""
    if seconds < 1.0:
        return rf"{seconds * 1000:.0f}\,ms"
    return rf"{seconds:.2f}\,s"


def fmt_rss(bytes_val):
    """Format bytes as human-readable MB."""
    return f"{bytes_val / 1024 / 1024:.0f} MB"


# -- LaTeX generation ---------------------------------------------------------


def write_latex(results, ds_names, args, path):
    """Write three LaTeX tables (prep, read, total) matching the paper format."""
    backends = [b for b in BACKENDS if any(
        results.get(ds, {}).get(b) is not None for ds in ds_names
    )]

    def _col_header():
        return " & ".join(LATEX_HEADERS[b] for b in backends)

    def _best_row(ds_name, metric, use_mean_std=False):
        """Build one row, bolding the best (lowest) value."""
        vals = {}
        for b in backends:
            r = results.get(ds_name, {}).get(b)
            if r and metric in r:
                vals[b] = statistics.mean(r[metric]) if use_mean_std else statistics.median(r[metric])

        best_b = min(vals, key=vals.get) if vals else None
        cells = []
        for b in backends:
            r = results.get(ds_name, {}).get(b)
            if r is None or metric not in r:
                cells.append("---")
            elif use_mean_std:
                m = statistics.mean(r[metric])
                std = statistics.stdev(r[metric]) if len(r[metric]) > 1 else 0.0
                s = rf"${fmt_latex_time(m)} \pm {fmt_latex_time(std)}$"
                if b == best_b:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            else:
                med = statistics.median(r[metric])
                s = fmt_latex_time(med)
                if b == best_b:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        return f"{ds_name:<14} & " + " & ".join(cells) + r" \\"

    n = args.num_runs
    bs = args.batch_size
    nw = args.num_workers

    tables = []

    # Table 1: Preparation time
    tables.append(rf"""\begin{{table}}[t]
\centering
\caption{{Preparation time comparison (median over {n} runs). Lower is better.}}
\label{{tab:prep-time}}
\begin{{tabular}}{{l{"c" * len(backends)}}}
\toprule
\textbf{{Dataset}} & {_col_header()} \\
\midrule""")
    for ds in ds_names:
        tables.append(_best_row(ds, "prep"))
    tables.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Table 2: Read time (mean +/- std across epochs)
    ne = args.num_epochs
    tables.append(rf"""
\begin{{table}}[t]
\centering
\caption{{One-epoch read time (mean $\pm$ std over {n} $\times$ {ne} epochs). Lower is better.}}
\label{{tab:read-time}}
\begin{{tabular}}{{l{"c" * len(backends)}}}
\toprule
\textbf{{Dataset}} & {_col_header()} \\
\midrule""")
    for ds in ds_names:
        tables.append(_best_row(ds, "read", use_mean_std=True))
    tables.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Table 3: Total (prep + mean epoch read)
    # Compute per-backend: median(prep) + mean(epoch reads)
    def _total_row(ds_name):
        vals = {}
        for b in backends:
            r = results.get(ds_name, {}).get(b)
            if r and "prep" in r and "read" in r:
                vals[b] = statistics.median(r["prep"]) + statistics.mean(r["read"])

        best_b = min(vals, key=vals.get) if vals else None
        cells = []
        for b in backends:
            if b not in vals:
                cells.append("---")
            else:
                s = fmt_latex_time(vals[b])
                if b == best_b:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        return f"{ds_name:<14} & " + " & ".join(cells) + r" \\"

    tables.append(rf"""
\begin{{table}}[t]
\centering
\caption{{Total time (median prep $+$ mean epoch read, {n} runs $\times$ {ne} epochs). Lower is better.}}
\label{{tab:total-time}}
\begin{{tabular}}{{l{"c" * len(backends)}}}
\toprule
\textbf{{Dataset}} & {_col_header()} \\
\midrule""")
    for ds in ds_names:
        tables.append(_total_row(ds))
    tables.append(r"""\bottomrule
\end{tabular}
\end{table}""")

    header = (
        f"% Benchmark results: stable-datasets vs HuggingFace Datasets vs Torchvision\n"
        f"% {n} runs, batch_size={bs}, num_workers={nw}, split={args.split}\n"
    )
    with open(path, "w") as f:
        f.write(header + "\n".join(tables) + "\n")


# -- Main ---------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset loading benchmark")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets to benchmark (default: all). Choices: {', '.join(DATASETS)}",
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="Base cache directory for downloads and processed data")
    parser.add_argument("--imagenet-root", default=IMAGENET_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Epochs per subprocess run for read-time mean +/- std")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("-o", "--output", default=None, help="Save raw results to JSON")
    parser.add_argument(
        "--latex-output", default="profile_tables.tex",
        help="LaTeX output path (default: profile_tables.tex)",
    )
    args = parser.parse_args()

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())
    invalid = [n for n in ds_names if n not in DATASETS]
    if invalid:
        print(f"Unknown datasets: {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    print(
        f"Benchmark: {args.num_runs} runs, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, split={args.split}"
    )
    print(f"Datasets: {ds_names}\n")

    # results[dataset][backend] = {"prep": [...], "read": [...], "total": [...], "peak_rss": [...]}
    results = {}

    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        results[ds_name] = {}
        print(f"{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        for backend in BACKENDS:
            label = BACKEND_LABELS[backend]

            if cfg.get(backend) is None and backend != "stable":
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue
            if backend == "stable" and cfg.get("stable") is None:
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue

            preps, all_epoch_times, rss_vals = [], [], []

            for run_idx in range(args.num_runs):
                r = run_one(backend, cfg, args)
                if "error" in r:
                    print(
                        f"  [{label}] run {run_idx + 1}/{args.num_runs}: "
                        f"ERROR - {r['error'][:100]}",
                        flush=True,
                    )
                    break
                preps.append(r["prep"])
                epoch_times = r["epoch_times"]
                all_epoch_times.extend(epoch_times)
                if "peak_rss" in r:
                    rss_vals.append(r["peak_rss"])
                mean_ep = statistics.mean(epoch_times)
                rss_str = f", rss={fmt_rss(r['peak_rss'])}" if "peak_rss" in r else ""
                print(
                    f"  [{label}] run {run_idx + 1}/{args.num_runs}: "
                    f"prep={fmt_time(r['prep'])} "
                    f"read={fmt_time(mean_ep)}/ep x{len(epoch_times)} "
                    f"({r['n']} samples{rss_str})",
                    flush=True,
                )

            if preps:
                entry = {
                    "prep": preps,
                    "read": all_epoch_times,  # all epoch times across all runs
                }
                if rss_vals:
                    entry["peak_rss"] = rss_vals
                results[ds_name][backend] = entry
            else:
                results[ds_name][backend] = None

            # Free disk for large datasets between backend runs
            if ds_name in LARGE_DATASETS:
                cleanup_cache(backend, args)

        print()

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Write LaTeX tables
    write_latex(results, ds_names, args, args.latex_output)
    print(f"LaTeX tables written to {args.latex_output}")

    # Print terminal summary tables
    col_w = 28

    # Prep: median (min-max)
    header = f"{'Dataset':<14} " + " ".join(
        f"{BACKEND_LABELS[b]:>{col_w}}" for b in BACKENDS
    )
    sep = "-" * len(header)
    print(f"\nPreparation Time (median, {args.num_runs} runs)")
    print(sep)
    print(header)
    print(sep)
    for ds_name in ds_names:
        parts = [f"{ds_name:<14}"]
        for b in BACKENDS:
            r = results[ds_name].get(b)
            if r is None:
                parts.append(f"{'--':>{col_w}}")
            else:
                parts.append(f"{fmt_median_range(r['prep']):>{col_w}}")
        print(" ".join(parts))
    print(sep)

    # Read: mean +/- std
    print(f"\n1-Epoch Read Time (mean +/- std, {args.num_runs} x {args.num_epochs} epochs)")
    print(sep)
    print(header)
    print(sep)
    for ds_name in ds_names:
        parts = [f"{ds_name:<14}"]
        for b in BACKENDS:
            r = results[ds_name].get(b)
            if r is None:
                parts.append(f"{'--':>{col_w}}")
            else:
                parts.append(f"{fmt_mean_std(r['read']):>{col_w}}")
        print(" ".join(parts))
    print(sep)


if __name__ == "__main__":
    main()
