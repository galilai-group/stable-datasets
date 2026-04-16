#!/usr/bin/env python3
"""Iteration-time benchmark: measures DataLoader epoch read times with warm caches.

Assumes caches are already built (run benchmark_prep.py first).
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

DEFAULT_CACHE_DIR = "/users/sboughan/scratch/.stable-datasets"
_WORKER_SCRIPT = str(Path(__file__).parent / "worker_iter.py")

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
    "Flowers102": {
        "stable": ("stable_datasets.images.flowers102", "Flowers102"),
        "hf": ("nelorth/oxford-flowers", None),
        "tv": ("Flowers102", "split_arg"),
    },
    "Imagenette": {
        "stable": ("stable_datasets.images.imagenette", "Imagenette"),
        "hf": None,
        "tv": None,
    },
    "ImageNet-1K": {
        "stable": ("stable_datasets.images.imagenet_1k", "ImageNet1K"),
        "hf": ("ILSVRC/imagenet-1k", None),
        "tv": ("ImageNet", "imagenet_dir"),
    },
}

BACKENDS = ("stable", "stable_lance", "hf", "tv")
BACKEND_LABELS = {
    "stable": "stable-datasets",
    "stable_lance": "stable-datasets (Lance)",
    "hf": "HF Datasets",
    "tv": "Torchvision",
}

IMAGENET_ROOT = os.environ.get("IMAGENET_ROOT", "/data/imagenet")
TORCHVISION_ROOT = os.path.expanduser("~/.cache/torchvision")



def fmt_time(seconds):
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_mean_std(values):
    if not values:
        return "--"
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{fmt_time(m)} +/- {fmt_time(s)}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Iteration-time benchmark (warm cache)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to benchmark. Choices: {', '.join(DATASETS)}")
    parser.add_argument("--backends", nargs="+", default=None,
                        help=f"Backends to benchmark. Choices: {', '.join(BACKENDS)}")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--imagenet-root", default=IMAGENET_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Enable DataLoader shuffle=True. Routes indices through backend.take "
        "with random-order batches -- the map-style training workload.",
    )
    parser.add_argument(
        "--history",
        default="benchmarks/results/profile_iter_history.json",
        help="Consolidated history file to append this run to. "
        "Set to '' or pass --no-history to suppress.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append to the consolidated history file.",
    )
    parser.add_argument("-o", "--output", default=None, help="Save raw results to JSON")
    args = parser.parse_args()

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())
    backends = tuple(args.backends) if args.backends else BACKENDS

    print(f"Iteration benchmark: {args.num_runs} runs x {args.num_epochs} epochs, "
          f"batch_size={args.batch_size}, num_workers={args.num_workers}")
    print(f"Datasets: {ds_names}")
    print(f"Backends: {list(backends)}\n")

    results = {}

    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        results[ds_name] = {}
        print(f"{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        for backend in backends:
            label = BACKEND_LABELS.get(backend, backend)

            # "stable_lance" rides on the "stable" config: same
            # builder module and class, different backend installed
            # post-hoc by worker_iter.py.
            cfg_key = "stable" if backend == "stable_lance" else backend
            if cfg.get(cfg_key) is None:
                results[ds_name][backend] = None
                print(f"  [{label}] n/a", flush=True)
                continue

            stable_cfg = cfg.get("stable")
            mod_path = stable_cfg[0] if stable_cfg else "n/a"
            cls_name = stable_cfg[1] if stable_cfg else "n/a"

            if backend == "tv":
                cls_name = cfg["tv"][0]

            cmd = [sys.executable, _WORKER_SCRIPT,
                   backend, mod_path, cls_name, args.split,
                   str(args.batch_size), str(args.num_workers), str(args.num_epochs)]

            if backend == "hf":
                hf_cfg = cfg["hf"]
                cmd.extend([hf_cfg[0], str(hf_cfg[1])])
            elif backend == "tv":
                tv_cfg = cfg["tv"]
                tv_root = args.imagenet_root if tv_cfg[1] == "imagenet_dir" else os.path.join(args.cache_dir, "torchvision")
                cmd.extend([tv_root, tv_cfg[1]])

            env = os.environ.copy()
            env["STABLE_DATASETS_CACHE_DIR"] = args.cache_dir
            env["HF_HOME"] = os.path.join(args.cache_dir, "huggingface")
            env["TV_DOWNLOAD"] = "1"
            if args.shuffle:
                env["STABLE_DATASETS_SHUFFLE"] = "1"

            timeout = 36000 if "imagenet" in ds_name.lower() else 3600

            all_epoch_times = []
            rss_vals = []

            for run_idx in range(args.num_runs):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=timeout, env=env)
                except subprocess.TimeoutExpired:
                    print(f"  [{label}] run {run_idx+1}: TIMEOUT", flush=True)
                    break

                stdout = result.stdout.strip()
                if not stdout:
                    err = result.stderr[:200] if result.stderr else "no output"
                    print(f"  [{label}] run {run_idx+1}: ERROR - {err[:100]}", flush=True)
                    break

                try:
                    r = json.loads(stdout.split("\n")[-1])
                except json.JSONDecodeError:
                    print(f"  [{label}] run {run_idx+1}: ERROR - {stdout[:100]}", flush=True)
                    break

                if "error" in r:
                    print(f"  [{label}] run {run_idx+1}: ERROR - {r['error'][:100]}", flush=True)
                    break

                epoch_times = r["epoch_times"]
                all_epoch_times.extend(epoch_times)
                if "uss_total" in r:
                    rss_vals.append(r)
                mean_ep = statistics.mean(epoch_times)
                uss_str = ""
                if "uss_total" in r:
                    uss_str = f", uss={r['uss_total'] / 1024 / 1024:.0f} MB (main={r['uss_main'] / 1024 / 1024:.0f}, workers={r['uss_workers'] / 1024 / 1024:.0f})"
                print(f"  [{label}] run {run_idx+1}/{args.num_runs}: "
                      f"read={fmt_time(mean_ep)}/ep x{len(epoch_times)} "
                      f"({r['n']} samples{uss_str})", flush=True)

            if all_epoch_times:
                entry = {"read": all_epoch_times}
                if rss_vals:
                    entry["memory"] = rss_vals
                results[ds_name][backend] = entry
            else:
                results[ds_name][backend] = None

        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Append to consolidated history file unless suppressed.
    if args.history and not args.no_history:
        from experiment_manager import ExperimentManager

        mgr = ExperimentManager(args.history)

        slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
        config = {
            "num_epochs": args.num_epochs,
            "num_workers": args.num_workers,
            "num_runs": args.num_runs,
            "shuffle": args.shuffle,
            "decode": os.environ.get("STABLE_DATASETS_DECODE", "1") == "1",
        }
        if slurm_mem_mb:
            config["mem_gb"] = int(slurm_mem_mb) // 1024

        appended = mgr.append(
            slurm_job_id=os.environ.get("SLURM_JOB_ID", "local"),
            config=config,
            results=results,
        )
        if appended:
            print(f"Appended to history at {mgr.path} "
                  f"(now {len(mgr.runs)} runs)")
        else:
            print(f"Skipped history append (duplicate slurm_job_id)")

    # Terminal summary
    col_w = 28
    header = f"{'Dataset':<14} " + " ".join(
        f"{BACKEND_LABELS.get(b, b):>{col_w}}" for b in backends
    )
    sep = "-" * len(header)
    print(f"\n1-Epoch Read Time (mean +/- std, {args.num_runs} x {args.num_epochs} epochs)")
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
                parts.append(f"{fmt_mean_std(r['read']):>{col_w}}")
        print(" ".join(parts))
    print(sep)


if __name__ == "__main__":
    main()
