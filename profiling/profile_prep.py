#!/usr/bin/env python3
"""Prep-time-only benchmark: measures cache-build time with raw data pre-downloaded.

Times from import to having a usable dataset, excluding download time.
No DataLoader iteration — just prep.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CACHE_DIR = "/users/sboughan/scratch/.stable-datasets"
_WORKER_SCRIPT = str(Path(__file__).parent / "worker_prep.py")

DATASETS = {
    "CIFAR-10": {
        "stable": ("stable_datasets.images.cifar10", "CIFAR10"),
        "hf": ("cifar10", None),
    },
    "CIFAR-100": {
        "stable": ("stable_datasets.images.cifar100", "CIFAR100"),
        "hf": ("cifar100", None),
    },
    "FashionMNIST": {
        "stable": ("stable_datasets.images.fashion_mnist", "FashionMNIST"),
        "hf": ("fashion_mnist", None),
    },
    "SVHN": {
        "stable": ("stable_datasets.images.svhn", "SVHN"),
        "hf": ("svhn", "cropped_digits"),
    },
    "STL-10": {
        "stable": ("stable_datasets.images.stl10", "STL10"),
        "hf": ("tanganke/stl10", None),
    },
    "Flowers102": {
        "stable": ("stable_datasets.images.flowers102", "Flowers102"),
        "hf": ("nelorth/oxford-flowers", None),
    },
    "ImageNet-1K": {
        "stable": ("stable_datasets.images.imagenet_1k", "ImageNet1K"),
        "hf": ("ILSVRC/imagenet-1k", None),
    },
}



def fmt_time(seconds):
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prep-time-only benchmark")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--backends", nargs="+", default=["stable", "hf"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())

    print(f"Prep-only benchmark, split={args.split}")
    print(f"Datasets: {ds_names}")
    print(f"Backends: {args.backends}\n")

    results = {}

    for ds_name in ds_names:
        cfg = DATASETS[ds_name]
        results[ds_name] = {}
        print(f"{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        for backend in args.backends:
            if cfg.get(backend) is None:
                results[ds_name][backend] = None
                print(f"  [{backend}] n/a", flush=True)
                continue

            stable_cfg = cfg.get("stable")
            mod_path = stable_cfg[0] if stable_cfg else "n/a"
            cls_name = stable_cfg[1] if stable_cfg else "n/a"

            cmd = [sys.executable, _WORKER_SCRIPT,
                   backend, mod_path, cls_name, args.split]

            if backend == "hf":
                hf_cfg = cfg["hf"]
                cmd.extend([hf_cfg[0], str(hf_cfg[1])])

            env = os.environ.copy()
            env["STABLE_DATASETS_CACHE_DIR"] = args.cache_dir
            env["HF_HOME"] = os.path.join(args.cache_dir, "huggingface")

            timeout = 36000 if "imagenet" in ds_name.lower() else 3600

            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                results[ds_name][backend] = {"error": f"timeout ({timeout}s)"}
                print(f"  [{backend}] TIMEOUT", flush=True)
                continue

            stdout = result.stdout.strip()
            if not stdout:
                err = result.stderr[:200] if result.stderr else "no output"
                results[ds_name][backend] = {"error": err}
                print(f"  [{backend}] ERROR: {err[:100]}", flush=True)
                continue

            try:
                r = json.loads(stdout.split("\n")[-1])
            except json.JSONDecodeError:
                results[ds_name][backend] = {"error": stdout[:200]}
                print(f"  [{backend}] ERROR: {stdout[:100]}", flush=True)
                continue

            if "error" in r:
                results[ds_name][backend] = r
                print(f"  [{backend}] ERROR: {r['error'][:100]}", flush=True)
            else:
                results[ds_name][backend] = r
                uss_mb = r['uss'] / 1024 / 1024
                rss_mb = r['rss'] / 1024 / 1024
                print(f"  [{backend}] prep={fmt_time(r['prep'])} "
                      f"({r['n']} samples, uss={uss_mb:.0f} MB, rss={rss_mb:.0f} MB)", flush=True)

        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
