"""Subprocess worker for iteration profiling.

Loads a dataset, iterates through a DataLoader for N epochs,
and reports timing + memory (USS) as JSON on stdout.

Usage (called by profile_iter.py, not directly):
    python worker_iter.py <backend> <mod_path> <cls_name> <split>
                          <batch_size> <num_workers> <num_epochs> [extra...]
"""

import gc
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import psutil
import torch
from torch.utils.data import DataLoader


def _list_collate(batch):
    return batch


backend = sys.argv[1]
mod_path = sys.argv[2]
cls_name = sys.argv[3]
split = sys.argv[4]
batch_size = int(sys.argv[5])
num_workers = int(sys.argv[6])
num_epochs = int(sys.argv[7])
extra = sys.argv[8:]

try:
    gc.collect()

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

    n = len(ds)

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_list_collate
    )

    epoch_times = []
    for epoch in range(num_epochs):
        gc.collect()
        t0 = time.perf_counter()
        for _ in loader:
            pass
        epoch_times.append(time.perf_counter() - t0)

    # Memory: USS (private only, excludes shared mmap pages) and RSS for reference
    proc = psutil.Process()
    mem = proc.memory_full_info()
    uss_main = mem.uss
    rss_main = mem.rss

    # Measure worker USS
    uss_workers = 0
    rss_workers = 0
    for child in proc.children(recursive=True):
        try:
            cmem = child.memory_full_info()
            uss_workers += cmem.uss
            rss_workers += cmem.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    print(
        json.dumps(
            {
                "epoch_times": epoch_times,
                "n": n,
                "uss_main": uss_main,
                "uss_workers": uss_workers,
                "uss_total": uss_main + uss_workers,
                "rss_main": rss_main,
                "rss_workers": rss_workers,
                "rss_total": rss_main + rss_workers,
            }
        )
    )

except Exception as e:
    print(json.dumps({"error": str(e)}))
