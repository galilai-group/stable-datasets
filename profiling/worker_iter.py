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

    decode_on = os.environ.get("STABLE_DATASETS_DECODE", "1") == "1"

    if backend == "stable":
        import importlib

        mod = importlib.import_module(mod_path)
        ds = getattr(mod, cls_name)(split=split)
        if not decode_on:
            ds = ds.set_decode(False)

    elif backend == "stable_lance":
        # Exercise the Phase C Lance writer path via the public
        # runtime-override kwarg. First run on each dataset pays a
        # cache miss through write_lance_cache; subsequent runs hit
        # the Lance cache via BaseDatasetBuilder's normal cache-hit
        # branch. Arrow and Lance caches coexist at distinct paths
        # because cache_fingerprint hashes the format.
        import importlib

        mod = importlib.import_module(mod_path)
        ds = getattr(mod, cls_name)(split=split, storage_format="lance")
        if not decode_on:
            ds = ds.set_decode(False)

    elif backend == "hf":
        import datasets as hf_datasets

        hub_path = extra[0]
        config_name = extra[1] if len(extra) > 1 and extra[1] != "None" else None
        kw = {"name": config_name} if config_name else {}
        ds = hf_datasets.load_dataset(hub_path, split=split, **kw)
        if not decode_on:
            for col, feat in ds.features.items():
                if isinstance(feat, hf_datasets.Image):
                    ds = ds.cast_column(col, hf_datasets.Image(decode=False))
        # HF iterable path: convert to IterableDataset so the comparison
        # against our StableIterableDataset is apples-to-apples (both
        # streaming, both with in-dataset buffered shuffle).
        if os.environ.get("STABLE_DATASETS_ITERABLE", "0") == "1":
            ds = ds.to_iterable_dataset(num_shards=max(1, num_workers))
            if os.environ.get("STABLE_DATASETS_SHUFFLE", "0") == "1":
                ds = ds.shuffle(seed=0, buffer_size=10_000)

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

    shuffle = os.environ.get("STABLE_DATASETS_SHUFFLE", "0") == "1"
    iterable = os.environ.get("STABLE_DATASETS_ITERABLE", "0") == "1"
    sampler_kind = os.environ.get("STABLE_DATASETS_SAMPLER", "")
    persistent = os.environ.get("STABLE_DATASETS_PERSISTENT_WORKERS", "1") == "1"
    mp_ctx = os.environ.get("STABLE_DATASETS_MP_CONTEXT", "spawn")

    # Map-style vs iterable-style. For iterable we wrap the StableDataset
    # in StableIterableDataset (streaming via backend.iter_batches) and
    # turn off DataLoader-side shuffle (shuffle is the dataset's
    # responsibility via buffered reservoir shuffle). HF datasets stay
    # on their native map-style path for iterable too; we only restructure
    # for our own backends.
    if iterable and backend.startswith("stable"):
        from stable_datasets.iterable import StableIterableDataset

        ds = StableIterableDataset(ds, shuffle=shuffle, seed=0, buffer_size=10_000)
        shuffle = False  # DataLoader can't shuffle an IterableDataset

    # Build the optional backend-aware sampler.
    sampler = None
    if sampler_kind:
        if iterable:
            raise ValueError("--sampler is not supported on iterable-style datasets")
        if backend.startswith("stable"):
            sampler = ds.make_sampler(sampler_kind, seed=0, within_shard="random")
            shuffle = False  # sampler is mutually exclusive with shuffle=True

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_list_collate,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent
        loader_kwargs["multiprocessing_context"] = mp_ctx
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle

    loader = DataLoader(ds, **loader_kwargs)

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
