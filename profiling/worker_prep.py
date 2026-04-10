"""Subprocess worker for prep-time profiling.

Loads a dataset (triggering cache build if needed) and reports
timing + memory (USS) as JSON on stdout.

Usage (called by profile_prep.py, not directly):
    python worker_prep.py <backend> <mod_path> <cls_name> <split> [extra...]
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import psutil


backend = sys.argv[1]
mod_path = sys.argv[2]
cls_name = sys.argv[3]
split = sys.argv[4]
extra = sys.argv[5:]

try:
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

    prep = time.perf_counter() - t0
    n = len(ds)

    proc = psutil.Process()
    mem = proc.memory_full_info()
    uss = mem.uss
    rss = mem.rss

    print(json.dumps({"prep": prep, "n": n, "uss": uss, "rss": rss}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
