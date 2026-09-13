"""Sequentially materialize the processed cache for every benchmark dataset.

Run BEFORE training sweeps so concurrent SLURM workers find a cache hit and
skip the (race-prone) build path. If a cache shard already exists, the
underlying ``BaseDatasetBuilder.__new__`` short-circuits and this script
returns in <1s for that dataset.

Usage:
    .venv/bin/python -m benchmarks.prewarm                # all included image datasets
    .venv/bin/python -m benchmarks.prewarm emnist_mnist   # one dataset
    .venv/bin/python -m benchmarks.prewarm fgvcaircraft_family,hasyv2
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback

from benchmarks.dataset import (
    INCLUDED_IMAGE_DATASETS,
    _get_dataset_class,
    _with_data_dirs,
    get_config,
)

# Must resolve to the SAME root training uses, or prewarming is a no-op: run.py
# passes cfg.data_dir (conf/config.yaml -> $STABLE_DATASETS_ROOT, default
# ~/scratch/stable-datasets-iclr) down to _with_data_dirs(). This script used to
# default to a repo-relative "./.anonymous-datasets-cache", which populated a
# cache under HOME that no training run ever read -- and HOME is a 100 GB quota
# you do not want several hundred GB of datasets in. ~/scratch is a symlink to
# /oscar/scratch/$USER, so both spellings are the same directory.
DATA_DIR = os.environ.get(
    "STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prewarm")


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _report_existing() -> None:
    """Show what is already cached so a re-run is visibly incremental.

    Prewarm is idempotent -- BaseDatasetBuilder.__new__ short-circuits on a
    cache hit -- so re-running only fetches what is missing. This just makes
    that legible instead of leaving you wondering if it is re-downloading.
    """
    for sub in ("downloads", "processed"):
        d = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(d):
            log.info(f"  {sub}/: (absent -- will be created)")
            continue
        entries = sorted(e for e in os.listdir(d) if not e.startswith("."))
        size = _dir_size(d)
        log.info(f"  {sub}/: {len(entries)} entries, {size / 1e9:.1f} GB")
        if entries:
            log.info(f"    {', '.join(entries[:12])}{' ...' if len(entries) > 12 else ''}")


def prewarm(name: str) -> tuple[str, float, str]:
    """Instantiate train + validation splits for ``name``. Returns (status, elapsed_s, msg)."""
    t0 = time.time()
    try:
        cfg = get_config(name)
        cls = _get_dataset_class(cfg)
        kwargs = _with_data_dirs(cfg, DATA_DIR)
        cls(split="train", **kwargs)
        try:
            cls(split="validation", **kwargs)
        except (ValueError, KeyError):
            try:
                cls(split="test", **kwargs)
            except (ValueError, KeyError):
                pass  # builder has no separate val/test split
        return ("ok", time.time() - t0, "")
    except Exception as e:
        return ("FAIL", time.time() - t0, f"{type(e).__name__}: {e}")


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1].split(",")
    else:
        names = sorted(INCLUDED_IMAGE_DATASETS)

    log.info(f"prewarming {len(names)} datasets to {DATA_DIR}")
    _report_existing()
    results = []
    for i, name in enumerate(names, 1):
        log.info(f"[{i}/{len(names)}] {name}")
        status, dt, msg = prewarm(name)
        results.append((name, status, dt, msg))
        log.info(f"[{i}/{len(names)}] {name}: {status} in {dt:.1f}s {msg}")

    log.info("=" * 60)
    log.info("summary")
    for name, status, dt, msg in results:
        log.info(f"  {name:30s} {status:5s} {dt:7.1f}s  {msg}")
    failed = [r for r in results if r[1] != "ok"]
    if failed:
        log.error(f"{len(failed)}/{len(results)} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
