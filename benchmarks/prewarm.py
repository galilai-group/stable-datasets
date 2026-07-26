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
import sys
import time
import traceback

from benchmarks.dataset import (
    INCLUDED_IMAGE_DATASETS,
    _get_dataset_class,
    _with_data_dirs,
    get_config,
)

DATA_DIR = "./.anonymous-datasets-cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prewarm")


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
