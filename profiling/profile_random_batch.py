"""Microbench: shuffled random-batch fetch, Arrow vs Lance.

Measures the actual DataLoader map-style shuffled workload --
``backend.take(batch_of_random_indices)`` repeated many times -- which
``profile_iter.py`` with its default SequentialSampler does *not*
exercise. Also varies batch size to show whether larger batches amortize
Lance's per-call overhead.

Not a full DataLoader run (no collation, no formatter, no worker forks)
-- this isolates the storage-backend cost so we can see the underlying
cost curve without DataLoader plumbing on top.

Usage:
    python profiling/profile_random_batch.py \\
        --arrow-cache /path/to/arrow_cache_dir \\
        --lance-cache /path/to/lance_cache_dir \\
        --batch-sizes 32 128 256 1024 \\
        --num-batches 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from stable_datasets.backend import ArrowBackend, _upgrade_binary_columns
from stable_datasets.lance_backend import LanceBackend


def _load_arrow(cache_dir: Path) -> ArrowBackend:
    import json

    meta = json.loads((cache_dir / "_metadata.json").read_text())
    shard_paths = [cache_dir / name for name in meta["shard_filenames"]]
    # Use the first shard's schema (after binary upgrade) as the backend schema.
    mmap = pa.memory_map(str(shard_paths[0]), "r")
    schema = _upgrade_binary_columns(ipc.open_file(mmap).read_all()).schema
    return ArrowBackend(
        shard_paths=shard_paths,
        shard_row_counts=meta["shard_row_counts"],
        schema=schema,
    )


def _time_take(backend, indices_list: list[np.ndarray]) -> tuple[float, int]:
    t0 = time.perf_counter()
    total_rows = 0
    for idx in indices_list:
        sub = backend.take(idx)
        total_rows += sub.num_rows
    return time.perf_counter() - t0, total_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-cache", type=Path, required=True)
    parser.add_argument("--lance-cache", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 256, 1024])
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading Arrow backend from {args.arrow_cache}")
    arrow = _load_arrow(args.arrow_cache)
    print(f"Loading Lance backend from {args.lance_cache}")
    lance = LanceBackend(uri=args.lance_cache)

    n = arrow.num_rows
    assert lance.num_rows == n, f"row count mismatch: arrow={n} lance={lance.num_rows}"
    print(f"Both backends report num_rows={n}")

    print(
        f"\nShuffled random-batch fetch microbench "
        f"({args.num_batches} batches per size, best of {args.repeat})\n"
    )
    header = (
        f"{'batch_size':>10}  "
        f"{'arrow (ms/batch)':>20}  "
        f"{'lance (ms/batch)':>20}  "
        f"{'ratio (lance/arrow)':>22}"
    )
    print(header)
    print("-" * len(header))

    for bs in args.batch_sizes:
        rng = np.random.default_rng(args.seed)
        indices_list = [
            rng.integers(0, n, size=bs, dtype=np.int64)
            for _ in range(args.num_batches)
        ]

        arrow_best = float("inf")
        lance_best = float("inf")
        for _ in range(args.repeat):
            dt, rows = _time_take(arrow, indices_list)
            assert rows == bs * args.num_batches, f"arrow row mismatch: {rows}"
            arrow_best = min(arrow_best, dt)

            dt, rows = _time_take(lance, indices_list)
            assert rows == bs * args.num_batches, f"lance row mismatch: {rows}"
            lance_best = min(lance_best, dt)

        arrow_ms = arrow_best / args.num_batches * 1000
        lance_ms = lance_best / args.num_batches * 1000
        ratio = lance_ms / arrow_ms
        print(
            f"{bs:>10}  "
            f"{arrow_ms:>20.3f}  "
            f"{lance_ms:>20.3f}  "
            f"{ratio:>22.2f}x"
        )


if __name__ == "__main__":
    main()
