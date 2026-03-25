"""Iterable dataset for streaming with worker sharding and buffered shuffle.

Provides ``StableIterableDataset`` for efficient streaming in PyTorch
DataLoader with multiple workers.  Supports shard-level worker partitioning
and reservoir-based row-level shuffle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


try:
    from torch.utils.data import IterableDataset as _IterableBase
except ImportError:

    class _IterableBase:
        """Fallback base when PyTorch is not installed."""

        pass


class StableIterableDataset(_IterableBase):
    """An iterable-style dataset with worker sharding and buffered shuffle.

    Wraps a ``StableDataset`` for efficient streaming in PyTorch DataLoader
    with multiple workers.  Shards are partitioned across workers so each
    worker reads a disjoint subset.

    Parameters
    ----------
    dataset : StableDataset
        The underlying map-style dataset (must be shard-backed).
    shuffle : bool
        Whether to shuffle shard order and apply buffered row-level shuffle.
    seed : int
        Base random seed.
    buffer_size : int
        Size of the reservoir buffer for row-level shuffle.
    transform : callable, optional
        Transform applied to each yielded row dict.
    """

    def __init__(
        self,
        dataset,
        *,
        shuffle: bool = False,
        seed: int = 0,
        buffer_size: int = 10_000,
        transform: Callable | None = None,
    ):
        self._dataset = dataset
        self._shuffle = shuffle
        self._seed = seed
        self._buffer_size = buffer_size
        self._transform = transform
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Set the epoch for varying shuffle seed across epochs."""
        self._epoch = epoch

    def __iter__(self):
        ds = self._dataset

        # Non-shard-backed fallback
        if not ds._is_shard_backed:
            for i in range(len(ds)):
                row = ds[i]
                if self._transform:
                    row = self._transform(row)
                yield row
            return

        # Determine worker sharding
        try:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
        except ImportError:
            worker_info = None

        all_shards = list(range(len(ds._shard_paths)))
        if worker_info is not None:
            my_shards = all_shards[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id
        else:
            my_shards = all_shards
            worker_id = 0

        effective_seed = self._seed + self._epoch * 1000 + worker_id
        rng = np.random.default_rng(effective_seed) if self._shuffle else None

        if self._shuffle and rng is not None:
            rng.shuffle(my_shards)

        formatter = ds._formatter

        def _row_gen():
            for batch in ds._backend.iter_batches(shard_indices=my_shards):
                batch_dict = batch.to_pydict()
                n = batch.num_rows
                for i in range(n):
                    row = {k: v[i] for k, v in batch_dict.items()}
                    yield formatter.format_row(row)

        if self._shuffle and self._buffer_size > 0:
            yield from self._buffered_shuffle(_row_gen(), rng)
        else:
            for row in _row_gen():
                if self._transform:
                    row = self._transform(row)
                yield row

    def _buffered_shuffle(self, row_gen, rng):
        """Reservoir-based buffered shuffle (Fisher-Yates).

        Fills a buffer from the row generator, then yields random elements
        as new rows arrive to maintain the buffer at capacity.
        """
        buffer = []
        for row in row_gen:
            if len(buffer) < self._buffer_size:
                buffer.append(row)
            else:
                idx = rng.integers(0, len(buffer))
                out = buffer[idx]
                buffer[idx] = row
                if self._transform:
                    out = self._transform(out)
                yield out
        # Flush remaining buffer in random order
        rng.shuffle(buffer)
        for row in buffer:
            if self._transform:
                row = self._transform(row)
            yield row
