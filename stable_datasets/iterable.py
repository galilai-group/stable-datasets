"""Iterable dataset for streaming with worker sharding and buffered shuffle.

Provides ``StableIterableDataset`` for efficient streaming in PyTorch
DataLoader with multiple workers.  Supports shard-level worker partitioning
and reservoir-based row-level shuffle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa


try:
    import torch
    from torch.utils.data import IterableDataset as _IterableBase
except ImportError:
    torch = None

    class _IterableBase:
        """Fallback base when PyTorch is not installed."""

        pass


class StableIterableDataset(_IterableBase):
    """An iterable-style dataset with worker sharding and buffered shuffle.

    Wraps a ``StableDataset`` for efficient streaming in PyTorch DataLoader
    with multiple workers.  Shards are partitioned across workers so each
    worker reads a disjoint subset.

    The rows are the rows of the wrapped dataset. The index view, the
    format, the video decode config and the transform of the dataset
    apply.

    Each pass yields every row exactly once over all distributed ranks and
    DataLoader workers. All partitions use the same shard order for an
    epoch. Each partition then reads a disjoint part of that order.
    Partitions can get different numbers of rows, so ranks can get
    different numbers of batches.

    The epoch advances by one on each pass in the main process and in
    persistent DataLoader workers. With ``persistent_workers=False``, call
    :meth:`set_epoch` before each pass.

    Parameters
    ----------
    dataset : StableDataset
        The underlying map-style dataset.
    shuffle : bool
        Whether to shuffle shard order and apply buffered row-level shuffle.
    seed : int
        Base random seed. Use the same seed on all ranks.
    buffer_size : int
        Size of the reservoir buffer for row-level shuffle.
    transform : callable, optional
        Transform applied to each yielded row dict.
    rank : int, optional
        Rank of this process. Defaults to the ``torch.distributed`` rank,
        or 0. Give ``rank`` and ``world_size`` together.
    world_size : int, optional
        Number of ranks. Defaults to the ``torch.distributed`` world size,
        or 1.
    """

    def __init__(
        self,
        dataset,
        *,
        shuffle: bool = False,
        seed: int = 0,
        buffer_size: int = 10_000,
        transform: Callable | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        if (rank is None) != (world_size is None):
            raise ValueError("Give rank and world_size together, or give neither.")
        if rank is not None and not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}")
        self._dataset = dataset
        self._shuffle = shuffle
        self._seed = int(seed)
        self._buffer_size = buffer_size
        self._transform = transform
        self._rank = rank
        self._world_size = world_size
        # Hold [epoch, set_epoch call count] in shared memory so that
        # persistent DataLoader workers see each set_epoch call.
        if torch is not None:
            self._epoch_state = torch.zeros(2, dtype=torch.int64).share_memory_()
        else:
            self._epoch_state = np.zeros(2, dtype=np.int64)
        self._seen_set_epoch_calls = 0
        self._passes = 0

    def set_epoch(self, epoch: int):
        """Set the epoch of the next pass.

        Each pass after this call advances the epoch by one. The DataLoader
        makes new workers for each pass when ``persistent_workers=False``.
        These workers do not see earlier passes, so call this method before
        each pass in that case.
        """
        self._epoch_state[0] = int(epoch)
        self._epoch_state[1] += 1

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # A worker that the DataLoader starts with spawn has no process group.
        # Record the rank of this process before the worker starts.
        if state["_rank"] is None:
            state["_rank"], state["_world_size"] = _distributed_rank()
        return state

    def __iter__(self):
        ds = self._dataset
        backend = ds._backend
        epoch = self._next_epoch()
        partition, num_partitions = self._partition()

        # All partitions use the same order. Only the row buffer seed changes per partition.
        order_rng = np.random.default_rng(np.random.SeedSequence([self._seed, epoch]))
        row_rng = np.random.default_rng(np.random.SeedSequence([self._seed, epoch, partition]))

        if ds._indices is not None or not backend.is_file_backed:
            positions = np.arange(len(ds), dtype=np.int64)
            if self._shuffle:
                order_rng.shuffle(positions)
            row_gen = ds._iter_positions(np.array_split(positions, num_partitions)[partition])
        else:
            shards = np.arange(backend.num_shards)
            if self._shuffle:
                order_rng.shuffle(shards)
            if len(shards) >= num_partitions:
                row_gen = self._iter_shard_rows(shards[partition::num_partitions])
            else:
                row_gen = self._iter_shard_rows(shards, partition=partition, num_partitions=num_partitions)

        if self._shuffle and self._buffer_size > 0:
            yield from self._buffered_shuffle(row_gen, row_rng)
        else:
            for row in row_gen:
                if self._transform:
                    row = self._transform(row)
                yield row

    def _next_epoch(self) -> int:
        """Return the epoch of this pass and advance the pass count."""
        epoch, set_epoch_calls = (int(v) for v in self._epoch_state)
        if set_epoch_calls != self._seen_set_epoch_calls:
            self._seen_set_epoch_calls = set_epoch_calls
            self._passes = 0
        self._passes += 1
        return epoch + self._passes - 1

    def _partition(self) -> tuple[int, int]:
        """Return the index of this (rank, worker) pair and the number of pairs."""
        rank, world_size = self._rank, self._world_size
        if rank is None:
            rank, world_size = _distributed_rank()
        rank = 0 if rank is None else int(rank)
        world_size = 1 if world_size is None else int(world_size)
        worker_id, num_workers = 0, 1
        if torch is not None:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
            if worker_info is not None:
                worker_id, num_workers = worker_info.id, worker_info.num_workers
        return rank * num_workers + worker_id, world_size * num_workers

    def _iter_shard_rows(self, shard_indices, *, partition: int = 0, num_partitions: int = 1):
        """Yield the dataset rows of the given shards.

        Keep only the rows whose logical index is ``partition`` modulo
        ``num_partitions``. Each row goes through the dataset row pipeline.
        """
        ds = self._dataset
        shard_starts = np.cumsum([0, *ds._backend.shard_row_counts])
        for shard_id in shard_indices:
            offset = int(shard_starts[shard_id])
            for batch in ds._backend.iter_batches(shard_indices=[int(shard_id)]):
                positions = np.arange(offset, offset + batch.num_rows)
                offset += batch.num_rows
                if num_partitions > 1:
                    keep = positions % num_partitions == partition
                    batch = batch.filter(pa.array(keep))
                    positions = positions[keep]
                yield from ds._format_table(batch, positions.tolist())

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


def _distributed_rank() -> tuple[int | None, int | None]:
    """Return the ``torch.distributed`` rank and world size, or ``(None, None)``."""
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return None, None
