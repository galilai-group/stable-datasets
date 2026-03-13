"""Shard-aware sampler for cache-friendly DataLoader access to sharded datasets.

Standard ``DataLoader(shuffle=True)`` uses a ``RandomSampler`` that scatters
indices across all shards, causing LRU cache thrashing on large datasets.
``ShardAwareSampler`` emits global indices grouped by shard so that
consecutive ``__getitem__`` calls hit the same shard, keeping the LRU cache
hot.

Shuffling happens at two levels: shard order is permuted, and row order
within each shard is permuted.  This gives near-global randomness as long
as shards are larger than the training batch size.

Usage::

    from stable_datasets.sampler import ShardAwareSampler

    sampler = ShardAwareSampler(dataset, shuffle=True, seed=42)
    loader = DataLoader(dataset, sampler=sampler, batch_size=128, num_workers=16)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            ...
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler

from .arrow_dataset import StableDataset


class ShardAwareSampler(Sampler[int]):
    """Emits global indices grouped by shard for cache-friendly random access.

    Parameters
    ----------
    dataset:
        A shard-backed ``StableDataset``.
    shuffle:
        If ``True``, shuffle shard order and row order within each shard.
    seed:
        Base random seed.  Combined with ``epoch`` for per-epoch variation.
    num_replicas:
        Number of DDP processes.  Auto-detected from ``torch.distributed``
        if ``None``.
    rank:
        This process's rank.  Auto-detected if ``None``.
    drop_last:
        If ``True``, each rank sees ``floor(total / num_replicas)`` samples.
        If ``False`` (default), each rank sees ``ceil(total / num_replicas)``
        samples (padding with wrapped indices if needed).
    """

    def __init__(
        self,
        dataset: StableDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ):
        if not dataset._is_shard_backed:
            raise ValueError(
                "ShardAwareSampler requires a shard-backed StableDataset. "
                "For in-memory datasets, use torch's RandomSampler instead."
            )

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self._shard_row_counts = list(dataset._shard_row_counts)
        self._cumulative_offsets = list(dataset._shard_cumulative_offsets)
        self._num_shards = len(self._shard_row_counts)
        self._total_rows = len(dataset)

        # DDP auto-detection
        if num_replicas is None or rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = num_replicas or torch.distributed.get_world_size()
                rank = rank if rank is not None else torch.distributed.get_rank()
            else:
                num_replicas = num_replicas or 1
                rank = rank if rank is not None else 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"rank {rank} is invalid for num_replicas={num_replicas}")

        self.num_replicas = num_replicas
        self.rank = rank

        # All ranks must emit the same number of samples for DDP sync.
        if self.drop_last:
            self.num_samples = self._total_rows // self.num_replicas
        else:
            self.num_samples = math.ceil(self._total_rows / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch

    def _assign_shards(self, shard_order: list[int]) -> list[int]:
        """Partition shards across ranks, returning this rank's shards."""
        n = len(shard_order)
        # Pad shard list so it's divisible by num_replicas
        remainder = n % self.num_replicas
        if remainder:
            shard_order = shard_order + shard_order[: self.num_replicas - remainder]
        return shard_order[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            shard_order = torch.randperm(self._num_shards, generator=g).tolist()
        else:
            shard_order = list(range(self._num_shards))

        my_shards = self._assign_shards(shard_order)

        # Build the full index list for this rank's shards
        indices: list[int] = []
        for shard_id in my_shards:
            shard_start = self._cumulative_offsets[shard_id]
            shard_len = self._shard_row_counts[shard_id]

            if self.shuffle:
                sg = torch.Generator()
                sg.manual_seed(self.seed + self.epoch * self._num_shards + shard_id)
                local_perm = torch.randperm(shard_len, generator=sg).tolist()
            else:
                local_perm = range(shard_len)

            for local_idx in local_perm:
                indices.append(shard_start + local_idx)

        # Pad or truncate to num_samples so all DDP ranks emit equal counts.
        if len(indices) < self.num_samples:
            # Pad by wrapping from the start (preserves shard grouping for
            # the original indices; the padding tail is small).
            indices += indices[: self.num_samples - len(indices)]
        elif len(indices) > self.num_samples:
            indices = indices[: self.num_samples]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
