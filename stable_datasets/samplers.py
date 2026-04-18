"""Backend-aware samplers for :class:`StableDataset`.

PyTorch's :class:`~torch.utils.data.DataLoader` constructs a
:class:`~torch.utils.data.RandomSampler` when ``shuffle=True`` is
passed. That sampler yields indices in a full-random permutation
regardless of the underlying storage backend. For file-backed
storage formats partitioned into shards (Arrow) or fragments
(Lance), full-random access destroys any per-shard I/O locality
the format was designed to exploit.

This module exposes samplers that yield indices in shard-aware
orderings, preserving the classical PyTorch API (``DataLoader(ds,
sampler=...)``) while giving the user a knob that matches the
backend's access-pattern preferences:

    from stable_datasets.samplers import ShardShuffleSampler

    ds = CIFAR10(split="train", storage_format="lance")
    sampler = ShardShuffleSampler(ds, seed=42)
    loader = DataLoader(ds, batch_size=128, sampler=sampler,
                        num_workers=8, persistent_workers=True,
                        multiprocessing_context="spawn")

``DataLoader(ds, shuffle=True)`` continues to work unchanged for
users who need bit-exact full-random ordering (e.g. classification
reproduction). Samplers here are strictly opt-in.

See also
--------
``torch.utils.data.Sampler`` : base class.
``lance.sampler.ShardedFragmentSampler`` : Lance's own fragment
   sampler for its native :class:`lance.torch.data.LanceDataset`
   integration. ``ShardShuffleSampler`` is the nearest equivalent
   exposed through our unified backend protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
from torch.utils.data import Sampler


class ShardShuffleSampler(Sampler[int]):
    """Yield indices in shard-shuffled order.

    The shard (or Lance fragment) order is randomized each epoch.
    Within each shard, indices are yielded in an order controlled
    by ``within_shard``:

    * ``"random"`` (default): indices inside a shard are themselves
      permuted. Shuffle quality is closer to full-random while
      still preserving per-shard I/O locality (all samples from
      shard *k* are emitted before any sample from shard *k+1*).
      Recommended for scientific training where shuffle quality
      matters.
    * ``"sequential"``: indices inside a shard are yielded in on-
      disk order. Maximally I/O-friendly but shuffle quality is
      coarse at the shard level. Matches the behaviour of
      ``lance.sampler.ShardedFragmentSampler``.

    Parameters
    ----------
    dataset : StableDataset
        Must expose a ``StorageBackend``-compatible ``._backend``
        with ``num_shards`` and a way to iterate per-shard row
        ranges. Non-file-backed datasets fall back to a single
        shard covering the full dataset.
    seed : int, default 0
        Base seed; the epoch is XOR'd in via :meth:`set_epoch`.
    within_shard : {"random", "sequential"}, default "random"
        Within-shard row ordering.

    Notes
    -----
    *Epoch handling*: call :meth:`set_epoch` before each epoch when
    using :class:`~torch.utils.data.distributed.DistributedSampler`
    or any other stateful epoch pattern, so the random permutation
    differs between epochs. Mirrors PyTorch's own convention.

    *Fork-safety*: the sampler holds only integers and a seed; it
    pickles trivially and is safe to use with ``num_workers>0`` and
    ``multiprocessing_context="spawn"``.
    """

    def __init__(
        self,
        dataset,
        *,
        seed: int = 0,
        within_shard: Literal["random", "sequential"] = "random",
    ):
        if within_shard not in ("random", "sequential"):
            raise ValueError(
                f"within_shard must be 'random' or 'sequential', got {within_shard!r}"
            )
        self._n = len(dataset)
        self._seed = int(seed)
        self._within_shard = within_shard
        self._epoch = 0
        self._shard_ranges = self._compute_shard_ranges(dataset)

    @staticmethod
    def _compute_shard_ranges(dataset) -> list[tuple[int, int]]:
        """Return [(start, end_exclusive), ...] per shard.

        For backends with ``num_shards`` and per-shard row counts
        (our file-backed backends), we partition the dataset's
        index space into contiguous per-shard ranges. For other
        backends (e.g. in-memory tables, indexed views), the whole
        dataset becomes a single shard.
        """
        n = len(dataset)
        backend = getattr(dataset, "_backend", None)
        if backend is None or not getattr(backend, "is_file_backed", False):
            return [(0, n)]

        # ArrowBackend stores per-shard row counts as _shard_row_counts
        # (it predates this sampler; we accept both names so future
        # public property additions work without changes here).
        shard_row_counts = getattr(
            backend, "shard_row_counts", None
        ) or getattr(backend, "_shard_row_counts", None)
        if shard_row_counts is None:
            # Lance: read from fragments. ``_dataset`` opens lazily;
            # it's safe to touch here because the sampler is
            # constructed in the same process that's about to hand
            # it to DataLoader, and DataLoader workers never
            # reconstruct samplers -- they inherit an already-
            # instantiated one via pickle.
            try:
                fragments = backend._dataset.get_fragments()
                shard_row_counts = [f.count_rows() for f in fragments]
            except Exception:
                return [(0, n)]

        ranges: list[tuple[int, int]] = []
        start = 0
        for c in shard_row_counts:
            ranges.append((start, start + int(c)))
            start += int(c)
        if start != n:
            # Defensive: if shard row counts don't sum to len, fall
            # back to a single shard rather than risk missing rows.
            return [(0, n)]
        return ranges

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed ^ (self._epoch * 0x9E3779B1))

        shard_order = np.arange(len(self._shard_ranges))
        rng.shuffle(shard_order)

        for shard_idx in shard_order:
            start, end = self._shard_ranges[shard_idx]
            shard_indices = np.arange(start, end)
            if self._within_shard == "random":
                rng.shuffle(shard_indices)
            for idx in shard_indices:
                yield int(idx)
