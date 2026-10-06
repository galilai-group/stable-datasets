"""Lance implementation of :class:`StorageBackend`.

Lance is built on Arrow: its Python API returns ``pa.Table``,
``pa.RecordBatch``, and ``pa.Schema`` directly, with no adapter layer.
:class:`LanceBackend` is a thin wrapper over ``lance.LanceDataset`` that
exposes those Arrow return values through the same protocol as
:class:`ArrowBackend`, so :class:`StableDataset` consumes either
interchangeably.

**Read-only.** Lance is a storage format, not a mutable in-memory
table. In-memory operations on :class:`StableDataset`
(``rename_column``, ``add_column``, ``map``, derived subsets) always
produce a fresh :class:`ArrowBackend` over a ``pa.Table`` regardless of
the source backend; :class:`LanceBackend` has no ``table=...``
construction mode.

**Shards = fragments.** Lance partitions datasets internally into
*fragments*, which are the I/O units ``StableIterableDataset`` uses for
worker sharding. ``num_shards`` returns the fragment count, and
``iter_batches(shard_indices=...)`` iterates only those fragment
indices.

**Blob encoding is off by default.** Lance's blob encoding stores large
binary columns out-of-line and only pays off when paired with
``take_blobs`` and ``to_batches(blob_handling="all_binary")`` at read
time, plus per-column field metadata at write time. The read methods
here use plain ``take`` / ``to_batches``, which work for any Lance
dataset regardless of whether the column was blob-encoded.

**Pickling is URI-based.** ``__getstate__`` serializes only the dataset
URI, the storage options and cached row/shard counts; ``__setstate__``
reopens by URI via ``lance.dataset(...)``.

**URIs stay strings.** ``uri`` is a local path or a URI such as
``s3://bucket/prefix``. ``storage_options`` go to ``lance.dataset``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa

from .. import _uri


class LanceBackend:
    #: Hint to :class:`StableDataset.__getitems__` that this backend's
    #: batched ``take(indices)`` path should be used for random index
    #: reads. Batching amortizes Lance's Python/Rust call boundary.
    prefer_batched_take: bool = True

    def __init__(
        self,
        *,
        uri: str | Path,
        batch_readahead: int = 8,
        shard_row_counts: list[int] | None = None,
        storage_options: dict | None = None,
    ):
        """
        Parameters
        ----------
        uri : str or Path
            Local path or URI of the Lance dataset.
        batch_readahead : int, default 8
            Number of RecordBatches Lance reads ahead in the scanner
            when ``iter_batches`` is called. Matches Lance's own
            ``lance.torch.data.LanceDataset`` example which uses
            ``batch_readahead=8``. Higher values increase memory use
            during iteration but improve throughput on high-latency
            storage. Ignored by ``take``/``get_row``/``slice``.
        shard_row_counts : list of int, optional
            Row count of each fragment, from the cache metadata. With
            this value, the shape properties do not open the dataset.
        storage_options : dict, optional
            Object store options for ``lance.dataset``.
        """
        self._uri = os.fspath(uri)
        self._storage_options = storage_options
        self._ds = None  # opened lazily so DataLoader workers re-open after fork
        self._shard_row_counts = list(shard_row_counts) if shard_row_counts is not None else None
        self._cached_num_rows: int | None = None
        self._cached_num_shards: int | None = None
        if self._shard_row_counts is not None:
            self._cached_num_rows = sum(self._shard_row_counts)
            self._cached_num_shards = len(self._shard_row_counts)
        self._batch_readahead = int(batch_readahead)

    # -- Lazy open ------------------------------------------------------------
    #
    # The Lance dataset opens lazily in each process. Opening initializes
    # Lance's Rust runtime, so worker-based callers should pass cached
    # metadata and leave this property untouched before DataLoader forks.

    @property
    def _dataset(self):
        if self._ds is None:
            import lance

            self._ds = lance.dataset(self._uri, storage_options=self._storage_options)
        return self._ds

    # -- Shape ----------------------------------------------------------------

    @property
    def num_rows(self) -> int:
        if self._cached_num_rows is None:
            self._cached_num_rows = self._dataset.count_rows()
        return self._cached_num_rows

    @property
    def num_shards(self) -> int:
        if self._cached_num_shards is None:
            self._cached_num_shards = len(self._dataset.get_fragments())
        return self._cached_num_shards

    @property
    def shard_row_counts(self) -> list[int]:
        """Row count of each fragment, in fragment order."""
        if self._shard_row_counts is None:
            self._shard_row_counts = [fragment.count_rows() for fragment in self._dataset.get_fragments()]
        return list(self._shard_row_counts)

    @property
    def is_file_backed(self) -> bool:
        return True

    @property
    def cache_dir(self) -> Path | None:
        """Local cache directory, or ``None`` for a remote URI."""
        return _uri.local_path(self._uri)

    @property
    def schema(self) -> pa.Schema:
        return self._dataset.schema

    # -- Materialization ------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """Full materialization as a single ``pa.Table``.

        Expensive for large datasets. Use ``get_row``, ``take``,
        ``slice``, or ``iter_batches`` on hot paths.
        """
        return self._dataset.to_table()

    # -- Random access --------------------------------------------------------

    def get_row(self, idx: int) -> dict:
        sub = self._dataset.take([int(idx)])
        row = sub.to_pydict()
        return {k: v[0] for k, v in row.items()}

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        if isinstance(indices, np.ndarray):
            if indices.size == 0:
                return self.schema.empty_table()
            indices = indices.tolist()
        elif len(indices) == 0:
            return self.schema.empty_table()
        return self._dataset.take(indices)

    def slice(self, start: int, length: int) -> pa.Table:
        return self._dataset.to_table(offset=start, limit=length)

    # -- Sequential iteration -------------------------------------------------

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield record batches from Lance fragments.

        ``shard`` maps to Lance ``Fragment``. Worker partitioning in
        :class:`StableIterableDataset` works the same way as for the
        Arrow backend: each worker receives a disjoint set of fragment
        indices and iterates only those.

        With ``shuffle=True``, shuffle the fragment order. A ``seed`` of
        ``None`` gives a different order on each call.
        """
        fragments = self._dataset.get_fragments()

        if shard_indices is not None:
            fragments = [fragments[i] for i in shard_indices]

        if shuffle:
            rng = np.random.default_rng(seed)
            order = np.arange(len(fragments))
            rng.shuffle(order)
            fragments = [fragments[i] for i in order]

        for frag in fragments:
            yield from frag.to_batches(batch_readahead=self._batch_readahead)

    # -- Pickle / DataLoader compatibility ------------------------------------
    # Lance datasets reopen in constant time from a URI, so worker state
    # is just the path plus cached row/shard counts.

    def __getstate__(self) -> dict:
        return {
            "uri": self._uri,
            "storage_options": self._storage_options,
            "num_rows": self._cached_num_rows,
            "num_shards": self._cached_num_shards,
            "shard_row_counts": self._shard_row_counts,
            "batch_readahead": self._batch_readahead,
        }

    def __setstate__(self, state: dict) -> None:
        self._uri = state["uri"]
        self._storage_options = state["storage_options"]
        self._ds = None
        self._cached_num_rows = state.get("num_rows")
        self._cached_num_shards = state.get("num_shards")
        self._shard_row_counts = state.get("shard_row_counts")
        self._batch_readahead = state.get("batch_readahead", 8)
