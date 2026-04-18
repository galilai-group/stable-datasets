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
dataset regardless of whether the column was blob-encoded. A future
blob-aware subclass or flag can be added if profiling justifies it.

**Pickling is cheap.** ``__getstate__`` serializes only the dataset URI
plus cached row/shard counts; ``__setstate__`` reopens by URI via
``lance.dataset(...)``. This keeps DataLoader worker-fork cost constant
regardless of dataset size, unlike an ``ArrowBackend`` whose mmap'd
table state can be tens of GB.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa


class LanceBackend:
    #: Hint to :class:`StableDataset.__getitems__` that this backend's
    #: batched ``take(indices)`` path is faster than a per-row loop of
    #: ``get_row``. ArrowBackend leaves this unset (effectively False)
    #: because Arrow's ``slice(i, 1)`` on an mmap'd table is zero-copy
    #: and unbeatable per-row, while Arrow's ``take`` rebuilds offsets
    #: across chunks. For Lance the cost structure is inverted: every
    #: call crosses the Python<->Rust async boundary at fixed cost, so
    #: batching amortizes it. Benchmark: CIFAR-10 map-style shuffled
    #: access, 128-row batches, Lance batched is ~30x faster than
    #: Lance per-row.
    prefer_batched_take: bool = True

    def __init__(self, *, uri: str | Path, batch_readahead: int = 8):
        """
        Parameters
        ----------
        uri : str or Path
            Path to the Lance dataset directory.
        batch_readahead : int, default 8
            Number of RecordBatches Lance reads ahead in the scanner
            when ``iter_batches`` is called. Matches Lance's own
            ``lance.torch.data.LanceDataset`` example which uses
            ``batch_readahead=8``. Higher values increase memory use
            during iteration but improve throughput on high-latency
            storage. Ignored by ``take``/``get_row``/``slice``.
        """
        self._uri = Path(uri)
        self._ds = None  # opened lazily so DataLoader workers re-open after fork
        self._cached_num_rows: int | None = None
        self._cached_num_shards: int | None = None
        self._batch_readahead = int(batch_readahead)

    # -- Lazy open ------------------------------------------------------------
    #
    # Fork-safety contract: opening the Lance dataset initializes Lance's
    # Rust tokio runtime, which includes a worker thread pool. ``fork()``
    # does not duplicate threads -- only the main thread survives in the
    # child. A child process that inherits an already-initialized tokio
    # state but only has one thread will segfault on its first Lance
    # call. Therefore ``_dataset`` should NOT be touched in the main
    # process before a DataLoader fork. Every public accessor that
    # triggers it (``num_rows``, ``num_shards``, ``schema``, ``table``,
    # ``take``, etc.) is a potential footgun when called pre-fork.
    #
    # Callers that need metadata pre-fork should cache it at construction
    # time and pass it explicitly. ``StableDataset._shallow_copy`` is
    # fixed to forward ``num_rows`` for this reason.

    @property
    def _dataset(self):
        if self._ds is None:
            import lance

            self._ds = lance.dataset(str(self._uri))
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
    def is_file_backed(self) -> bool:
        return True

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
        """
        fragments = self._dataset.get_fragments()

        if shard_indices is not None:
            fragments = [fragments[i] for i in shard_indices]

        if shuffle and seed is not None:
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
            "uri": str(self._uri),
            "num_rows": self._cached_num_rows,
            "num_shards": self._cached_num_shards,
            "batch_readahead": self._batch_readahead,
        }

    def __setstate__(self, state: dict) -> None:
        self._uri = Path(state["uri"])
        self._ds = None
        self._cached_num_rows = state.get("num_rows")
        self._cached_num_shards = state.get("num_shards")
        self._batch_readahead = state.get("batch_readahead", 8)
