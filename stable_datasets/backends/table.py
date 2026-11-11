"""In-memory implementation of :class:`StorageBackend`.

:class:`TableBackend` serves a ``pa.Table`` held in memory: derived subsets,
column mutations and small datasets built from Python objects. Caches on
disk are Lance (:class:`LanceBackend`). Returns Arrow-native types
(:class:`pa.Table`, :class:`pa.RecordBatch`) and plain Python dicts;
decoding to user-facing types is the formatter's job.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pyarrow as pa

from ._indexed_arrow_table import IndexedArrowTable


class TableBackend:
    """A ``pa.Table`` in memory, read through an index of its record batches."""

    # DataLoader ``__getitems__`` uses batched gather.
    prefer_batched_take: bool = True

    def __init__(self, table: pa.Table, schema: pa.Schema | None = None):
        self._table = table
        self._schema = schema
        self._indexed_table = IndexedArrowTable(table)

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    @property
    def is_file_backed(self) -> bool:
        return False

    @property
    def cache_dir(self) -> None:
        return None

    @property
    def schema(self) -> pa.Schema:
        return self._schema if self._schema is not None else self._table.schema

    @property
    def table(self) -> pa.Table:
        return self._table

    def get_row(self, idx: int) -> dict:
        """Single row access. Uses slice() which avoids copying binary data."""
        row_slice = self._table.slice(idx, 1).to_pydict()
        return {k: v[0] for k, v in row_slice.items()}

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Batched row access without using ``pa.Table.take``."""
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.size == 0:
            return pa.Table.from_batches([], schema=self.schema)
        contiguous = _as_contiguous_slice(index_array)
        if contiguous is not None:
            return self._indexed_table.fast_slice(*contiguous)
        return self._indexed_table.fast_gather(index_array)

    def slice(self, start: int, length: int) -> pa.Table:
        return self._indexed_table.fast_slice(start, length)

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        yield from self._table.to_batches()

    @property
    def num_shards(self) -> int:
        return 0

    @property
    def shard_row_counts(self) -> list[int]:
        return []

    def __getstate__(self):
        return {"table": self._table, "schema": self._schema}

    def __setstate__(self, state):
        self.__init__(state["table"], state.get("schema"))


def _as_contiguous_slice(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 1:
        return int(indices[0]), 1
    if np.all(np.diff(indices) == 1):
        return int(indices[0]), int(indices.size)
    return None
