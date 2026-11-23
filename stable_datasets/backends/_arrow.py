"""The part every Arrow-based backend shares: Arrow batches in, :class:`Columns` out.

A backend implements ``_arrow_take``, ``_arrow_slice`` and ``_arrow_batches`` (Arrow tables and record batches)
and the ``schema`` and ``table`` properties. Arrow does not cross the backend boundary: :class:`StableDataset`
and the formatters see :class:`Columns`, numpy arrays and feature types only.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pyarrow as pa

from .columns import Columns


def columns_of(batch: pa.Table | pa.RecordBatch) -> Columns:
    return Columns({name: batch.column(name).to_pylist() for name in batch.column_names}, batch.num_rows)


def _values(array) -> np.ndarray | list:
    """numpy for fixed-width types without nulls, else Python values."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    t = array.type
    if array.null_count == 0 and (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)):
        return array.to_numpy(zero_copy_only=False)
    return array.to_pylist()


class ArrowReads:
    """Mixin: the public, Arrow-free read methods of the storage backend protocol."""

    @property
    def column_names(self) -> list[str]:
        return list(self.schema.names)

    def take(self, indices: np.ndarray | list[int]) -> Columns:
        return columns_of(self._arrow_take(indices))

    def slice(self, start: int, length: int) -> Columns:
        return columns_of(self._arrow_slice(start, length))

    def iter_batches(
        self, shard_indices: list[int] | None = None, shuffle: bool = False, seed: int | None = None
    ) -> Iterator[Columns]:
        for batch in self._arrow_batches(shard_indices=shard_indices, shuffle=shuffle, seed=seed):
            yield columns_of(batch)

    def column(self, name: str, indices: np.ndarray | None = None) -> np.ndarray | list:
        """One column's stored values at ``indices`` (all rows if None): numpy for fixed-width types, else a list."""
        if indices is None:
            return _values(self._arrow_slice(0, self.num_rows).column(name))
        return _values(self._arrow_take(np.asarray(indices, np.int64)).column(name))

    def materialize(self, indices: np.ndarray | None = None):
        """The rows at ``indices`` (all rows if None) in memory, as a :class:`TableBackend`."""
        from .table import TableBackend

        table = self.table if indices is None else self._arrow_take(np.asarray(indices, np.int64))
        return TableBackend(table, schema=self.schema)
