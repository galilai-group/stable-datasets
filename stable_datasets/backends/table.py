"""In-memory implementation of :class:`StorageBackend`.

:class:`TableBackend` serves a ``pa.Table`` held in memory: derived subsets,
column mutations and small datasets built from Python objects. Caches on
disk are Lance (:class:`LanceBackend`). Arrow stays inside: reads return
:class:`Columns`, and the column operations take Python or numpy values.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pyarrow as pa

from ._arrow import ArrowReads
from ._indexed_arrow_table import IndexedArrowTable


class TableBackend(ArrowReads):
    """A ``pa.Table`` in memory, read through an index of its record batches."""

    # DataLoader ``__getitems__`` uses batched gather.
    prefer_batched_take: bool = True

    def __init__(self, table: pa.Table, schema: pa.Schema | None = None):
        self._table = table
        self._schema = schema
        self._indexed_table = IndexedArrowTable(table)

    @staticmethod
    def from_columns(data, features) -> TableBackend:
        """Columns of Python or numpy values (2-D numpy arrays as lists of rows), typed by ``features``."""
        schema = features.to_arrow_schema()
        arrays = {}
        for field in schema:
            values = data[field.name]
            if isinstance(values, np.ndarray) and values.ndim > 1:
                values = list(values)
            arrays[field.name] = pa.array(values, type=field.type)
        return TableBackend(pa.table(arrays, schema=schema), schema=schema)

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

    def materialize(self, indices: np.ndarray | None = None) -> TableBackend:
        return self if indices is None else super().materialize(indices)

    # -- Column operations (each returns a new backend) ------------------------

    def with_column(self, name: str, values) -> TableBackend:
        """A backend with column ``name`` appended (a list, a numpy array or an Arrow array)."""
        array = values if isinstance(values, pa.Array | pa.ChunkedArray) else pa.array(values)
        return TableBackend(self._table.append_column(name, array))

    def drop_columns(self, names: list[str]) -> TableBackend:
        return TableBackend(self._table.drop_columns(names))

    def rename_columns(self, mapping: dict[str, str]) -> TableBackend:
        return TableBackend(self._table.rename_columns([mapping.get(n, n) for n in self._table.column_names]))

    def typed(self, features) -> TableBackend:
        """The same rows, with the Arrow schema of ``features``."""
        return TableBackend(self._table, schema=features.to_arrow_schema())

    def feature_of(self, name: str):
        """The feature type of a stored column, inferred from its Arrow type."""
        from ..schema import infer_feature

        return infer_feature(self.schema.field(name).type)

    def get_row(self, idx: int) -> dict:
        """Single row access. Uses slice() which avoids copying binary data."""
        row_slice = self._table.slice(idx, 1).to_pydict()
        return {k: v[0] for k, v in row_slice.items()}

    def _arrow_take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Batched row access without using ``pa.Table.take``."""
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.size == 0:
            return pa.Table.from_batches([], schema=self.schema)
        contiguous = _as_contiguous_slice(index_array)
        if contiguous is not None:
            return self._indexed_table.fast_slice(*contiguous)
        return self._indexed_table.fast_gather(index_array)

    def _arrow_slice(self, start: int, length: int) -> pa.Table:
        return self._indexed_table.fast_slice(start, length)

    def _arrow_batches(
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
