"""Arrow IPC implementation of :class:`StorageBackend`.

:class:`ArrowBackend` owns mmap lifetime, shard routing, and
pickle/unpickle for Arrow IPC shard files. Returns Arrow-native types
(:class:`pa.Table`, :class:`pa.RecordBatch`) and plain Python dicts;
carries no dependency on PIL, torch, or numpy beyond what Arrow itself
requires. Decoding to user-facing types is the formatter's job.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from ._indexed_arrow_table import IndexedArrowTable


_MAX_OPEN_SHARDS = 512


class ArrowBackend:
    """Arrow-native storage layer.

    Construction modes:

    - **File-backed** (typical): receives shard file paths + row counts.
      Mmaps lazily on first access; each DataLoader worker re-mmaps after fork.
    - **In-memory**: receives a ``pa.Table`` directly (for derived subsets,
      column mutations, ``flatten_indices`` output).

    ``StableDataset`` should never inspect shard internals — it delegates
    all storage concerns here.
    """

    # DataLoader ``__getitems__`` uses batched gather for Arrow-backed data.
    # Binary-heavy random access is rebuilt from record-batch slices.
    prefer_batched_take: bool = True

    def __init__(
        self,
        *,
        shard_paths: list[Path] | None = None,
        table: pa.Table | None = None,
        shard_row_counts: list[int] | None = None,
        schema: pa.Schema | None = None,
    ):
        self._shard_paths = [Path(p) for p in shard_paths] if shard_paths is not None else None
        self._table: pa.Table | None = table
        self._shard_row_counts = list(shard_row_counts) if shard_row_counts is not None else None
        self._schema = schema

        # Shard-level lazy mmap and cumulative offsets for row routing
        if self._shard_paths is not None and self._shard_row_counts is not None:
            self._shard_tables: list[IndexedArrowTable | None] = [None] * len(self._shard_paths)
            self._cumulative = [0]
            for c in self._shard_row_counts:
                self._cumulative.append(self._cumulative[-1] + c)
            self._open_order: deque[int] = deque()
        else:
            self._shard_tables = []
            self._cumulative = None
            self._open_order = deque()
        self._indexed_table: IndexedArrowTable | None = IndexedArrowTable(table) if table is not None else None

    # -- Public query API (StableDataset calls these) -------------------------

    @property
    def num_rows(self) -> int:
        """Row count without forcing table load."""
        if self._shard_row_counts is not None:
            return sum(self._shard_row_counts)
        if self._table is not None:
            return self._table.num_rows
        return self.table.num_rows

    @property
    def is_file_backed(self) -> bool:
        return self._shard_paths is not None

    @property
    def cache_dir(self) -> Path | None:
        if self._shard_paths:
            return self._shard_paths[0].parent
        return None

    @property
    def schema(self) -> pa.Schema:
        if self._schema is not None:
            return self._schema
        if self._table is not None:
            return self._table.schema
        return self.table.schema

    @property
    def table(self) -> pa.Table:
        """Materialize and return the full Arrow table.

        For single-file datasets this is a cheap mmap. For multi-shard
        datasets this concatenates all shards into one table — use only
        when full materialization is intended (e.g. column mutations).
        Hot paths should use ``get_row``, ``take``,
        or ``iter_batches`` instead.
        """
        if self._table is None:
            if self._shard_paths is not None:
                if self._shard_paths:
                    if len(self._shard_paths) == 1:
                        self._table = self._mmap_ipc(self._shard_paths[0], upgrade_binary=True)
                    else:
                        tables = [self._mmap_ipc(p, upgrade_binary=True) for p in self._shard_paths]
                        self._table = pa.concat_tables(tables)
                else:
                    if self._schema is not None:
                        self._table = pa.table(
                            {f.name: pa.array([], type=f.type) for f in self._schema},
                            schema=self._schema,
                        )
                    else:
                        self._table = pa.table({})
            else:
                raise RuntimeError("No shard paths or in-memory table.")
        return self._table

    def get_row(self, idx: int) -> dict:
        """Single row access. Uses slice() which avoids copying binary data."""
        if self._shard_paths is not None and self._cumulative is not None:
            if len(self._shard_paths) == 1:
                tbl = self._get_shard_table(0).table
                row_slice = tbl.slice(idx, 1).to_pydict()
                return {k: v[0] for k, v in row_slice.items()}
            shard_id, local = self._locate_row(idx)
            tbl = self._get_shard_table(shard_id).table
            row_slice = tbl.slice(local, 1).to_pydict()
            return {k: v[0] for k, v in row_slice.items()}
        row_slice = self.table.slice(idx, 1).to_pydict()
        return {k: v[0] for k, v in row_slice.items()}

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Batched row access without using ``pa.Table.take``."""
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.size == 0:
            return self._empty_table()

        if self._shard_paths is not None and self._cumulative is not None:
            if len(self._shard_paths) == 1:
                tbl = self._get_shard_table(0)
                return self._take_from_indexed(tbl, index_array)
            # Group indices by shard: {shard_id: (output_positions, local_indices)}
            shard_groups: dict[int, tuple[list[int], list[int]]] = {}
            for out_pos, idx in enumerate(index_array.tolist()):
                shard_id, local = self._locate_row(int(idx))
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = ([], [])
                shard_groups[shard_id][0].append(out_pos)
                shard_groups[shard_id][1].append(local)

            # One batched gather per shard, then concat + reorder.
            parts = []
            reorder = []
            offset = 0
            for shard_id, (out_positions, local_indices) in shard_groups.items():
                tbl = self._get_shard_table(shard_id)
                part = self._take_from_indexed(tbl, np.asarray(local_indices, dtype=np.int64))
                parts.append(part)
                for i, out_pos in enumerate(out_positions):
                    reorder.append((out_pos, offset + i))
                offset += len(out_positions)

            combined = pa.concat_tables(parts)
            reorder.sort()
            final_order = [src for _, src in reorder]
            if final_order == list(range(len(final_order))):
                return combined
            return IndexedArrowTable(combined).fast_gather(final_order)
        return self._take_from_indexed(self._get_indexed_table(), index_array)

    def slice(self, start: int, length: int) -> pa.Table:
        """Contiguous range access.

        For single-shard datasets, slices directly from the mmap'd table
        without triggering full materialization.
        """
        if self._shard_paths is not None and self._cumulative is not None:
            if len(self._shard_paths) == 1:
                tbl = self._get_shard_table(0)
                return tbl.fast_slice(start, length)
        return self._get_indexed_table().fast_slice(start, length)

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield record batches from shard files sequentially.

        Only one shard is mmap'd at a time, which can reduce peak
        Python-managed memory for multi-shard datasets.
        """
        if self._shard_paths is None:
            yield from self.table.to_batches()
            return

        if shard_indices is None:
            shard_indices = list(range(len(self._shard_paths)))

        if shuffle and seed is not None:
            rng = np.random.default_rng(seed)
            shard_indices = list(shard_indices)
            rng.shuffle(shard_indices)

        for shard_id in shard_indices:
            shard_table = self._mmap_ipc(self._shard_paths[shard_id], upgrade_binary=False)
            yield from shard_table.to_batches()
            del shard_table

    @property
    def num_shards(self) -> int:
        if self._shard_paths is not None:
            return len(self._shard_paths)
        return 0

    # -- Pickle / DataLoader compatibility ------------------------------------

    def __getstate__(self):
        state = {
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "schema": self._schema,
        }
        if self._shard_paths is None:
            state["table"] = self._table
        return state

    def __setstate__(self, state):
        self.__init__(
            shard_paths=state.get("shard_paths"),
            table=state.get("table"),
            shard_row_counts=state.get("shard_row_counts"),
            schema=state.get("schema"),
        )

    # -- Internal helpers -----------------------------------------------------

    def _get_shard_table(self, shard_id: int) -> IndexedArrowTable:
        if self._shard_tables[shard_id] is None:
            while len(self._open_order) >= _MAX_OPEN_SHARDS:
                old = self._open_order.popleft()
                self._shard_tables[old] = None
            table = self._mmap_ipc(self._shard_paths[shard_id], upgrade_binary=False)
            self._shard_tables[shard_id] = IndexedArrowTable(table)
            self._open_order.append(shard_id)
        return self._shard_tables[shard_id]

    def _get_indexed_table(self) -> IndexedArrowTable:
        if self._indexed_table is None:
            self._indexed_table = IndexedArrowTable(self.table)
        return self._indexed_table

    def _take_from_indexed(self, indexed_table: IndexedArrowTable, indices: np.ndarray) -> pa.Table:
        contiguous = _as_contiguous_slice(indices)
        if contiguous is not None:
            start, length = contiguous
            return indexed_table.fast_slice(start, length)
        return indexed_table.fast_gather(indices)

    def _empty_table(self) -> pa.Table:
        return pa.Table.from_batches([], schema=self.schema)

    def _locate_row(self, idx: int) -> tuple[int, int]:
        shard_id = bisect.bisect_right(self._cumulative, idx) - 1
        local = idx - self._cumulative[shard_id]
        return shard_id, local

    @staticmethod
    def _mmap_ipc(path: Path, *, upgrade_binary: bool) -> pa.Table:
        mmap = pa.memory_map(str(path), "r")
        reader = ipc.open_file(mmap)
        table = reader.read_all()
        if upgrade_binary:
            return _upgrade_binary_columns(table)
        return table


def _as_contiguous_slice(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    if indices.size == 1:
        return int(indices[0]), 1
    if np.all(np.diff(indices) == 1):
        return int(indices[0]), int(indices.size)
    return None


def _upgrade_binary_columns(table: pa.Table) -> pa.Table:
    """Cast any ``binary`` columns to ``large_binary`` at open time.

    PyArrow's compute kernels (notably ``take``) fail with an i32 offset
    overflow on a ``binary`` column whose cumulative bytes exceed 2GB,
    regardless of how many rows are selected. Any ImageNet-scale image
    cache written before the ``Image`` feature was upgraded to
    ``large_binary`` is vulnerable. Casting at open time rebuilds the
    offset buffers (one-time ~10MB per million rows) while the values
    buffer is shared, so the runtime cost is negligible and existing
    on-disk caches need no rewrite.
    """
    needs_cast = [f for f in table.schema if pa.types.is_binary(f.type) and not pa.types.is_large_binary(f.type)]
    if not needs_cast:
        return table
    new_schema = pa.schema([pa.field(f.name, pa.large_binary()) if f in needs_cast else f for f in table.schema])
    return table.cast(new_schema)
