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
            self._shard_tables: list[pa.Table | None] = [None] * len(self._shard_paths)
            self._cumulative = [0]
            for c in self._shard_row_counts:
                self._cumulative.append(self._cumulative[-1] + c)
            self._open_order: deque[int] = deque()
        else:
            self._shard_tables = []
            self._cumulative = None
            self._open_order = deque()

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
        when full materialization is intended (e.g. ``to_tensordict``,
        column mutations). Hot paths should use ``get_row``, ``take``,
        or ``iter_batches`` instead.
        """
        if self._table is None:
            if self._shard_paths is not None:
                if self._shard_paths:
                    if len(self._shard_paths) == 1:
                        self._table = self._mmap_ipc(self._shard_paths[0])
                    else:
                        tables = [self._mmap_ipc(p) for p in self._shard_paths]
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
                tbl = self._get_shard_table(0)
                row_slice = tbl.slice(idx, 1).to_pydict()
                return {k: v[0] for k, v in row_slice.items()}
            shard_id, local = self._locate_row(idx)
            tbl = self._get_shard_table(shard_id)
            row_slice = tbl.slice(local, 1).to_pydict()
            return {k: v[0] for k, v in row_slice.items()}
        row_slice = self.table.slice(idx, 1).to_pydict()
        return {k: v[0] for k, v in row_slice.items()}

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Batched row access.

        For multi-shard datasets, groups indices by shard and does one
        ``tbl.take()`` per shard (not per row), then reorders to match
        the requested index order.
        """
        if self._shard_paths is not None and self._cumulative is not None:
            if len(self._shard_paths) == 1:
                tbl = self._get_shard_table(0)
                return tbl.take(indices)
            # Group indices by shard: {shard_id: (output_positions, local_indices)}
            shard_groups: dict[int, tuple[list[int], list[int]]] = {}
            for out_pos, idx in enumerate(indices):
                shard_id, local = self._locate_row(int(idx))
                if shard_id not in shard_groups:
                    shard_groups[shard_id] = ([], [])
                shard_groups[shard_id][0].append(out_pos)
                shard_groups[shard_id][1].append(local)

            # One batched take() per shard, then concat + reorder
            parts = []
            reorder = []
            offset = 0
            for shard_id, (out_positions, local_indices) in shard_groups.items():
                tbl = self._get_shard_table(shard_id)
                part = tbl.take(local_indices)
                parts.append(part)
                for i, out_pos in enumerate(out_positions):
                    reorder.append((out_pos, offset + i))
                offset += len(out_positions)

            combined = pa.concat_tables(parts)
            # Reorder to match the originally requested index order
            reorder.sort()  # sort by out_pos
            final_order = [src for _, src in reorder]
            return combined.take(final_order)
        return self.table.take(indices)

    def slice(self, start: int, length: int) -> pa.Table:
        """Contiguous range access.

        For single-shard datasets, slices directly from the mmap'd table
        without triggering full materialization.
        """
        if self._shard_paths is not None and self._cumulative is not None:
            if len(self._shard_paths) == 1:
                tbl = self._get_shard_table(0)
                return tbl.slice(start, length)
        return self.table.slice(start, length)

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
            shard_table = self._mmap_ipc(self._shard_paths[shard_id])
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

    def _get_shard_table(self, shard_id: int) -> pa.Table:
        if self._shard_tables[shard_id] is None:
            while len(self._open_order) >= _MAX_OPEN_SHARDS:
                old = self._open_order.popleft()
                self._shard_tables[old] = None
            self._shard_tables[shard_id] = self._mmap_ipc(self._shard_paths[shard_id])
            self._open_order.append(shard_id)
        return self._shard_tables[shard_id]

    def _locate_row(self, idx: int) -> tuple[int, int]:
        shard_id = bisect.bisect_right(self._cumulative, idx) - 1
        local = idx - self._cumulative[shard_id]
        return shard_id, local

    @staticmethod
    def _mmap_ipc(path: Path) -> pa.Table:
        mmap = pa.memory_map(str(path), "r")
        reader = ipc.open_file(mmap)
        return reader.read_all()
