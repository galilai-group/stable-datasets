"""Arrow-native storage layer.

Owns mmap lifetime and table access. Knows nothing about PIL, torch, or numpy
beyond what Arrow itself uses. All public methods return Arrow-native types or
plain Python dicts (via ``to_pydict()``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


class ArrowBackend:
    """Arrow-native storage layer. Owns mmap lifetime and table access.

    Two construction modes:

    - **Shard-backed**: receives list of shard file paths. Does NOT mmap at
      init — each worker mmaps independently after fork (see lazy init).
    - **In-memory**: receives a ``pa.Table`` directly (for small derived
      subsets, ``flatten_indices`` output).
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

    @property
    def table(self) -> pa.Table:
        """Mmap all shards and concat on first access. Lazy."""
        if self._table is None:
            if self._shard_paths is not None:
                if self._shard_paths:
                    tables = [self._mmap_ipc(p) for p in self._shard_paths]
                    self._table = pa.concat_tables(tables)
                else:
                    # Zero-shard empty dataset — synthesise an empty table
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
        """Single row access. Returns {col_name: python_value}.

        Uses ``table.slice(idx, 1).to_pydict()`` — one C++ call for all
        columns.
        """
        row_slice = self.table.slice(idx, 1).to_pydict()
        return {k: v[0] for k, v in row_slice.items()}

    def take(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Batched row access. Returns a ``pa.Table`` with ``len(indices)`` rows.

        One C++ call regardless of how many indices.
        """
        return self.table.take(indices)

    def slice(self, start: int, length: int) -> pa.Table:
        """Contiguous range access. Returns a ``pa.Table``.

        Even cheaper than ``take()`` for sequential access patterns.
        """
        return self.table.slice(start, length)

    @property
    def num_rows(self) -> int:
        """Row count without forcing table load if shard_row_counts available."""
        if self._shard_row_counts is not None:
            return sum(self._shard_row_counts)
        if self._table is not None:
            return self._table.num_rows
        # Must load table to get count
        return self.table.num_rows

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Yield record batches from individual shard files.

        Only one shard is mmap'd at a time — bounded memory.
        ``shard_indices`` controls which shards to read (for worker
        partitioning). Does NOT use ``self._table`` — reads directly from
        shard files.
        """
        if self._shard_paths is None:
            # In-memory: yield the whole table as a single batch
            for batch in self.table.to_batches():
                yield batch
            return

        if shard_indices is None:
            shard_indices = list(range(len(self._shard_paths)))

        if shuffle and seed is not None:
            rng = np.random.default_rng(seed)
            shard_indices = list(shard_indices)
            rng.shuffle(shard_indices)

        for shard_id in shard_indices:
            shard_table = self._mmap_ipc(self._shard_paths[shard_id])
            for batch in shard_table.to_batches():
                yield batch
            del shard_table

    def __getstate__(self):
        state = {
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "schema": self._schema,
        }
        # Only serialize table for in-memory datasets
        if self._shard_paths is None:
            state["table"] = self._table
        return state

    def __setstate__(self, state):
        self._shard_paths = state.get("shard_paths")
        self._shard_row_counts = state.get("shard_row_counts")
        self._table = state.get("table")
        self._schema = state.get("schema")

    @staticmethod
    def _mmap_ipc(path: Path) -> pa.Table:
        mmap = pa.memory_map(str(path), "r")
        reader = ipc.open_file(mmap)
        return reader.read_all()
