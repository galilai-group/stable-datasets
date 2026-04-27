"""Indexed Arrow table helpers for fast random batch access.

The core gather path here is modeled on the strategy used by
``datasets``/Hugging Face: pre-index record-batch boundaries once, then
reconstruct arbitrary row batches from inexpensive ``RecordBatch.slice``
calls instead of relying on ``pyarrow.Table.take`` for non-contiguous
binary-heavy access.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


class IndexedArrowTable:
    """Arrow table wrapper with cheap contiguous slices and row gathers."""

    def __init__(self, table: pa.Table):
        self.table = table
        self.schema = table.schema
        self._batches: list[pa.RecordBatch] = [batch for batch in table.to_batches() if len(batch) > 0]
        self._offsets = np.cumsum([0] + [len(batch) for batch in self._batches], dtype=np.int64)

    @property
    def num_rows(self) -> int:
        return int(self._offsets[-1]) if len(self._offsets) else 0

    def empty_table(self) -> pa.Table:
        return pa.Table.from_batches([], schema=self.schema)

    def fast_slice(self, offset: int = 0, length: int | None = None) -> pa.Table:
        """Return a contiguous slice without materializing unrelated rows."""
        if offset < 0:
            raise IndexError("Offset must be non-negative")

        total_rows = self.num_rows
        if offset >= total_rows or (length is not None and length <= 0):
            return self.empty_table()

        if length is None or offset + length >= total_rows:
            end = total_rows
        else:
            end = offset + length

        start_batch = int(np.searchsorted(self._offsets, offset, side="right") - 1)
        end_batch = int(np.searchsorted(self._offsets, end - 1, side="right") - 1)

        if start_batch == end_batch:
            batch = self._batches[start_batch]
            local_offset = offset - int(self._offsets[start_batch])
            return pa.Table.from_batches([batch.slice(local_offset, end - offset)], schema=self.schema)

        batches = list(self._batches[start_batch : end_batch + 1])
        start_local = offset - int(self._offsets[start_batch])
        end_local = end - int(self._offsets[end_batch])
        batches[0] = batches[0].slice(start_local)
        batches[-1] = batches[-1].slice(0, end_local)
        return pa.Table.from_batches(batches, schema=self.schema)

    def fast_gather(self, indices: np.ndarray | list[int]) -> pa.Table:
        """Gather arbitrary rows while preserving order and duplicates."""
        if isinstance(indices, np.ndarray):
            if indices.size == 0:
                return self.empty_table()
            index_array = indices.astype(np.int64, copy=False)
        else:
            if len(indices) == 0:
                return self.empty_table()
            index_array = np.asarray(indices, dtype=np.int64)

        batch_indices = np.searchsorted(self._offsets, index_array, side="right") - 1
        gathered_batches = [
            self._batches[int(batch_idx)].slice(int(idx - self._offsets[int(batch_idx)]), 1)
            for batch_idx, idx in zip(batch_indices, index_array, strict=False)
        ]
        return pa.Table.from_batches(gathered_batches, schema=self.schema)
