"""PyArrow-backed dataset with optional TensorDict conversion.

Provides ``StableDataset`` (single split) and ``StableDatasetDict`` (multi-split)
with ``__len__``, ``__getitem__``, ``.features``, and ``.train_test_split()``
for downstream benchmarks.

``StableDataset`` is a lightweight handle.  File-backed instances store just the
Arrow IPC path and lazily memory-map it on first access.  This keeps pickle size
tiny (only the path is serialised) so ``DataLoader`` workers share OS pages via
``mmap`` instead of copying the full table.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image as PILImage

from .schema import (
    Array3D,
    DatasetInfo,
    Features,
    Image,
    Sequence,
    Video,
)


class StableDataset:
    """A single-split dataset backed by a PyArrow Table.

    Two construction modes:

    1. **File-backed** (normal path) — ``StableDataset(path, features, info)``.
       The Arrow IPC file is memory-mapped lazily.  Pickling serialises only the
       path so ``DataLoader`` workers each open their own independent mmap,
       sharing physical pages via the OS page cache.

    2. **In-memory** (for slicing / ``train_test_split``) —
       ``StableDataset(path=None, features, info, table=table)``.
       The table lives in-process memory.  Pickle falls back to serialising the
       full table; this is fine for small derived subsets but should not be used
       for full datasets fed to multi-worker DataLoaders.
    """

    def __init__(
        self,
        path: Path | str | None,
        features: Features,
        info: DatasetInfo,
        *,
        table: pa.Table | None = None,
        num_rows: int | None = None,
    ):
        self._path = Path(path) if path is not None else None
        self._features = features
        self._info = info
        # _table is lazily loaded for file-backed datasets, or set directly for
        # in-memory datasets (slices, train_test_split results).
        self._table: pa.Table | None = table
        # Cache row count so __len__ never triggers a full file read.
        self._num_rows = num_rows if num_rows is not None else (table.num_rows if table is not None else None)

    # ── Lazy table access ────────────────────────────────────────────────────

    @property
    def table(self) -> pa.Table:
        """Return the underlying Arrow table, memory-mapping from disk if needed."""
        if self._table is None:
            if self._path is None:
                raise RuntimeError("StableDataset has neither a path nor an in-memory table.")
            mmap = pa.memory_map(str(self._path), "r")
            reader = ipc.open_file(mmap)
            self._table = reader.read_all()
            if self._num_rows is None:
                self._num_rows = self._table.num_rows
        return self._table

    # ── Pickle: serialise path (tiny) instead of table (huge) ────────────────

    def __getstate__(self):
        state = {
            "path": self._path,
            "features": self._features,
            "info": self._info,
            "num_rows": self._num_rows,
        }
        # Only include the table for in-memory datasets (no backing file).
        if self._path is None:
            state["table"] = self._table
        return state

    def __setstate__(self, state):
        self._path = state["path"]
        self._features = state["features"]
        self._info = state["info"]
        self._num_rows = state["num_rows"]
        # File-backed: _table stays None until first access triggers mmap.
        self._table = state.get("table")

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def features(self) -> Features:
        return self._features

    @property
    def info(self) -> DatasetInfo:
        return self._info

    def __len__(self) -> int:
        if self._num_rows is not None:
            return self._num_rows
        return self.table.num_rows

    def __getitem__(self, idx):
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            return self._decode_row(idx)
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            sub = self.table.take(list(indices))
            return StableDataset(path=None, features=self._features, info=self._info, table=sub)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, StableDataset]:
        """Random split. Returns ``{"train": StableDataset, "test": StableDataset}``."""
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)
        split_idx = int(n * (1 - test_size))
        train_indices = indices[:split_idx].tolist()
        test_indices = indices[split_idx:].tolist()
        tbl = self.table
        return {
            "train": StableDataset(path=None, features=self._features, info=self._info, table=tbl.take(train_indices)),
            "test": StableDataset(path=None, features=self._features, info=self._info, table=tbl.take(test_indices)),
        }

    def to_tensordict(self, columns: list[str] | None = None):
        """Convert numeric columns to a ``tensordict.TensorDict``.

        Image and Video columns are skipped (they stay lazy-decoded).
        Requires ``tensordict`` to be installed.
        """
        import torch
        from tensordict import TensorDict

        tbl = self.table
        td = {}
        for col_name, feat in self._features.items():
            if isinstance(feat, Image | Video | Array3D):
                continue
            if columns and col_name not in columns:
                continue
            col = tbl.column(col_name)
            if isinstance(feat, Sequence):
                td[col_name] = torch.tensor(col.to_pylist())
            else:
                td[col_name] = torch.from_numpy(col.to_numpy(zero_copy_only=False))
        return TensorDict(td, batch_size=[len(self)])

    def _decode_row(self, idx: int) -> dict:
        """Decode a single row, converting Arrow values back to Python objects."""
        tbl = self.table
        result = {}
        for col_name in tbl.column_names:
            feat = self._features.get(col_name)
            raw = tbl.column(col_name)[idx]

            if isinstance(feat, Image):
                img_bytes = raw.as_py()
                if img_bytes is not None:
                    result[col_name] = PILImage.open(io.BytesIO(img_bytes))
                else:
                    result[col_name] = None
            elif isinstance(feat, Array3D):
                arr_bytes = raw.as_py()
                if arr_bytes is not None:
                    result[col_name] = np.frombuffer(arr_bytes, dtype=feat.dtype).reshape(feat.shape)
                else:
                    result[col_name] = None
            else:
                result[col_name] = raw.as_py()
        return result


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``."""

    pass
