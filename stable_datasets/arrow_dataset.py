"""PyArrow-backed dataset with optional TensorDict conversion.

Provides ``StableDataset`` (single split) and ``StableDatasetDict`` (multi-split)
with ``__len__``, ``__getitem__``, ``__iter__``, ``.features``, and
``.train_test_split()`` for downstream benchmarks.

``StableDataset`` supports two construction modes:

1. **Shard-backed** — directory of Arrow IPC shards.  Only the needed shard
   is memory-mapped for ``__getitem__``; ``__iter__`` reads one shard at a
   time with bounded memory.
2. **In-memory** — for small derived subsets (slices, ``train_test_split``).

All modes keep pickle size tiny (paths only) so ``DataLoader`` workers share
OS pages via ``mmap`` instead of copying data.
"""

from __future__ import annotations

import io
from collections import OrderedDict
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


# Default maximum number of shard mmaps to keep open simultaneously.
_DEFAULT_MAX_OPEN_SHARDS = 4


class _ShardLRU:
    """Bounded LRU cache for memory-mapped shard tables.

    Evicts the least-recently-used shard when ``maxsize`` is exceeded so that
    pathological random-access patterns don't pin all shards in memory.
    """

    def __init__(self, maxsize: int = _DEFAULT_MAX_OPEN_SHARDS):
        self._maxsize = maxsize
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()

    def get(self, shard_id: int, shard_path: Path) -> pa.Table:
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        # Load and insert
        table = _mmap_ipc(shard_path)
        self._cache[shard_id] = table
        self._cache.move_to_end(shard_id)
        # Evict oldest if over capacity
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return table

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __contains__(self, shard_id: int):
        return shard_id in self._cache


class StableDataset:
    """A single-split dataset backed by a directory of Arrow IPC shards.

    Two construction modes:

    1. **Shard-backed** — ``StableDataset(features, info, shard_dir=...,
       shard_paths=[...], shard_row_counts=[...])``.
       Only the needed shard is memory-mapped; ``__iter__`` streams one shard
       at a time.

    2. **In-memory** — ``StableDataset(features, info, table=table)``.
       For small derived subsets (slices, splits).  Pickle serialises the full
       table.
    """

    def __init__(
        self,
        features: Features,
        info: DatasetInfo,
        *,
        table: pa.Table | None = None,
        num_rows: int | None = None,
        # Shard-backed construction
        shard_dir: Path | str | None = None,
        shard_paths: list[Path] | None = None,
        shard_row_counts: list[int] | None = None,
        max_open_shards: int = _DEFAULT_MAX_OPEN_SHARDS,
    ):
        self._features = features
        self._info = info
        self._table: pa.Table | None = table

        # Shard-backed state
        self._shard_dir = Path(shard_dir) if shard_dir is not None else None
        self._shard_paths = [Path(p) for p in shard_paths] if shard_paths is not None else None
        self._shard_row_counts = list(shard_row_counts) if shard_row_counts is not None else None
        self._shard_lru = _ShardLRU(maxsize=max_open_shards) if self._shard_paths is not None else None

        # Pre-compute cumulative row offsets for shard→global mapping
        self._shard_cumulative_offsets: list[int] | None = None
        if self._shard_row_counts is not None:
            cumulative = [0]
            for c in self._shard_row_counts:
                cumulative.append(cumulative[-1] + c)
            self._shard_cumulative_offsets = cumulative

        # Cache row count so __len__ never triggers a full file read.
        if num_rows is not None:
            self._num_rows = num_rows
        elif table is not None:
            self._num_rows = table.num_rows
        elif self._shard_row_counts is not None:
            self._num_rows = sum(self._shard_row_counts)
        else:
            self._num_rows = None

    @property
    def _is_shard_backed(self) -> bool:
        return self._shard_paths is not None

    # Lazy table access

    @property
    def table(self) -> pa.Table:
        """Return the underlying Arrow table, memory-mapping from disk if needed.

        For shard-backed datasets this concatenates all shards — prefer
        ``__getitem__`` or ``__iter__`` for large datasets.
        """
        if self._table is None:
            if self._is_shard_backed:
                if self._shard_paths:
                    tables = [_mmap_ipc(p) for p in self._shard_paths]
                    self._table = pa.concat_tables(tables)
                else:
                    # Zero-shard empty dataset — synthesise an empty table.
                    self._table = pa.table(
                        {name: pa.array([], type=feat.to_arrow_type()) for name, feat in self._features.items()},
                        schema=self._features.to_arrow_schema(),
                    )
            else:
                raise RuntimeError("StableDataset has no shard paths or in-memory table.")
            if self._num_rows is None:
                self._num_rows = self._table.num_rows
        return self._table

    def __getstate__(self):
        state = {
            "features": self._features,
            "info": self._info,
            "num_rows": self._num_rows,
            "shard_dir": self._shard_dir,
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "max_open_shards": self._shard_lru._maxsize if self._shard_lru is not None else _DEFAULT_MAX_OPEN_SHARDS,
        }
        # Only include the table for in-memory datasets (no shard backing).
        if self._shard_paths is None:
            state["table"] = self._table
        return state

    def __setstate__(self, state):
        self.__init__(
            features=state["features"],
            info=state["info"],
            num_rows=state["num_rows"],
            table=state.get("table"),
            shard_dir=state.get("shard_dir"),
            shard_paths=state.get("shard_paths"),
            shard_row_counts=state.get("shard_row_counts"),
            max_open_shards=state.get("max_open_shards", _DEFAULT_MAX_OPEN_SHARDS),
        )

    # Public API

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
        """Return a decoded row dict (int index) or a new in-memory dataset (slice).

        For shard-backed datasets, integer indexing maps the global row to a
        specific shard via cumulative offsets and memory-maps only that shard.
        A bounded LRU cache (default 4 shards) prevents repeated mmap/munmap
        churn under random-access workloads while capping resident memory.
        """
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            if self._is_shard_backed:
                return self._decode_row_sharded(idx)
            return self._decode_row(idx)
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            if self._is_shard_backed:
                return self._slice_sharded(indices)
            sub = self.table.take(indices)
            return StableDataset(features=self._features, info=self._info, table=sub)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __iter__(self):
        """Iterate over all rows, yielding decoded dicts.

        For shard-backed datasets, reads one shard at a time so peak memory
        is bounded to ~1 shard.  Each shard is memory-mapped independently
        and released after all its rows are yielded.  This deliberately
        bypasses the LRU cache used by ``__getitem__`` so that a full
        sequential scan does not evict shards that random access may need.
        """
        if self._is_shard_backed:
            yield from self._iter_shards(shuffle=False)
        else:
            for i in range(len(self)):
                yield self._decode_row(i)

    def iter_epoch(self, *, shuffle_shards: bool = True, seed: int | None = None):
        """Iterate over all rows with optional shard-level shuffling.

        For non-sharded datasets, this is equivalent to ``__iter__``.
        """
        if self._is_shard_backed:
            yield from self._iter_shards(shuffle=shuffle_shards, seed=seed)
        else:
            yield from self

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
            "train": StableDataset(features=self._features, info=self._info, table=tbl.take(train_indices)),
            "test": StableDataset(features=self._features, info=self._info, table=tbl.take(test_indices)),
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

    # Internal: in-memory row decoding

    def _decode_row(self, idx: int) -> dict:
        """Decode a single row from the full table."""
        return _decode_row_from_table(self.table, idx, self._features)

    def _decode_row_sharded(self, idx: int) -> dict:
        """Decode a single row (only maps the needed shard via LRU)."""
        shard_id, local_offset = self._locate_row(idx)
        shard_table = self._shard_lru.get(shard_id, self._shard_paths[shard_id])
        return _decode_row_from_table(shard_table, local_offset, self._features)

    def _locate_row(self, idx: int) -> tuple[int, int]:
        """Map a global row index to (shard_id, local_offset) using cumulative offsets."""
        lo, hi = 0, len(self._shard_row_counts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._shard_cumulative_offsets[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        shard_id = lo
        local_offset = idx - self._shard_cumulative_offsets[shard_id]
        return shard_id, local_offset

    def _slice_sharded(self, indices: list[int]) -> StableDataset:
        """Slice a shard-backed dataset by gathering rows from relevant shards."""
        from collections import defaultdict

        shard_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for out_pos, global_idx in enumerate(indices):
            shard_id, local_offset = self._locate_row(global_idx)
            shard_groups[shard_id].append((out_pos, local_offset))

        schema = self._features.to_arrow_schema()
        col_data: dict[str, list] = {name: [None] * len(indices) for name in self._features}

        for shard_id in sorted(shard_groups):
            shard_table = self._shard_lru.get(shard_id, self._shard_paths[shard_id])
            for out_pos, local_offset in shard_groups[shard_id]:
                for col_name in self._features:
                    col_data[col_name][out_pos] = shard_table.column(col_name)[local_offset].as_py()

        arrays = []
        for col_name in self._features:
            feat = self._features[col_name]
            arrays.append(pa.array(col_data[col_name], type=feat.to_arrow_type()))
        sub_table = pa.table(dict(zip(self._features.keys(), arrays)), schema=schema)
        return StableDataset(features=self._features, info=self._info, table=sub_table)

    def _iter_shards(self, *, shuffle: bool = False, seed: int | None = None):
        """Iterate over rows one shard at a time.

        Only one shard is referenced at a time; dropping the previous reference
        lets the OS reclaim pages.
        """
        shard_order = list(range(len(self._shard_paths)))
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(shard_order)

        for shard_id in shard_order:
            shard_table = _mmap_ipc(self._shard_paths[shard_id])
            for row_idx in range(shard_table.num_rows):
                yield _decode_row_from_table(shard_table, row_idx, self._features)
            del shard_table  # release mmap reference


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``."""

    pass


def _mmap_ipc(path: Path) -> pa.Table:
    """Memory-map an Arrow IPC file and return the table."""
    mmap = pa.memory_map(str(path), "r")
    reader = ipc.open_file(mmap)
    return reader.read_all()


def _decode_row_from_table(tbl: pa.Table, idx: int, features: Features) -> dict:
    """Decode a single row from an Arrow table into a Python dict."""
    result = {}
    for col_name in tbl.column_names:
        feat = features.get(col_name)
        raw = tbl.column(col_name)[idx]

        if isinstance(feat, Image):
            img_bytes = raw.as_py()
            if img_bytes is not None:
                img = PILImage.open(io.BytesIO(img_bytes))
                img.load()
                result[col_name] = img
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
