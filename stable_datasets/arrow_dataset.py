"""PyArrow-backed dataset with optional TensorDict conversion.

Provides ``StableDataset`` (single split) and ``StableDatasetDict`` (multi-split)
with ``__len__``, ``__getitem__``, ``__iter__``, ``.features``, and
``.train_test_split()`` for downstream benchmarks.

``StableDataset`` supports three construction modes:

1. **Shard-backed** — directory of Arrow IPC shards.  Only the needed shard
   is memory-mapped for ``__getitem__``; ``__iter__`` reads one shard at a
   time with bounded memory.
2. **In-memory** — for small derived subsets (slices, ``train_test_split``).
3. **Indexed view** — a virtual view via ``_indices`` sharing the same
   underlying data.  All derived datasets (shuffled, filtered, split) create
   an indices array sharing the same on-disk shards.  Zero data copying.

All modes keep pickle size tiny (paths only) so ``DataLoader`` workers share
OS pages via ``mmap`` instead of copying data.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Callable
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

    Three construction modes:

    1. **Shard-backed** — ``StableDataset(features, info, shard_dir=...,
       shard_paths=[...], shard_row_counts=[...])``.
       Only the needed shard is memory-mapped; ``__iter__`` streams one shard
       at a time.

    2. **In-memory** — ``StableDataset(features, info, table=table)``.
       For small derived subsets (slices, splits).  Pickle serialises the full
       table.

    3. **Indexed view** — ``StableDataset(..., _indices=array)``.
       A virtual view into a shard-backed or in-memory dataset.  All derived
       datasets (shuffled, filtered, split) create an indices array sharing
       the same underlying data.  Zero data copying.
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
        # Index indirection
        _indices: np.ndarray | None = None,
        # Format control
        _format_type: str | None = None,
        _transform: Callable | None = None,
    ):
        self._features = features
        self._info = info
        self._table: pa.Table | None = table

        # Shard-backed state
        self._shard_dir = Path(shard_dir) if shard_dir is not None else None
        self._shard_paths = [Path(p) for p in shard_paths] if shard_paths is not None else None
        self._shard_row_counts = list(shard_row_counts) if shard_row_counts is not None else None
        self._shard_lru = _ShardLRU(maxsize=max_open_shards) if self._shard_paths is not None else None

        # Pre-compute cumulative row offsets for shard->global mapping
        self._shard_cumulative_offsets: list[int] | None = None
        if self._shard_row_counts is not None:
            cumulative = [0]
            for c in self._shard_row_counts:
                cumulative.append(cumulative[-1] + c)
            self._shard_cumulative_offsets = cumulative

        # Index indirection
        self._indices = np.asarray(_indices, dtype=np.uint64) if _indices is not None else None

        # Format and transform
        self._format_type = _format_type
        self._transform = _transform

        # Cache row count so __len__ never triggers a full file read.
        # _indices takes priority since it defines the virtual size.
        if self._indices is not None:
            self._num_rows = len(self._indices)
        elif num_rows is not None:
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
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_transform": self._transform,
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
            _indices=state.get("_indices"),
            _format_type=state.get("_format_type"),
            _transform=state.get("_transform"),
        )

    # Public API

    @property
    def features(self) -> Features:
        return self._features

    @property
    def info(self) -> DatasetInfo:
        return self._info

    @property
    def column_names(self) -> list[str]:
        """Return the list of column names."""
        return list(self._features.keys())

    @property
    def num_rows(self) -> int:
        """Return the number of rows."""
        return len(self)

    def __len__(self) -> int:
        if self._num_rows is not None:
            return self._num_rows
        return self.table.num_rows

    def __getitem__(self, idx):
        """Return a decoded row dict (int index) or a new indexed view (slice).

        For shard-backed datasets, integer indexing maps the global row to a
        specific shard via cumulative offsets and memory-maps only that shard.
        A bounded LRU cache (default 4 shards) prevents repeated mmap/munmap
        churn under random-access workloads while capping resident memory.

        When ``_indices`` is set, virtual indices are resolved to physical
        indices before lookup — no data copying.
        """
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            # Resolve through indices
            if self._indices is not None:
                idx = int(self._indices[idx])
            if self._is_shard_backed:
                row = self._decode_row_sharded(idx)
            else:
                row = self._decode_row(idx)
            return self._apply_formatting(row)
        if isinstance(idx, slice):
            # Shard-backed or indexed: create a zero-copy indexed view
            if self._is_shard_backed or self._indices is not None:
                physical = np.arange(*idx.indices(len(self)), dtype=np.uint64)
                if self._indices is not None:
                    physical = self._indices[physical]
                return self._view_with_indices(physical)
            # In-memory without indices: materialize the slice
            sub = self.table.take(list(range(*idx.indices(len(self)))))
            return StableDataset(
                features=self._features,
                info=self._info,
                table=sub,
                _format_type=self._format_type,
                _transform=self._transform,
            )
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __iter__(self):
        """Iterate over all rows, yielding decoded dicts.

        For shard-backed datasets without indices, reads one shard at a time
        so peak memory is bounded to ~1 shard.  For indexed datasets,
        iterates in virtual order via ``__getitem__``.
        """
        if self._is_shard_backed and self._indices is None:
            yield from self._iter_shards(shuffle=False)
        else:
            for i in range(len(self)):
                yield self[i]

    def iter_epoch(self, *, shuffle_shards: bool = True, seed: int | None = None):
        """Iterate over all rows with optional shard-level shuffling.

        For indexed shard-backed datasets, groups rows by shard for I/O
        locality.  For non-sharded datasets, this is equivalent to
        ``__iter__``.
        """
        if self._indices is not None and self._is_shard_backed:
            yield from self._iter_indexed(shuffle=shuffle_shards, seed=seed)
        elif self._is_shard_backed:
            yield from self._iter_shards(shuffle=shuffle_shards, seed=seed)
        else:
            yield from self

    # Selection / shuffling / filtering

    def select(self, indices) -> StableDataset:
        """Return a view containing only the specified row indices.

        For shard-backed datasets, creates a zero-copy indexed view.
        Composes with existing ``_indices`` if present.
        """
        indices = np.asarray(indices, dtype=np.uint64)
        if self._indices is not None:
            indices = self._indices[indices]
        if self._is_shard_backed:
            return self._view_with_indices(indices)
        sub = self.table.take(indices.tolist())
        return StableDataset(
            features=self._features,
            info=self._info,
            table=sub,
            _format_type=self._format_type,
            _transform=self._transform,
        )

    def shuffle(self, seed: int = 42) -> StableDataset:
        """Return a shuffled view of this dataset.

        The returned dataset shares the same underlying shards; only a
        permuted index array is created.  *seed* controls the random
        permutation for reproducibility.
        """
        perm = np.random.default_rng(seed).permutation(len(self))
        return self.select(perm)

    def filter(self, fn: Callable[[dict], bool]) -> StableDataset:
        """Return a view containing rows where ``fn(row)`` is True.

        *fn* receives a decoded row dict and returns True for rows to
        keep.  Every row is decoded once during filtering.
        """
        matching = [i for i in range(len(self)) if fn(self[i])]
        return self.select(matching)

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, StableDataset]:
        """Random split via index indirection.  No data materialization.

        Returns ``{"train": StableDataset, "test": StableDataset}``.
        """
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self))
        split_idx = int(len(self) * (1 - test_size))
        return {
            "train": self.select(perm[:split_idx]),
            "test": self.select(perm[split_idx:]),
        }

    # Format and transform pipeline

    def with_format(self, format_type: str | None) -> StableDataset:
        """Return a view with the specified output format.

        Supported values:
        - ``None`` (default): PIL Image, numpy Array3D, Python scalars
        - ``"torch"``: Image -> CHW float tensor, scalars -> tensors
        - ``"numpy"``: Image -> HWC numpy array
        - ``"raw"``: Image -> bytes, Array3D -> bytes (skip PIL decode)
        """
        return self._shallow_copy(_format_type=format_type)

    def with_transform(self, fn: Callable | None) -> StableDataset:
        """Return a view with a transform applied in ``__getitem__``
        after format conversion."""
        return self._shallow_copy(_transform=fn)

    def as_iterable(
        self,
        *,
        shuffle: bool = False,
        seed: int = 0,
        buffer_size: int = 10_000,
        transform: Callable | None = None,
    ):
        """Return a ``StableIterableDataset`` wrapping this dataset."""
        from .iterable import StableIterableDataset

        return StableIterableDataset(
            self,
            shuffle=shuffle,
            seed=seed,
            buffer_size=buffer_size,
            transform=transform,
        )

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

    # Internal: helpers

    def _view_with_indices(self, indices: np.ndarray) -> StableDataset:
        """Return a new StableDataset sharing the same backing data
        with the given physical indices.  Zero-copy view."""
        return StableDataset(
            features=self._features,
            info=self._info,
            table=self._table,
            shard_dir=self._shard_dir,
            shard_paths=self._shard_paths,
            shard_row_counts=self._shard_row_counts,
            max_open_shards=self._shard_lru._maxsize if self._shard_lru else _DEFAULT_MAX_OPEN_SHARDS,
            _indices=np.asarray(indices, dtype=np.uint64),
            _format_type=self._format_type,
            _transform=self._transform,
        )

    def _shallow_copy(self, **overrides) -> StableDataset:
        """Return a shallow copy with optional attribute overrides."""
        kw = {
            "features": self._features,
            "info": self._info,
            "table": self._table,
            "shard_dir": self._shard_dir,
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "max_open_shards": self._shard_lru._maxsize if self._shard_lru else _DEFAULT_MAX_OPEN_SHARDS,
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_transform": self._transform,
        }
        kw.update(overrides)
        return StableDataset(**kw)

    def _apply_formatting(self, row: dict) -> dict:
        """Apply format conversion and transform to a decoded row."""
        if self._format_type in ("torch", "numpy"):
            row = _apply_format(row, self._features, self._format_type)
        if self._transform is not None:
            row = self._transform(row)
        return row

    # Internal: in-memory row decoding

    def _decode_row(self, idx: int) -> dict:
        """Decode a single row from the full table."""
        return _decode_row_from_table(self.table, idx, self._features, self._format_type)

    def _decode_row_sharded(self, idx: int) -> dict:
        """Decode a single row (only maps the needed shard via LRU)."""
        shard_id, local_offset = self._locate_row(idx)
        shard_table = self._shard_lru.get(shard_id, self._shard_paths[shard_id])
        return _decode_row_from_table(shard_table, local_offset, self._features, self._format_type)

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
                row = _decode_row_from_table(shard_table, row_idx, self._features, self._format_type)
                yield self._apply_formatting(row)
            del shard_table  # release mmap reference

    def _iter_indexed(self, *, shuffle: bool = False, seed: int | None = None):
        """Iterate indexed rows grouped by shard for I/O locality.

        Groups entries of ``_indices`` by target shard, optionally permutes
        shard order, then for each shard mmaps it, yields its rows in
        ``_indices`` order, and releases the mmap.
        """
        from collections import defaultdict

        shard_groups: dict[int, list[int]] = defaultdict(list)
        for physical_idx in self._indices:
            shard_id, local_offset = self._locate_row(int(physical_idx))
            shard_groups[shard_id].append(local_offset)

        shard_order = list(shard_groups.keys())
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(shard_order)

        for shard_id in shard_order:
            shard_table = _mmap_ipc(self._shard_paths[shard_id])
            for local_offset in shard_groups[shard_id]:
                row = _decode_row_from_table(shard_table, local_offset, self._features, self._format_type)
                yield self._apply_formatting(row)
            del shard_table


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``."""

    pass


def _mmap_ipc(path: Path) -> pa.Table:
    """Memory-map an Arrow IPC file and return the table."""
    mmap = pa.memory_map(str(path), "r")
    reader = ipc.open_file(mmap)
    return reader.read_all()


def _decode_row_from_table(
    tbl: pa.Table,
    idx: int,
    features: Features,
    format_type: str | None = None,
) -> dict:
    """Decode a single row from an Arrow table into a Python dict.

    When ``format_type`` is ``"raw"``, Image and Array3D columns return
    raw bytes instead of decoded PIL Images / numpy arrays.
    """
    result = {}
    for col_name in tbl.column_names:
        feat = features.get(col_name)
        raw = tbl.column(col_name)[idx]

        if isinstance(feat, Image):
            img_bytes = raw.as_py()
            if img_bytes is None:
                result[col_name] = None
            elif format_type == "raw":
                result[col_name] = img_bytes
            else:
                img = PILImage.open(io.BytesIO(img_bytes))
                img.load()
                result[col_name] = img
        elif isinstance(feat, Array3D):
            arr_bytes = raw.as_py()
            if arr_bytes is None:
                result[col_name] = None
            elif format_type == "raw":
                result[col_name] = arr_bytes
            else:
                result[col_name] = np.frombuffer(arr_bytes, dtype=feat.dtype).reshape(feat.shape)
        else:
            result[col_name] = raw.as_py()
    return result


def _apply_format(row: dict, features: Features, format_type: str) -> dict:
    """Apply format conversion to a decoded row (post-processing step)."""
    if format_type == "torch":
        import torch

        result = {}
        for col_name, value in row.items():
            feat = features.get(col_name)
            if value is None:
                result[col_name] = None
            elif isinstance(feat, Image):
                arr = np.array(value)
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
                result[col_name] = torch.from_numpy(arr.astype(np.float32) / 255.0)
            elif isinstance(feat, Array3D):
                result[col_name] = torch.from_numpy(np.array(value, dtype=np.float32))
            elif isinstance(value, (int, float)):
                result[col_name] = torch.tensor(value)
            else:
                result[col_name] = value
        return result
    if format_type == "numpy":
        result = {}
        for col_name, value in row.items():
            feat = features.get(col_name)
            if isinstance(feat, Image) and value is not None:
                result[col_name] = np.array(value)
            else:
                result[col_name] = value
        return result
    return row
