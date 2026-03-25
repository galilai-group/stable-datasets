"""PyArrow-backed dataset with optional TensorDict conversion.

Provides ``StableDataset`` (single split) and ``StableDatasetDict`` (multi-split)
with ``__len__``, ``__getitem__``, ``__getitems__``, ``__iter__``,
``.features``, and ``.train_test_split()`` for downstream benchmarks.

Architecture: three layers with strict boundaries::

    ArrowBackend   -> only touches Arrow, returns Arrow-native or minimal Python dicts
        |
    Formatter      -> converts Arrow output to user-requested format (PIL/torch/numpy/raw)
        |
    StableDataset  -> orchestrates backend + formatter + indices + transform
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .backend import ArrowBackend
from .cache import _CACHE_FORMAT_VERSION, _features_fingerprint
from .formatting import get_formatter
from .schema import (
    Array3D,
    DatasetInfo,
    Features,
    Image,
    Sequence,
    Video,
)


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
        backend: ArrowBackend | None = None,
        table: pa.Table | None = None,
        num_rows: int | None = None,
        # Shard-backed construction
        shard_dir: Path | str | None = None,
        shard_paths: list[Path] | None = None,
        shard_row_counts: list[int] | None = None,
        # Index indirection
        _indices: np.ndarray | None = None,
        # Format control
        _format_type: str | None = None,
        _decode_images: bool = True,
        _transform: Callable | None = None,
        # Legacy compat (ignored)
        max_open_shards: int = 4,
    ):
        self._features = features
        self._info = info

        # Shard metadata (kept for streaming path / pickle)
        self._shard_dir = Path(shard_dir) if shard_dir is not None else None
        self._shard_paths = [Path(p) for p in shard_paths] if shard_paths is not None else None
        self._shard_row_counts = list(shard_row_counts) if shard_row_counts is not None else None

        # Build or accept backend
        arrow_schema = features.to_arrow_schema()
        if backend is not None:
            self._backend = backend
        elif shard_paths is not None:
            self._backend = ArrowBackend(
                shard_paths=shard_paths,
                shard_row_counts=shard_row_counts,
                schema=arrow_schema,
            )
        elif table is not None:
            self._backend = ArrowBackend(table=table, schema=arrow_schema)
        else:
            self._backend = ArrowBackend(shard_paths=[], shard_row_counts=[], schema=arrow_schema)

        # Index indirection
        self._indices = np.asarray(_indices, dtype=np.int64) if _indices is not None else None

        # Format and transform
        self._format_type = _format_type
        self._decode_images = _decode_images
        self._transform = _transform
        self._formatter = get_formatter(_format_type, features, decode_images=_decode_images)

        # Precompute whether we have binary columns (Image/Array3D/Video).
        # When present, __getitems__ avoids batched take() because Arrow
        # physically copies variable-length binary data into new buffers,
        # whereas per-row slice() is zero-copy.
        self._has_binary_cols = any(
            isinstance(f, (Image, Array3D, Video)) for f in features.values()
        )

        # Cache row count so __len__ never triggers a full file read.
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

    # -- Compatibility shims --------------------------------------------------

    @property
    def _table(self):
        """Expose backend's table reference for test compatibility."""
        return self._backend._table

    @property
    def _is_shard_backed(self) -> bool:
        return self._shard_paths is not None

    # -- Lazy table access ----------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """Return the underlying Arrow table, memory-mapping from disk if needed.

        For shard-backed datasets this concatenates all shards — prefer
        ``__getitem__`` or ``__iter__`` for large datasets.
        """
        tbl = self._backend.table
        if self._num_rows is None:
            self._num_rows = tbl.num_rows
        return tbl

    # -- Pickle / DataLoader compatibility ------------------------------------

    def __getstate__(self):
        state = {
            "features": self._features,
            "info": self._info,
            "num_rows": self._num_rows,
            "shard_dir": self._shard_dir,
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_decode_images": self._decode_images,
            "_transform": self._transform,
        }
        # Only include the table for in-memory datasets (no shard backing).
        if self._shard_paths is None:
            state["table"] = self._backend._table
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
            _indices=state.get("_indices"),
            _format_type=state.get("_format_type"),
            _decode_images=state.get("_decode_images", True),
            _transform=state.get("_transform"),
        )

    # -- Public API -----------------------------------------------------------

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
        return self._backend.num_rows

    def __getitem__(self, idx):
        """Return a decoded row dict (int index) or a new indexed view (slice)."""
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            physical = int(self._indices[idx]) if self._indices is not None else idx
            row = self._backend.get_row(physical)
            row = self._formatter.format_row(row)
            if self._transform is not None:
                row = self._transform(row)
            return row

        if isinstance(idx, slice):
            indices = np.arange(*idx.indices(len(self)), dtype=np.int64)
            if self._indices is not None:
                indices = self._indices[indices]
            if self._is_shard_backed or self._indices is not None:
                return self._view_with_indices(indices)
            # In-memory without indices: materialize the slice
            sub = self._backend.take(indices.tolist())
            return StableDataset(
                features=self._features,
                info=self._info,
                table=sub,
                _format_type=self._format_type,
                _decode_images=self._decode_images,
                _transform=self._transform,
            )
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched sample loading. Called by PyTorch DataLoader automatically.

        For datasets with binary columns (Image, Array3D, Video), uses
        per-row ``slice()`` which is zero-copy from mmap.  For purely
        numeric/scalar datasets, uses batched ``take()`` to reduce Python
        call overhead.
        """
        if self._has_binary_cols:
            return [self[i] for i in indices]

        idx_array = np.asarray(indices, dtype=np.int64)
        if self._indices is not None:
            idx_array = self._indices[idx_array]

        batch_table = self._backend.take(idx_array)
        rows = self._formatter.format_batch(batch_table)

        if self._transform is not None:
            rows = [self._transform(row) for row in rows]

        return rows

    def __iter__(self):
        """Iterate over all rows, yielding decoded dicts.

        For shard-backed datasets without indices, reads one shard at a time
        so peak memory is bounded to ~1 shard.
        """
        if self._is_shard_backed and self._indices is None:
            for batch in self._backend.iter_batches():
                batch_dict = batch.to_pydict()
                n = batch.num_rows
                for i in range(n):
                    row = {k: v[i] for k, v in batch_dict.items()}
                    row = self._formatter.format_row(row)
                    if self._transform is not None:
                        row = self._transform(row)
                    yield row
        else:
            for i in range(len(self)):
                yield self[i]

    def iter_epoch(self, *, shuffle_shards: bool = True, seed: int | None = None):
        """Iterate over all rows with optional shard-level shuffling.

        For indexed shard-backed datasets, groups rows by shard for I/O
        locality. For non-sharded datasets, this is equivalent to
        ``__iter__``.
        """
        if self._indices is not None:
            # Indexed: iterate in virtual order via __getitem__
            for i in range(len(self)):
                yield self[i]
        elif self._is_shard_backed:
            for batch in self._backend.iter_batches(shuffle=shuffle_shards, seed=seed):
                batch_dict = batch.to_pydict()
                n = batch.num_rows
                for i in range(n):
                    row = {k: v[i] for k, v in batch_dict.items()}
                    row = self._formatter.format_row(row)
                    if self._transform is not None:
                        row = self._transform(row)
                    yield row
        else:
            yield from self

    # -- Selection / shuffling / filtering ------------------------------------

    def select(self, indices) -> StableDataset:
        """Return a view containing only the specified row indices.

        For shard-backed datasets, creates a zero-copy indexed view.
        Composes with existing ``_indices`` if present.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if self._indices is not None:
            indices = self._indices[indices]
        return self._view_with_indices(indices)

    def shuffle(self, seed: int = 42) -> StableDataset:
        """Return a shuffled view of this dataset."""
        perm = np.random.default_rng(seed).permutation(len(self))
        return self.select(perm)

    def filter(self, fn: Callable[[dict], bool]) -> StableDataset:
        """Return a view containing rows where ``fn(row)`` is True."""
        matching = [i for i in range(len(self)) if fn(self[i])]
        return self.select(matching)

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, StableDataset]:
        """Random split via index indirection. No data materialization."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(self))
        split_idx = int(len(self) * (1 - test_size))
        return {
            "train": self.select(perm[:split_idx]),
            "test": self.select(perm[split_idx:]),
        }

    # -- Format and transform pipeline ----------------------------------------

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
        """Return a view with a transform applied after format conversion."""
        return self._shallow_copy(_transform=fn)

    def set_decode(self, decode: bool) -> StableDataset:
        """Control whether Image columns are decoded or left as raw bytes.

        When ``decode=False``, Image columns return raw bytes regardless of
        ``format_type``. Useful for custom decode pipelines (torchvision, DALI).
        """
        return self._shallow_copy(_decode_images=decode)

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
        """Convert numeric columns to a ``tensordict.TensorDict``."""
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

    def flatten_indices(self, cache_dir: Path | None = None) -> StableDataset:
        """Materialize an indexed view into a new contiguous Arrow file.

        Writes the rows selected by ``_indices`` into a new Arrow IPC file in
        physical order. Returns a new ``StableDataset`` backed by that file
        with no indices mapping.

        If ``_indices`` is None (already contiguous), returns self.
        """
        if self._indices is None:
            return self

        if cache_dir is None:
            parent = self._shard_dir.parent if self._shard_dir else Path(tempfile.gettempdir())
            cache_dir = Path(tempfile.mkdtemp(dir=parent, prefix=".flatten_"))
        else:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        materialized = self._backend.take(self._indices)

        out_path = cache_dir / "shard-00000.arrow"
        schema = self._features.to_arrow_schema()
        with pa.OSFile(str(out_path), "wb") as sink:
            writer = ipc.new_file(sink, schema)
            batch_size = 10_000
            for start in range(0, materialized.num_rows, batch_size):
                end = min(start + batch_size, materialized.num_rows)
                writer.write_table(materialized.slice(start, end - start))
            writer.close()

        import json

        meta = {
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "schema_fingerprint": _features_fingerprint(self._features),
            "num_rows": materialized.num_rows,
            "num_shards": 1,
            "shard_filenames": ["shard-00000.arrow"],
            "shard_row_counts": [materialized.num_rows],
        }
        (cache_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))

        return StableDataset(
            features=self._features,
            info=self._info,
            shard_dir=cache_dir,
            shard_paths=[out_path],
            shard_row_counts=[materialized.num_rows],
            num_rows=materialized.num_rows,
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _transform=self._transform,
        )

    # -- Internal helpers -----------------------------------------------------

    def _view_with_indices(self, indices: np.ndarray) -> StableDataset:
        """Return a new StableDataset sharing the same backing data."""
        return StableDataset(
            features=self._features,
            info=self._info,
            backend=self._backend,
            shard_dir=self._shard_dir,
            shard_paths=self._shard_paths,
            shard_row_counts=self._shard_row_counts,
            _indices=np.asarray(indices, dtype=np.int64),
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _transform=self._transform,
        )

    def _shallow_copy(self, **overrides) -> StableDataset:
        """Return a shallow copy with optional attribute overrides."""
        kw = {
            "features": self._features,
            "info": self._info,
            "backend": self._backend,
            "shard_dir": self._shard_dir,
            "shard_paths": self._shard_paths,
            "shard_row_counts": self._shard_row_counts,
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_decode_images": self._decode_images,
            "_transform": self._transform,
        }
        kw.update(overrides)
        return StableDataset(**kw)


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``."""

    pass
