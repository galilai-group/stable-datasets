"""Map-style dataset built on a pluggable storage backend.

Provides :class:`StableDataset` (single split) and
:class:`StableDatasetDict` (multi-split), exposing ``__len__``,
``__getitem__``, ``__getitems__``, ``__iter__``, ``.features``, and
``.train_test_split()``.

Architecture: three layers with strict boundaries::

    StorageBackend  -> row access, iteration, pickling (returns Arrow types)
        |
    Formatter       -> Arrow -> user type (PIL / torch / numpy / raw)
        |
    StableDataset   -> orchestrates backend + formatter + indices + transform

:class:`StableDataset` depends only on the :class:`StorageBackend`
protocol, never on a concrete implementation or on-disk layout.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .backends.arrow_shards import ArrowBackend
from .backends.protocol import StorageBackend
from .cache import _CACHE_FORMAT_VERSION, _features_fingerprint
from .formatting import get_formatter
from .schema import (
    Array3D,
    DatasetInfo,
    Features,
    Image,
    Sequence,
    Video,
    VideoDecodeConfig,
    VideoRef,
)


class StableDataset:
    """A single-split dataset backed by Arrow.

    Users interact with rows, columns, and transforms — never with files or
    shards.  All storage details are delegated to ``ArrowBackend``.

    Construction:

    1. **File-backed** (typical) — pass ``backend=ArrowBackend(shard_paths=...)``.
    2. **In-memory** — pass ``backend=ArrowBackend(table=table)``.
    3. **Indexed view** — pass ``_indices=array`` to create a virtual view
       sharing the same backend.  Zero data copying.
    """

    def __init__(
        self,
        features: Features,
        info: DatasetInfo,
        *,
        # Storage — pass exactly one of: backend, shard_paths, table
        backend: StorageBackend | None = None,
        shard_paths: list[Path] | None = None,
        shard_row_counts: list[int] | None = None,
        table: pa.Table | None = None,
        num_rows: int | None = None,
        # Index indirection
        _indices: np.ndarray | None = None,
        # Format control
        _format_type: str | None = None,
        _decode_images: bool = True,
        _video_decode_config: VideoDecodeConfig | None = None,
        _transform: Callable | None = None,
        _cache_dir: Path | None = None,
    ):
        self._features = features
        self._info = info

        # Build backend from convenience args if not provided directly
        self._backend: StorageBackend
        if backend is not None:
            self._backend = backend
        elif shard_paths is not None:
            self._backend = ArrowBackend(
                shard_paths=shard_paths,
                shard_row_counts=shard_row_counts,
                schema=features.to_arrow_schema(),
            )
        elif table is not None:
            self._backend = ArrowBackend(table=table, schema=features.to_arrow_schema())
        else:
            raise ValueError("Must provide one of: backend, shard_paths, table")

        # Index indirection
        self._indices = np.asarray(_indices, dtype=np.int64) if _indices is not None else None

        # Format and transform
        self._format_type = _format_type
        self._decode_images = _decode_images
        self._video_decode_config = _video_decode_config
        self._transform = _transform
        self._cache_dir = Path(_cache_dir) if _cache_dir is not None else self._infer_cache_dir()
        self._formatter = get_formatter(
            _format_type,
            features,
            decode_images=_decode_images,
            cache_dir=self._cache_dir,
        )

        # Precompute whether we have binary columns (Image/Array3D/Video).
        self._has_binary_cols = any(isinstance(f, (Image, Array3D, Video)) for f in features.values())

        # Cache row count
        if self._indices is not None:
            self._num_rows = len(self._indices)
        elif num_rows is not None:
            self._num_rows = num_rows
        else:
            self._num_rows = self._backend.num_rows

    # -- Pickle / DataLoader compatibility ------------------------------------

    def __getstate__(self):
        return {
            "features": self._features,
            "info": self._info,
            "backend": self._backend,
            "num_rows": self._num_rows,
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_decode_images": self._decode_images,
            "_video_decode_config": self._video_decode_config,
            "_transform": self._transform,
            "_cache_dir": self._cache_dir,
        }

    def __setstate__(self, state):
        self.__init__(
            features=state["features"],
            info=state["info"],
            backend=state["backend"],
            num_rows=state["num_rows"],
            _indices=state.get("_indices"),
            _format_type=state.get("_format_type"),
            _decode_images=state.get("_decode_images", True),
            _video_decode_config=state.get("_video_decode_config"),
            _transform=state.get("_transform"),
            _cache_dir=state.get("_cache_dir"),
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
        return list(self._features.keys())

    @property
    def num_rows(self) -> int:
        return len(self)

    @property
    def table(self) -> pa.Table:
        """Materialize and return the full Arrow table.

        For single-file datasets this is a cheap mmap. For multi-file
        datasets this concatenates all files — prefer ``__getitem__``
        or ``__iter__`` for row access. Use this for bulk operations
        like column mutations.
        """
        return self._backend.table

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
            row = self._apply_video_decode_row(row, sample_index=idx)
            if self._transform is not None:
                row = self._transform(row)
            return row

        if isinstance(idx, slice):
            indices = np.arange(*idx.indices(len(self)), dtype=np.int64)
            if self._indices is not None:
                indices = self._indices[indices]
            return self._view_with_indices(indices)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched sample loading (called by PyTorch DataLoader).

        Policy is backend-sensitive. ArrowBackend's ``slice(i, 1)`` on
        an mmap'd table is zero-copy and unbeatable per-row, while its
        ``take`` rebuilds chunk offsets -- so for binary columns the
        per-row loop wins. LanceBackend inverts this: every call
        crosses the Python<->Rust async boundary at fixed cost, so the
        batched ``take`` path amortizes it and the per-row loop is
        catastrophic. Backends advertise their preference via
        ``prefer_batched_take``; when absent it defaults to False
        (Arrow's shape).
        """
        prefer_batched = getattr(self._backend, "prefer_batched_take", False)
        if self._has_binary_cols and not prefer_batched:
            rows = []
            sample_indices = []
            for idx in indices:
                normalized = self._normalize_index(int(idx))
                physical = int(self._indices[normalized]) if self._indices is not None else normalized
                rows.append(self._formatter.format_row(self._backend.get_row(physical)))
                sample_indices.append(normalized)
            rows = self._apply_video_decode_batch(rows, sample_indices=sample_indices)
            if self._transform is not None:
                rows = [self._transform(row) for row in rows]
            return rows

        idx_array = np.asarray(indices, dtype=np.int64)
        sample_indices = [self._normalize_index(int(idx)) for idx in idx_array.tolist()]
        if self._indices is not None:
            idx_array = self._indices[np.asarray(sample_indices, dtype=np.int64)]
        else:
            idx_array = np.asarray(sample_indices, dtype=np.int64)

        batch_table = self._backend.take(idx_array)
        rows = self._formatter.format_batch(batch_table)
        rows = self._apply_video_decode_batch(rows, sample_indices=sample_indices)

        if self._transform is not None:
            rows = [self._transform(row) for row in rows]

        return rows

    def __iter__(self):
        """Iterate over all rows, yielding decoded dicts."""
        if self._backend.is_file_backed and self._indices is None:
            yield from self._iter_batches_formatted(self._backend.iter_batches())
        else:
            for i in range(len(self)):
                yield self[i]

    def iter_epoch(self, *, shuffle_shards: bool = True, seed: int | None = None):
        """Iterate with optional shard-level shuffling."""
        if self._indices is not None:
            for i in range(len(self)):
                yield self[i]
        elif self._backend.is_file_backed:
            yield from self._iter_batches_formatted(self._backend.iter_batches(shuffle=shuffle_shards, seed=seed))
        else:
            yield from self

    def _iter_batches_formatted(self, batch_iter):
        """Format Arrow batches in bulk and yield individual rows."""
        sample_offset = 0
        for batch in batch_iter:
            # Use batch formatting: one to_pydict() + column-wise decode
            rows = self._formatter.format_batch(batch)
            sample_indices = list(range(sample_offset, sample_offset + len(rows)))
            rows = self._apply_video_decode_batch(rows, sample_indices=sample_indices)
            sample_offset += len(rows)
            if self._transform is not None:
                for row in rows:
                    yield self._transform(row)
            else:
                yield from rows

    # -- Selection / shuffling / filtering ------------------------------------

    def select(self, indices) -> StableDataset:
        """Return a view containing only the specified row indices."""
        indices = np.asarray(indices, dtype=np.int64)
        if self._indices is not None:
            indices = self._indices[indices]
        return self._view_with_indices(indices)

    def shuffle(self, seed: int = 42) -> StableDataset:
        """Return a shuffled view."""
        perm = np.random.default_rng(seed).permutation(len(self))
        return self.select(perm)

    def filter(
        self,
        fn: Callable,
        *,
        batched: bool = False,
        batch_size: int = 1000,
    ) -> StableDataset:
        """Return a view containing rows where ``fn`` returns True.

        Non-batched (default): ``fn(row_dict) -> bool``, applied per row.
        Batched: ``fn(dict_of_lists) -> list[bool]``, applied per batch
        using sequential scan for better performance on large datasets.

        Returns an indexed view — no data is materialized.
        """
        if not batched:
            matching = [i for i in range(len(self)) if fn(self[i])]
        elif self._backend.is_file_backed and self._indices is None:
            # Sequential scan via iter_batches — avoids take() overhead
            matching = []
            row_offset = 0
            for batch in self._backend.iter_batches():
                batch_dict = batch.to_pydict()
                mask = fn(batch_dict)
                for i, keep in enumerate(mask):
                    if keep:
                        matching.append(row_offset + i)
                row_offset += batch.num_rows
        else:
            # Indexed or in-memory: gather batches via take()
            matching = []
            for start in range(0, len(self), batch_size):
                end = min(start + batch_size, len(self))
                idx_array = np.arange(start, end, dtype=np.int64)
                if self._indices is not None:
                    idx_array = self._indices[idx_array]
                batch_table = self._backend.take(idx_array)
                batch_dict = batch_table.to_pydict()
                mask = fn(batch_dict)
                for i, keep in enumerate(mask):
                    if keep:
                        matching.append(start + i)
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

    # -- Materializing transformations ----------------------------------------

    def map(
        self,
        fn: Callable,
        *,
        batched: bool = False,
        batch_size: int = 1000,
        with_indices: bool = False,
        remove_columns: list[str] | None = None,
        features: Features | None = None,
        cache_dir: Path | str | None = None,
    ) -> StableDataset:
        """Apply a function to every row/batch and return a new dataset.

        This is a **materializing operation** — output is written
        incrementally to Arrow IPC files via the sharded cache pipeline,
        so memory usage stays bounded regardless of dataset size.
        Use ``with_transform`` for lazy per-row transforms during iteration.

        Non-batched: ``fn(row_dict) -> row_dict`` (or ``fn(row_dict, idx)``
        if ``with_indices=True``).
        Batched: ``fn(dict_of_lists) -> dict_of_lists`` (or
        ``fn(dict_of_lists, list_of_indices)``).

        Parameters
        ----------
        features : Features, optional
            Output schema. If None, columns matching input features keep
            their types; new columns are inferred from Arrow types.
            Provide explicitly when the output schema is ambiguous.
        cache_dir : path, optional
            Where to write the output cache. If None, uses a temp directory.
        """
        from .cache import write_sharded_arrow_cache

        remove_set = set(remove_columns) if remove_columns else set()

        # Infer output features from a probe example if not provided
        if features is None:
            probe = self._backend.get_row(int(self._indices[0]) if self._indices is not None else 0)
            if batched:
                probe_batch = {k: [v] for k, v in probe.items()}
                probe_out = fn(probe_batch, [0]) if with_indices else fn(probe_batch)
                probe_row = {k: v[0] for k, v in probe_out.items()}
            else:
                probe_row = fn(probe, 0) if with_indices else fn(probe)

            features = Features()
            for col_name in probe_row:
                if col_name in remove_set:
                    continue
                if col_name in self._features:
                    features[col_name] = self._features[col_name]
                else:
                    # Infer from the probe value
                    val = probe_row[col_name]
                    if isinstance(val, int):
                        features[col_name] = _infer_feature(pa.int64())
                    elif isinstance(val, float):
                        features[col_name] = _infer_feature(pa.float64())
                    elif isinstance(val, str):
                        features[col_name] = _infer_feature(pa.string())
                    else:
                        features[col_name] = _infer_feature(pa.binary())

        # Build output generator that feeds write_sharded_arrow_cache
        def _map_gen():
            out_idx = 0
            if not batched:
                for i in range(len(self)):
                    physical = int(self._indices[i]) if self._indices is not None else i
                    row = self._backend.get_row(physical)
                    out = fn(row, i) if with_indices else fn(row)
                    if remove_set:
                        out = {k: v for k, v in out.items() if k not in remove_set}
                    yield out_idx, out
                    out_idx += 1
            else:
                for start in range(0, len(self), batch_size):
                    end = min(start + batch_size, len(self))
                    idx_array = np.arange(start, end, dtype=np.int64)
                    if self._indices is not None:
                        idx_array = self._indices[idx_array]
                    batch_table = self._backend.take(idx_array)
                    batch_dict = batch_table.to_pydict()
                    if with_indices:
                        out = fn(batch_dict, list(range(start, end)))
                    else:
                        out = fn(batch_dict)
                    if remove_set:
                        out = {k: v for k, v in out.items() if k not in remove_set}
                    # Expand batch output into individual rows
                    n_out = len(next(iter(out.values())))
                    for i in range(n_out):
                        yield out_idx, {k: v[i] for k, v in out.items()}
                        out_idx += 1

        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp(prefix=".map_"))
        else:
            cache_dir = Path(cache_dir)

        meta = write_sharded_arrow_cache(
            _map_gen(),
            features,
            cache_dir,
            batch_size=batch_size,
            lineage={
                "operation": "map",
                "batched": batched,
                "with_indices": with_indices,
                "remove_columns": remove_columns,
                "source_num_rows": len(self),
            },
        )

        return StableDataset(
            features=features,
            info=self._info,
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
            num_rows=meta.num_rows,
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _video_decode_config=self._video_decode_config,
            _transform=self._transform,
            _cache_dir=self._cache_dir,
        )

    # -- Column mutations -----------------------------------------------------

    def _logical_table(self) -> pa.Table:
        """Return the table reflecting the current logical view.

        If this dataset has an indices mapping, materializes only the
        selected rows. Column mutations use this table to respect
        indexed views.
        """
        tbl = self.table
        if self._indices is not None:
            tbl = tbl.take(self._indices)
        return tbl

    def add_column(self, name: str, column) -> StableDataset:
        """Return a new dataset with an additional column.

        ``column`` can be a ``pa.Array``, a Python list, or a numpy array.
        """
        if not isinstance(column, pa.Array):
            column = pa.array(column)
        tbl = self._logical_table().append_column(name, column)
        new_features = Features({**self._features, name: _infer_feature(column.type)})
        return self._with_table(tbl, new_features)

    def remove_columns(self, columns: list[str] | str) -> StableDataset:
        """Return a new dataset without the specified columns."""
        if isinstance(columns, str):
            columns = [columns]
        tbl = self._logical_table().drop_columns(columns)
        new_features = Features({k: v for k, v in self._features.items() if k not in columns})
        return self._with_table(tbl, new_features)

    def rename_column(self, old_name: str, new_name: str) -> StableDataset:
        """Return a new dataset with a column renamed."""
        tbl = self._logical_table()
        names = [new_name if n == old_name else n for n in tbl.column_names]
        tbl = tbl.rename_columns(names)
        new_features = Features({(new_name if k == old_name else k): v for k, v in self._features.items()})
        return self._with_table(tbl, new_features)

    def rename_columns(self, mapping: dict[str, str]) -> StableDataset:
        """Return a new dataset with columns renamed per the mapping."""
        tbl = self._logical_table()
        names = [mapping.get(n, n) for n in tbl.column_names]
        tbl = tbl.rename_columns(names)
        new_features = Features({mapping.get(k, k): v for k, v in self._features.items()})
        return self._with_table(tbl, new_features)

    # -- Format and transform pipeline ----------------------------------------

    def with_format(self, format_type: str | None) -> StableDataset:
        """Return a view with the specified output format.

        Supported: ``None`` (PIL/numpy/Python), ``"torch"``, ``"numpy"``, ``"raw"``.
        """
        return self._shallow_copy(_format_type=format_type)

    def with_transform(self, fn: Callable | None) -> StableDataset:
        """Return a view with a transform applied after format conversion."""
        return self._shallow_copy(_transform=fn)

    def set_decode(self, decode: bool) -> StableDataset:
        """Control whether Image columns are decoded or left as raw bytes."""
        return self._shallow_copy(_decode_images=decode)

    def set_video_decode(
        self,
        config: VideoDecodeConfig | Mapping | None = None,
        **kwargs,
    ) -> StableDataset:
        """Return a view that decodes a video column at read time.

        Passing ``None`` with no keyword arguments disables video decoding on
        the returned view.
        """
        if config is None and not kwargs:
            return self._shallow_copy(_video_decode_config=None)
        if config is None:
            next_config = VideoDecodeConfig(**kwargs)
        elif isinstance(config, VideoDecodeConfig):
            next_config = replace(config, **kwargs) if kwargs else config
        elif isinstance(config, Mapping):
            data = dict(config)
            data.update(kwargs)
            next_config = VideoDecodeConfig(**data)
        else:
            raise TypeError(
                "set_video_decode expects a VideoDecodeConfig, mapping, None, "
                f"or keyword arguments, got {type(config).__name__}."
            )
        self._validate_video_decode_config(next_config)
        return self._shallow_copy(_video_decode_config=next_config)

    def make_sampler(self, kind: str = "shard_shuffle", **kwargs):
        """Return a backend-aware ``torch.utils.data.Sampler`` for this dataset.

        Convenience wrapper around the classes in
        :mod:`stable_datasets.samplers`. Use as::

            sampler = ds.make_sampler("shard_shuffle", seed=42)
            loader = DataLoader(ds, batch_size=128, sampler=sampler, ...)

        ``DataLoader(ds, shuffle=True)`` (full-random via
        :class:`~torch.utils.data.RandomSampler`) continues to work
        unchanged; this is strictly opt-in for users who want an
        iteration order matched to the backend's I/O layout.

        Parameters
        ----------
        kind : str, default ``"shard_shuffle"``
            Currently the only supported kind.
        **kwargs :
            Forwarded to the underlying sampler class (e.g.
            ``seed``, ``within_shard``).
        """
        from stable_datasets.samplers import ShardShuffleSampler

        if kind == "shard_shuffle":
            return ShardShuffleSampler(self, **kwargs)
        raise ValueError(f"Unknown sampler kind: {kind!r}")

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

    def flatten_indices(self, cache_dir: Path | None = None) -> StableDataset:
        """Materialize an indexed view into a new contiguous Arrow file."""
        if self._indices is None:
            return self

        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp(prefix=".flatten_"))
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
            "layout": "arrow-shards",
            "schema_fingerprint": _features_fingerprint(self._features),
            "num_rows": materialized.num_rows,
            "num_shards": 1,
            "shard_filenames": ["shard-00000.arrow"],
            "shard_row_counts": [materialized.num_rows],
        }
        (cache_dir / "_metadata.json").write_text(json.dumps(meta, indent=2))

        backend = ArrowBackend(
            shard_paths=[out_path],
            shard_row_counts=[materialized.num_rows],
            schema=schema,
        )
        return StableDataset(
            features=self._features,
            info=self._info,
            backend=backend,
            num_rows=materialized.num_rows,
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _video_decode_config=self._video_decode_config,
            _transform=self._transform,
            _cache_dir=self._cache_dir,
        )

    # -- Internal helpers -----------------------------------------------------

    def _view_with_indices(self, indices: np.ndarray) -> StableDataset:
        """Return a new StableDataset sharing the same backend."""
        return StableDataset(
            features=self._features,
            info=self._info,
            backend=self._backend,
            _indices=np.asarray(indices, dtype=np.int64),
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _video_decode_config=self._video_decode_config,
            _transform=self._transform,
            _cache_dir=self._cache_dir,
        )

    def _shallow_copy(self, **overrides) -> StableDataset:
        """Return a shallow copy with optional attribute overrides.

        ``num_rows`` is forwarded from the current instance. Backends
        with process-local lazy state, such as :class:`LanceBackend`,
        can keep dataset handles unopened until the copy is used.
        """
        kw = {
            "features": self._features,
            "info": self._info,
            "backend": self._backend,
            "num_rows": self._num_rows,
            "_indices": self._indices,
            "_format_type": self._format_type,
            "_decode_images": self._decode_images,
            "_video_decode_config": self._video_decode_config,
            "_transform": self._transform,
            "_cache_dir": self._cache_dir,
        }
        kw.update(overrides)
        return StableDataset(**kw)

    def _with_table(self, table: pa.Table, features: Features | None = None) -> StableDataset:
        """Return a new in-memory dataset from a modified table."""
        return StableDataset(
            features=features or self._features,
            info=self._info,
            backend=ArrowBackend(table=table, schema=(features or self._features).to_arrow_schema()),
            _format_type=self._format_type,
            _decode_images=self._decode_images,
            _video_decode_config=self._video_decode_config,
            _transform=self._transform,
            _cache_dir=self._cache_dir,
        )

    def _normalize_index(self, idx: int) -> int:
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for dataset of length {n}")
        return idx

    def _validate_video_decode_config(self, config: VideoDecodeConfig) -> None:
        feat = self._features.get(config.column)
        if not isinstance(feat, Video):
            raise ValueError(f"Video decode column {config.column!r} must exist and be a Video feature.")

    def _apply_video_decode_row(self, row: dict, *, sample_index: int | None) -> dict:
        config = self._video_decode_config
        if config is None:
            return row
        decoded = self._decode_video_value(row, config, sample_index=sample_index)
        out = dict(row)
        out[config.column] = decoded
        return out

    def _apply_video_decode_batch(
        self,
        rows: list[dict],
        *,
        sample_indices: list[int] | None,
    ) -> list[dict]:
        config = self._video_decode_config
        if config is None or not rows:
            return rows
        if config.decode_fn_batched is not None:
            refs = [self._coerce_video_ref(row[config.column], config) for row in rows]
            decoded = config.decode_fn_batched(
                refs,
                config,
                rows=rows,
                sample_indices=sample_indices,
            )
            if len(decoded) != len(rows):
                raise ValueError("decode_fn_batched must return one decoded value per input row.")
            out_rows = [dict(row) for row in rows]
            for row, value in zip(out_rows, decoded):
                row[config.column] = value
            return out_rows
        return [
            self._apply_video_decode_row(
                row,
                sample_index=None if sample_indices is None else sample_indices[i],
            )
            for i, row in enumerate(rows)
        ]

    def _decode_video_value(
        self,
        row: Mapping,
        config: VideoDecodeConfig,
        *,
        sample_index: int | None,
    ):
        ref = self._coerce_video_ref(row[config.column], config)
        if config.decode_fn is not None:
            return config.decode_fn(
                ref,
                config,
                row=row,
                sample_index=sample_index,
            )
        return _decode_video_builtin(ref, config, sample_index=sample_index)

    def _coerce_video_ref(self, value, config: VideoDecodeConfig) -> VideoRef:
        if isinstance(value, VideoRef):
            return value
        if isinstance(value, Mapping):
            return VideoRef(value, cache_dir=self._cache_dir)
        raise TypeError(
            f"Video decode column {config.column!r} produced {type(value).__name__}; "
            "expected VideoRef or raw video struct."
        )

    def _infer_cache_dir(self) -> Path | None:
        cache_dir = getattr(self._backend, "cache_dir", None)
        if cache_dir is not None:
            return Path(cache_dir)
        return None


def _decode_video_builtin(
    ref: VideoRef,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
):
    if config.decoder == "cv2":
        frames = _decode_video_cv2(ref, config, sample_index=sample_index)
    elif config.decoder == "decord":
        frames = _decode_video_decord(ref, config, sample_index=sample_index)
    elif config.decoder == "torchcodec":
        frames = _decode_video_torchcodec(ref, config, sample_index=sample_index)
    else:  # pragma: no cover - VideoDecodeConfig validates this.
        raise ValueError(f"Unknown video decoder {config.decoder!r}.")
    return _format_decoded_video_frames(frames, config, sample_index=sample_index)


def _decode_video_cv2(
    ref: VideoRef,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("VideoDecodeConfig(decoder='cv2') requires opencv-python.") from exc

    path = ref.path
    if path is None:
        raise ValueError("cv2 video decoding requires a filesystem path.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video for decoding: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if not frames:
                raise ValueError(f"Could not decode any frames from {path}")
            indices = _sample_video_indices(len(frames), config, sample_index=sample_index)
            return np.stack([frames[int(i)] for i in indices], axis=0)

        indices = _sample_video_indices(total_frames, config, sample_index=sample_index)
        decoded = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Could not decode frame {int(idx)} from {path}")
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return np.stack(decoded, axis=0)
    finally:
        cap.release()


def _decode_video_decord(
    ref: VideoRef,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
) -> np.ndarray:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("VideoDecodeConfig(decoder='decord') requires decord.") from exc
    path = ref.path
    if path is None:
        raise ValueError("decord video decoding requires a filesystem path.")
    reader = VideoReader(str(path), ctx=cpu(0))
    indices = _sample_video_indices(len(reader), config, sample_index=sample_index)
    return reader.get_batch(indices.astype(np.int64).tolist()).asnumpy()


def _decode_video_torchcodec(
    ref: VideoRef,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
) -> np.ndarray:
    try:
        import cv2
        import torch
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise ImportError(
            "VideoDecodeConfig(decoder='torchcodec') requires torchcodec, torch, and opencv-python."
        ) from exc

    path = ref.path
    if path is None:
        raise ValueError("torchcodec video decoding requires a filesystem path.")

    cap = cv2.VideoCapture(str(path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if total_frames <= 0 or fps <= 0:
        raise ValueError(f"Could not determine frame count/fps for {path}")

    decoder = VideoDecoder(str(path))
    indices = _sample_video_indices(total_frames, config, sample_index=sample_index)
    frames = []
    for idx in indices:
        frame = decoder.get_frame_at(float(idx) / fps)
        data = getattr(frame, "data", frame)
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype.kind == "f" and arr.max(initial=0) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)
        frames.append(arr[..., :3])
    return np.stack(frames, axis=0)


def _sample_video_indices(
    total_frames: int,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
) -> np.ndarray:
    total_frames = int(total_frames)
    if total_frames <= 0:
        raise ValueError("Cannot sample frames from an empty video.")
    num_frames = int(config.num_frames)
    stride = int(config.frame_stride)

    if config.sampling == "uniform":
        if total_frames >= num_frames:
            return np.rint(np.linspace(0, total_frames - 1, num_frames)).astype(np.int64)
        return _pad_video_indices(np.arange(total_frames, dtype=np.int64), config)

    span = (num_frames - 1) * stride + 1
    if total_frames >= span:
        if config.sampling == "start":
            start = 0
        elif config.sampling == "center":
            start = (total_frames - span) // 2
        elif config.sampling == "random":
            if config.seed is None:
                rng = np.random.default_rng()
            else:
                idx_seed = 0 if sample_index is None else int(sample_index)
                rng = np.random.default_rng(int(config.seed) + idx_seed)
            start = int(rng.integers(0, total_frames - span + 1))
        else:  # pragma: no cover - VideoDecodeConfig validates this.
            raise ValueError(f"Unknown sampling strategy {config.sampling!r}.")
        return start + np.arange(num_frames, dtype=np.int64) * stride

    base = np.arange(0, total_frames, stride, dtype=np.int64)
    if base.size == 0:
        base = np.array([0], dtype=np.int64)
    return _pad_video_indices(base[:num_frames], config)


def _pad_video_indices(indices: np.ndarray, config: VideoDecodeConfig) -> np.ndarray:
    num_frames = int(config.num_frames)
    if indices.size >= num_frames:
        return indices[:num_frames].astype(np.int64)
    if config.pad == "error":
        raise ValueError(f"Video has only {indices.size} sampled frames, but num_frames={num_frames}.")
    if config.pad == "repeat_last":
        pad = np.full(num_frames - indices.size, int(indices[-1]), dtype=np.int64)
        return np.concatenate([indices.astype(np.int64), pad])
    if config.pad == "loop":
        return np.resize(indices.astype(np.int64), num_frames)
    raise ValueError(f"Unknown pad strategy {config.pad!r}.")


def _format_decoded_video_frames(
    frames: np.ndarray,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
):
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Decoded video must have shape (T, H, W, C), got {frames.shape}.")
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    frames = _resize_video_frames(frames, config)
    frames = _crop_video_frames(frames, config, sample_index=sample_index)

    if config.dtype == "float32":
        frames = frames.astype(np.float32, copy=False)
        if config.scale == "zero_one":
            frames = frames / 255.0
    elif config.dtype == "uint8":
        frames = frames.astype(np.uint8, copy=False)
    else:  # pragma: no cover - VideoDecodeConfig validates this.
        raise ValueError(f"Unknown decoded dtype {config.dtype!r}.")

    frames = _layout_video_frames(frames, config.layout)
    if config.output == "numpy":
        return np.ascontiguousarray(frames)
    if config.output == "torch":
        import torch

        return torch.from_numpy(np.ascontiguousarray(frames))
    raise ValueError(f"Unknown decoded output {config.output!r}.")


def _resize_video_frames(frames: np.ndarray, config: VideoDecodeConfig) -> np.ndarray:
    if config.resize is None:
        return frames
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("VideoDecodeConfig(resize=...) requires opencv-python.") from exc
    if isinstance(config.resize, int):
        height = width = int(config.resize)
    else:
        height, width = config.resize
    resized = [cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(resized, axis=0)


def _crop_video_frames(
    frames: np.ndarray,
    config: VideoDecodeConfig,
    *,
    sample_index: int | None,
) -> np.ndarray:
    if config.crop == "none":
        return frames
    if config.resize is None:
        raise ValueError("VideoDecodeConfig(crop=...) requires resize to define the crop size.")
    if isinstance(config.resize, int):
        crop_h = crop_w = int(config.resize)
    else:
        crop_h, crop_w = config.resize
    _, height, width, _ = frames.shape
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop size {(crop_h, crop_w)} exceeds frame size {(height, width)}.")
    if config.crop == "center":
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
    elif config.crop == "random":
        if config.seed is None:
            rng = np.random.default_rng()
        else:
            idx_seed = 0 if sample_index is None else int(sample_index)
            rng = np.random.default_rng(int(config.seed) + idx_seed)
        top = int(rng.integers(0, height - crop_h + 1))
        left = int(rng.integers(0, width - crop_w + 1))
    else:  # pragma: no cover - VideoDecodeConfig validates this.
        raise ValueError(f"Unknown crop mode {config.crop!r}.")
    return frames[:, top : top + crop_h, left : left + crop_w, :]


def _layout_video_frames(frames: np.ndarray, layout: str) -> np.ndarray:
    if layout == "THWC":
        return frames
    if layout == "TCHW":
        return np.transpose(frames, (0, 3, 1, 2))
    if layout == "CTHW":
        return np.transpose(frames, (3, 0, 1, 2))
    raise ValueError(f"Unknown decoded video layout {layout!r}.")


def _infer_feature(arrow_type: pa.DataType):
    """Infer a Feature type from an Arrow data type.

    Covers common scalar, integer, float, string, binary, boolean, and
    list types. Raises ``TypeError`` for types that cannot be mapped
    unambiguously — callers should provide explicit ``features=`` instead.
    """
    from .schema import Value

    # Integer types
    _INT_MAP = {
        pa.int8(): "int8",
        pa.int16(): "int16",
        pa.int32(): "int32",
        pa.int64(): "int64",
        pa.uint8(): "uint8",
        pa.uint16(): "uint16",
        pa.uint32(): "uint32",
        pa.uint64(): "uint64",
    }
    if arrow_type in _INT_MAP:
        return Value(_INT_MAP[arrow_type])

    # Float types
    if pa.types.is_float16(arrow_type):
        return Value("float16")
    if pa.types.is_float32(arrow_type):
        return Value("float32")
    if pa.types.is_float64(arrow_type):
        return Value("float64")

    # Boolean
    if pa.types.is_boolean(arrow_type):
        return Value("bool")

    # String
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return Value("string")

    # Binary
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return Value("binary")

    # List → Sequence
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        inner = _infer_feature(arrow_type.value_type)
        return Sequence(inner)

    raise TypeError(
        f"Cannot infer Feature type for Arrow type {arrow_type!r}. "
        f"Provide explicit features= to map() or add_column()."
    )


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``."""

    def set_video_decode(
        self,
        config: VideoDecodeConfig | Mapping | None = None,
        **kwargs,
    ) -> StableDatasetDict:
        """Return a split dict where each split applies the same video decode view."""
        return StableDatasetDict(
            {split: dataset.set_video_decode(config, **kwargs) for split, dataset in self.items()}
        )
