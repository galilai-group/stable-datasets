"""Base classes for tabular ML datasets.

``TabularDataset`` wraps a PyArrow table (all columns, including the target)
alongside optional pre-defined train/test split indices for n-fold cross-
validation.  It is the shared return type for all tabular suite loaders
(TabArena, and future suites).

``TabularBaseDatasetBuilder`` is a :class:`~stable_datasets.utils.BaseDatasetBuilder`
subclass tailored for tabular suites.  It keeps the same SOURCE validation
contract (``homepage`` + ``citation`` required) but drops the ``assets``
requirement (tabular data is typically downloaded programmatically, e.g. via
the ``openml`` package) and returns a :class:`TabularDataset` rather than a
:class:`~stable_datasets.arrow_dataset.StableDataset`.

Unlike ``StableDataset``, tabular datasets are stored as a single Arrow IPC
file rather than shards, because tabular datasets are typically small enough
to fit in memory and require whole-table access for fold operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa

from stable_datasets.schema import BuilderConfig
from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    _default_processed_cache_dir,
)


@dataclass(frozen=True)
class TabularTaskInfo:
    """Static metadata for a single tabular ML task.

    ``n_rows`` and ``n_features`` always reflect the *full* dataset, even when
    this object is attached to a fold subset.
    """

    task_id: int
    task_name: str
    problem_type: str  # "binary" | "multiclass" | "regression"
    target_col: str
    n_rows: int
    n_features: int
    n_folds: int
    n_repeats: int


# Splits are stored as {repeat_id: {fold_id: (train_indices, test_indices)}}.
# Indices are plain Python lists of ints so they serialise to JSON cleanly.
_Splits = dict[int, dict[int, tuple[list[int], list[int]]]]


class TabularDataset:
    """Arrow-backed tabular ML dataset with optional pre-defined CV splits.

    Construction is handled by suite loaders (e.g. ``TabArena``); you should
    not need to instantiate this class directly.

    Row access::

        ds[0]              # dict of column values for one row
        ds[10:50]          # new TabularDataset with rows 10-49
        for row in ds: ... # iterate all rows as dicts

    Data as pandas / Arrow::

        ds.to_pandas()     # pd.DataFrame (full table, features + target)
        ds.X               # pa.Table  (feature columns only)
        ds.y               # pa.ChunkedArray  (target column)

    Cross-validation::

        train, test = ds.get_fold(fold=0, repeat=0)
        for fold, repeat, train, test in ds.iter_folds():
            ...

    Metadata::

        ds.task_id, ds.task_name, ds.problem_type
        ds.target_col, ds.n_folds, ds.n_repeats
        ds.info            # TabularTaskInfo dataclass
    """

    def __init__(
        self,
        table: pa.Table,
        info: TabularTaskInfo,
        *,
        splits: _Splits | None = None,
    ):
        self._table = table
        self._info = info
        # splits is None for fold subsets (they have no further CV splits).
        self._splits: _Splits = splits or {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def info(self) -> TabularTaskInfo:
        return self._info

    @property
    def task_id(self) -> int:
        return self._info.task_id

    @property
    def task_name(self) -> str:
        return self._info.task_name

    @property
    def problem_type(self) -> str:
        return self._info.problem_type

    @property
    def target_col(self) -> str:
        return self._info.target_col

    @property
    def n_folds(self) -> int:
        return self._info.n_folds

    @property
    def n_repeats(self) -> int:
        return self._info.n_repeats

    # ------------------------------------------------------------------
    # Row-level access
    # ------------------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """The underlying Arrow table (all columns, including the target)."""
        return self._table

    @property
    def X(self) -> pa.Table:
        """Feature columns only (all columns except the target)."""
        cols = [c for c in self._table.column_names if c != self._info.target_col]
        return self._table.select(cols)

    @property
    def y(self) -> pa.ChunkedArray:
        """Target column as a PyArrow ChunkedArray."""
        return self._table.column(self._info.target_col)

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int | slice) -> dict | TabularDataset:
        """Return a row dict (int) or a new in-memory TabularDataset (slice)."""
        if isinstance(idx, int):
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"Index {idx} out of range for dataset of length {n}")
            return {col: self._table.column(col)[idx].as_py() for col in self._table.column_names}
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return self._take(indices)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __iter__(self) -> Iterator[dict]:
        """Iterate all rows as plain Python dicts."""
        for i in range(len(self)):
            yield {col: self._table.column(col)[i].as_py() for col in self._table.column_names}

    def to_pandas(self) -> pd.DataFrame:
        """Return the full table as a pandas DataFrame (features + target)."""
        return self._table.to_pandas()

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def get_fold(self, fold: int = 0, repeat: int = 0) -> tuple[TabularDataset, TabularDataset]:
        """Return ``(train, test)`` for the given fold/repeat pair.

        Both datasets are in-memory Arrow subsets with no splits of their own.
        Indices are the pre-defined OpenML splits stored at download time.

        Args:
            fold: Zero-based fold index (0 to ``n_folds - 1``).
            repeat: Zero-based repeat index (0 to ``n_repeats - 1``).

        Returns:
            A ``(train, test)`` tuple of ``TabularDataset`` instances.

        Raises:
            ValueError: If this dataset has no pre-defined splits, or if the
                requested fold/repeat combination does not exist.
        """
        if not self._splits:
            raise ValueError(
                "This TabularDataset has no pre-defined splits. "
                "Fold subsets (returned by get_fold / iter_folds) cannot be split further."
            )
        try:
            train_idx, test_idx = self._splits[repeat][fold]
        except KeyError:
            raise ValueError(
                f"No splits for repeat={repeat}, fold={fold}. "
                f"Available: {self.n_repeats} repeat(s), {self.n_folds} fold(s)."
            )
        return self._take(train_idx), self._take(test_idx)

    def iter_folds(self) -> Iterator[tuple[int, int, TabularDataset, TabularDataset]]:
        """Yield ``(fold, repeat, train, test)`` for every fold/repeat pair.

        Iterates repeats in ascending order, and folds within each repeat in
        ascending order.

        Raises:
            ValueError: If this dataset has no pre-defined splits.
        """
        if not self._splits:
            raise ValueError(
                "This TabularDataset has no pre-defined splits. "
                "Fold subsets (returned by get_fold / iter_folds) cannot be split further."
            )
        for repeat in sorted(self._splits):
            for fold in sorted(self._splits[repeat]):
                train, test = self.get_fold(fold=fold, repeat=repeat)
                yield fold, repeat, train, test

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _take(self, indices: list[int]) -> TabularDataset:
        """Return a new in-memory TabularDataset with only the given rows."""
        return TabularDataset(self._table.take(indices), self._info)

    def __repr__(self) -> str:
        return (
            f"TabularDataset("
            f"task={self._info.task_name!r}, "
            f"n_rows={len(self)}, "
            f"problem_type={self._info.problem_type!r}, "
            f"n_folds={self._info.n_folds}, "
            f"n_repeats={self._info.n_repeats}"
            f")"
        )


class TabularBaseDatasetBuilder(BaseDatasetBuilder):
    """Base class for tabular dataset suite builders.

    Inherits VERSION enforcement, SOURCE freezing, and provenance validation
    from :class:`~stable_datasets.utils.BaseDatasetBuilder`.  Differences
    from the standard builder:

    - ``assets`` is **not** required in ``SOURCE`` — tabular suites download
      data programmatically (e.g. via ``openml``), not from static URLs.
    - :meth:`__new__` returns a :class:`TabularDataset`, not a
      :class:`~stable_datasets.arrow_dataset.StableDataset`.
    - :meth:`_info` and :meth:`_generate_examples` are not called.

    Subclasses must define:

    - ``VERSION`` — a :class:`~stable_datasets.schema.Version` instance.
    - ``SOURCE`` — mapping with at least ``homepage`` and ``citation``.
    - :meth:`_build_tabular_dataset` — download/load logic returning a
      :class:`TabularDataset`.

    Example construction (mirrors the standard builder pattern)::

        ds = MyTabularSuite(task_id=42)
        ds = MyTabularSuite(task_id=42, processed_cache_dir="/tmp/cache")
    """

    # Marks this class as an abstract intermediate builder so that
    # BaseDatasetBuilder.__init_subclass__ skips VERSION/SOURCE validation
    # for *this* class only (not for its subclasses).
    _ABSTRACT_BUILDER = True

    @staticmethod
    def _validate_source(source: Mapping) -> None:
        """Validate tabular SOURCE: ``homepage`` and ``citation`` required; ``assets`` optional."""
        if not isinstance(source.get("homepage"), str):
            raise TypeError("SOURCE['homepage'] must be a string and must be present.")
        if not isinstance(source.get("citation"), str):
            raise TypeError("SOURCE['citation'] must be a string and must be present.")

    def __init__(self, config_name: str | None = None, **kwargs):
        """Initialise builder config without calling ``_info()`` (not used here)."""
        if self.BUILDER_CONFIGS:
            if config_name is None:
                config_name = self.DEFAULT_CONFIG_NAME or self.BUILDER_CONFIGS[0].name
            matched = [c for c in self.BUILDER_CONFIGS if c.name == config_name]
            if not matched:
                available = [c.name for c in self.BUILDER_CONFIGS]
                raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
            self.config = matched[0]
        else:
            self.config = BuilderConfig(name=config_name or "default")

    def __new__(
        cls,
        *args,
        task_id: int | None = None,
        task_name: str | None = None,
        processed_cache_dir=None,
        download_dir=None,
        **kwargs,
    ) -> TabularDataset:
        """Download / load a tabular task, caching on disk, and return a :class:`TabularDataset`.

        Args:
            task_id: Suite-specific task identifier.
            task_name: Human-readable task name (resolved to ``task_id`` if needed).
            processed_cache_dir: Override the processed Arrow cache directory.
                Defaults to ``~/.stable_datasets/processed/``.
            download_dir: Override the raw-download directory.
                Defaults to ``~/.stable_datasets/downloads/``.

        Returns:
            A :class:`TabularDataset` with all rows and pre-defined fold/repeat splits.
        """
        instance = object.__new__(cls)

        # Cache dir setup — mirrors BaseDatasetBuilder.__new__.
        if processed_cache_dir is None:
            processed_cache_dir = str(_default_processed_cache_dir())
        instance._processed_cache_dir = Path(processed_cache_dir)

        if download_dir is None:
            download_dir = str(_default_dest_folder())
        instance._raw_download_dir = Path(download_dir)

        # Config init (skips _info() — see __init__ above).
        instance.__init__(*args, **kwargs)

        # SOURCE validation — same contract as BaseDatasetBuilder.
        source = instance._source()
        if not isinstance(source, Mapping):
            raise TypeError(f"{cls.__name__}._source() must return a mapping.")
        cls._validate_source(source)

        return instance._build_tabular_dataset(task_id=task_id, task_name=task_name)

    def _build_tabular_dataset(self, task_id: int | None = None, task_name: str | None = None) -> TabularDataset:
        """Load or download a single task and return a :class:`TabularDataset`.

        Must be overridden by concrete subclasses.
        """
        raise NotImplementedError
