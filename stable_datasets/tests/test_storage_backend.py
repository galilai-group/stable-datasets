"""Cross-backend tests for the StorageBackend protocol.

Every test in this file runs twice via a parameterized fixture: once
with :class:`ArrowBackend` directly, once with :class:`LanceBackend`
reading a Lance dataset converted from the same Arrow cache. Any test
that passes for Arrow and fails for Lance is evidence of a protocol
leak -- some place where :class:`StableDataset` or its consumers reach
past the abstraction and depend on Arrow-specific behavior.

This is the load-bearing test for the Lance-migration plan: if the
protocol is watertight enough to make both backends interchangeable at
this level, the same should hold for real workloads.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.backend import ArrowBackend
from stable_datasets.cache import write_sharded_arrow_cache
from stable_datasets.lance_backend import LanceBackend
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value
from stable_datasets.tools.arrow_to_lance import arrow_to_lance


BackendKind = str  # "arrow" | "lance"


@pytest.fixture(params=["arrow", "lance"], ids=["arrow", "lance"])
def make_ds(request, tmp_path) -> Callable[..., StableDataset]:
    """Factory fixture: returns a callable that builds a file-backed
    :class:`StableDataset` on the parameterized backend.

    For the Lance variant, the Arrow cache is converted on the fly via
    ``arrow_to_lance`` -- same source generator, same rows, different
    storage format.
    """
    kind: BackendKind = request.param

    def _make(n: int = 10, batch_size: int = 5) -> StableDataset:
        features = Features(
            {"x": Value("int32"), "label": ClassLabel(names=["a", "b"])}
        )
        info = DatasetInfo(features=features)

        def gen():
            for i in range(n):
                yield i, {"x": i, "label": i % 2}

        cache_dir = tmp_path / f"arrow_cache_{n}"
        meta = write_sharded_arrow_cache(
            gen(), features, cache_dir, batch_size=batch_size
        )

        if kind == "arrow":
            backend = ArrowBackend(
                shard_paths=meta.shard_paths,
                shard_row_counts=meta.shard_row_counts,
                schema=features.to_arrow_schema(),
            )
        elif kind == "lance":
            lance_dir = tmp_path / f"lance_cache_{n}"
            arrow_to_lance(cache_dir, lance_dir, overwrite=True)
            backend = LanceBackend(uri=lance_dir)
        else:
            raise AssertionError(f"unknown backend {kind}")

        return StableDataset(
            features=features, info=info, backend=backend, num_rows=meta.num_rows
        )

    _make.kind = kind  # type: ignore[attr-defined]
    return _make


# ── Shape + scalar access ────────────────────────────────────────────────────


class TestShape:
    def test_len(self, make_ds):
        ds = make_ds(n=10)
        assert len(ds) == 10

    def test_features_preserved(self, make_ds):
        ds = make_ds(n=5)
        assert set(ds.features.keys()) == {"x", "label"}
        assert isinstance(ds.features["label"], ClassLabel)
        assert ds.features["label"].num_classes == 2

    def test_column_names(self, make_ds):
        ds = make_ds(n=5)
        assert set(ds.column_names) == {"x", "label"}

    def test_getitem_scalar(self, make_ds):
        ds = make_ds(n=5)
        for i in range(5):
            row = ds[i]
            assert row["x"] == i
            assert row["label"] in (0, 1)

    def test_negative_indexing(self, make_ds):
        ds = make_ds(n=5)
        assert ds[-1]["x"] == 4
        assert ds[-5]["x"] == 0

    def test_index_out_of_range(self, make_ds):
        ds = make_ds(n=5)
        with pytest.raises((IndexError, ValueError, Exception)):
            _ = ds[10]


# ── Batched access ───────────────────────────────────────────────────────────


class TestBatchedAccess:
    def test_getitems(self, make_ds):
        ds = make_ds(n=20)
        # __getitems__ is the torch DataLoader batched-fetch API
        rows = ds.__getitems__([0, 5, 19])
        assert len(rows) == 3
        assert [r["x"] for r in rows] == [0, 5, 19]

    def test_getitems_empty(self, make_ds):
        ds = make_ds(n=10)
        rows = ds.__getitems__([])
        assert rows == []

    def test_getitems_duplicates_and_order(self, make_ds):
        ds = make_ds(n=10)
        rows = ds.__getitems__([7, 2, 7, 0])
        assert [r["x"] for r in rows] == [7, 2, 7, 0]


# ── Iteration ────────────────────────────────────────────────────────────────


class TestIteration:
    def test_full_pass_count(self, make_ds):
        ds = make_ds(n=17, batch_size=5)
        assert sum(1 for _ in ds) == 17

    def test_full_pass_values(self, make_ds):
        ds = make_ds(n=10)
        xs = [row["x"] for row in ds]
        assert sorted(xs) == list(range(10))


# ── Pickle (DataLoader worker fork simulation) ───────────────────────────────


class TestPickle:
    def test_roundtrip(self, make_ds):
        ds = make_ds(n=12)
        blob = pickle.dumps(ds)
        ds2 = pickle.loads(blob)
        assert len(ds2) == 12
        assert ds2[3]["x"] == 3
        assert ds2[11]["x"] == 11

    def test_roundtrip_preserves_features(self, make_ds):
        ds = make_ds(n=5)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert isinstance(ds2.features["label"], ClassLabel)
        assert ds2.features["label"].names == ["a", "b"]


# ── Protocol conformance ─────────────────────────────────────────────────────


class TestProtocol:
    def test_backend_satisfies_protocol(self, make_ds):
        from stable_datasets.storage import StorageBackend

        ds = make_ds(n=5)
        assert isinstance(ds._backend, StorageBackend)

    def test_num_rows_property(self, make_ds):
        ds = make_ds(n=13)
        assert ds._backend.num_rows == 13

    def test_schema_property(self, make_ds):
        ds = make_ds(n=5)
        schema = ds._backend.schema
        assert isinstance(schema, pa.Schema)
        assert {f.name for f in schema} == {"x", "label"}

    def test_take_batch(self, make_ds):
        ds = make_ds(n=10)
        sub = ds._backend.take([0, 3, 9])
        assert isinstance(sub, pa.Table)
        assert sub.num_rows == 3
        assert sub.column("x").to_pylist() == [0, 3, 9]

    def test_take_numpy_indices(self, make_ds):
        ds = make_ds(n=10)
        sub = ds._backend.take(np.array([1, 4, 7], dtype=np.int64))
        assert sub.num_rows == 3
        assert sub.column("x").to_pylist() == [1, 4, 7]

    def test_slice(self, make_ds):
        ds = make_ds(n=20)
        sub = ds._backend.slice(5, 3)
        assert sub.num_rows == 3
        assert sub.column("x").to_pylist() == [5, 6, 7]

    def test_iter_batches_row_count(self, make_ds):
        ds = make_ds(n=23, batch_size=7)
        total = sum(b.num_rows for b in ds._backend.iter_batches())
        assert total == 23
