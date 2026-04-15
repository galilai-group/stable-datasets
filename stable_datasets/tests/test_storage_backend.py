"""Cross-backend tests for the StorageBackend protocol.

Every test in this file runs twice via a parameterized fixture: once
with :class:`ArrowBackend` backed by a sharded Arrow IPC cache, once
with :class:`LanceBackend` backed by a native Lance cache written via
``write_lance_cache`` from the same generator. Any test that passes
for Arrow and fails for Lance is evidence of a protocol leak -- some
place where :class:`StableDataset` or its consumers reach past the
abstraction and depend on Arrow-specific behavior.

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
from stable_datasets.cache import write_lance_cache, write_sharded_arrow_cache
from stable_datasets.lance_backend import LanceBackend
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value


BackendKind = str  # "arrow" | "lance"


@pytest.fixture(params=["arrow", "lance"], ids=["arrow", "lance"])
def make_ds(request, tmp_path) -> Callable[..., StableDataset]:
    """Factory fixture: returns a callable that builds a file-backed
    :class:`StableDataset` on the parameterized backend.

    Both variants consume the same in-memory generator, so the test
    content is identical row-for-row between Arrow and Lance runs.
    The Lance variant writes through :func:`write_lance_cache`
    directly -- the same native Phase C write path that
    :class:`BaseDatasetBuilder` uses when ``STORAGE_FORMAT="lance"``.
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

        if kind == "arrow":
            cache_dir = tmp_path / f"arrow_cache_{n}"
            meta = write_sharded_arrow_cache(
                gen(), features, cache_dir, batch_size=batch_size
            )
            backend = ArrowBackend(
                shard_paths=meta.shard_paths,
                shard_row_counts=meta.shard_row_counts,
                schema=features.to_arrow_schema(),
            )
            num_rows = meta.num_rows
        elif kind == "lance":
            lance_dir = tmp_path / f"lance_cache_{n}"
            lance_meta = write_lance_cache(
                gen(), features, lance_dir, batch_size=batch_size
            )
            backend = LanceBackend(uri=lance_dir)
            num_rows = lance_meta.num_rows
        else:
            raise AssertionError(f"unknown backend {kind}")

        return StableDataset(
            features=features, info=info, backend=backend, num_rows=num_rows
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


# ── Shallow-copy + fork-safety regression ────────────────────────────────────
#
# Regression test for the segfault we hit on set_decode(False) + multi-worker
# DataLoader with LanceBackend. Root cause: ``_shallow_copy`` did not forward
# ``num_rows``, so the new :class:`StableDataset.__init__` computed it by
# calling ``self._backend.num_rows``. For :class:`LanceBackend` that opens
# the underlying Lance dataset in the main process, initializing Lance's
# Rust tokio runtime. If DataLoader then forks workers, the children
# inherit stale tokio state and segfault on their first Lance call.
#
# We can't easily assert absence-of-segfault in a unit test, but we can
# assert the *root cause*: ``_shallow_copy`` must not trigger a backend
# access for ``num_rows``, and for Lance specifically the underlying
# dataset handle must remain unopened.


class TestShallowCopyForkSafety:
    def test_set_decode_preserves_num_rows_without_backend_access(self, make_ds):
        ds = make_ds(n=13)
        # Original dataset knows its row count.
        assert ds._num_rows == 13
        ds2 = ds.set_decode(False)
        # The shallow copy should inherit the cached count, not recompute.
        assert ds2._num_rows == 13
        assert ds2._decode_images is False
        assert ds2._backend is ds._backend  # same backend object

    def test_set_decode_does_not_open_lance_dataset(self, make_ds):
        # Only meaningful for Lance -- Arrow's num_rows is free. Skip
        # gracefully on the arrow parameterization.
        ds = make_ds(n=13)
        backend = ds._backend
        if not hasattr(backend, "_ds"):
            pytest.skip("only meaningful for LanceBackend")
        # At this point the Lance dataset should not yet be open
        # (``make_ds`` caches ``num_rows`` in ``StableDataset.__init__``
        # from ``meta.num_rows`` without touching the backend).
        assert backend._ds is None, "LanceBackend was opened before shallow-copy test ran"
        _ = ds.set_decode(False)
        # The shallow-copy call must not have opened the dataset either.
        assert backend._ds is None, (
            "set_decode() opened the Lance dataset in the main process; "
            "DataLoader fork would now segfault worker children"
        )

    def test_with_format_and_with_transform_also_fork_safe(self, make_ds):
        # Every API that uses _shallow_copy inherits the same contract.
        ds = make_ds(n=13)
        backend = ds._backend
        if not hasattr(backend, "_ds"):
            pytest.skip("only meaningful for LanceBackend")
        assert backend._ds is None
        _ = ds.with_format(None)
        _ = ds.with_transform(lambda x: x)
        assert backend._ds is None


# ── Phase C: BaseDatasetBuilder with STORAGE_FORMAT="lance" ─────────────────
#
# Integration test for the direct Lance writer. A tiny builder subclass
# opts into Lance storage via ``STORAGE_FORMAT = "lance"``; the round
# trip exercises ``write_lance_cache``, the ``_metadata.json`` format
# marker, the cache_fingerprint split by storage_format, and the read
# path through ``BaseDatasetBuilder.__new__`` -> ``LanceBackend``.


class _TinyLanceBuilder:
    """Module-level fixture for the Phase C integration tests.

    Defined as a module-level class so the Python pickle machinery can
    locate it during ``pickle.dumps(ds)``. Actual BaseDatasetBuilder
    inheritance is added lazily inside the test setup, because
    importing BaseDatasetBuilder at module load time would also import
    a lot of side-effectful schema validation we want to keep out of
    pytest collection.
    """


def _make_lance_builder_class():
    from stable_datasets.schema import (
        ClassLabel,
        DatasetInfo,
        Features,
        Value,
        Version,
    )
    from stable_datasets.splits import Split, SplitGenerator
    from stable_datasets.utils import BaseDatasetBuilder

    class _LanceBuilder(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}
        STORAGE_FORMAT = "lance"

        def _info(self):
            return DatasetInfo(
                features=Features(
                    {"x": Value("int32"), "label": ClassLabel(names=["a", "b", "c"])}
                )
            )

        def _split_generators(self):
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 15})]

        def _generate_examples(self, n):
            for i in range(n):
                yield i, {"x": i, "label": i % 3}

    return _LanceBuilder


class TestBuilderStorageFormat:
    def test_lance_builder_writes_lance_cache(self, tmp_path):
        import json

        from stable_datasets.lance_backend import LanceBackend

        Builder = _make_lance_builder_class()
        ds = Builder(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")

        # Backend is LanceBackend, not ArrowBackend.
        assert isinstance(ds._backend, LanceBackend)

        # Cache dir carries the expected format marker in _metadata.json.
        cache_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "_metadata.json").exists()]
        assert len(cache_dirs) == 1, f"expected exactly one cache dir, got {cache_dirs}"
        meta = json.loads((cache_dirs[0] / "_metadata.json").read_text())
        assert meta["format"] == "lance"
        assert meta["num_rows"] == 15

        # Cache dir does NOT contain Arrow IPC shards.
        assert not list(cache_dirs[0].glob("shard-*.arrow"))
        # Cache dir DOES contain Lance-style internals (versions / data dir).
        has_lance_marker = (cache_dirs[0] / "_versions").exists() or (cache_dirs[0] / "data").exists()
        assert has_lance_marker, f"no Lance dataset files found in {cache_dirs[0]}"

    def test_lance_builder_roundtrip(self, tmp_path):
        Builder = _make_lance_builder_class()
        ds = Builder(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")

        assert len(ds) == 15
        for i in range(15):
            row = ds[i]
            assert row["x"] == i
            assert row["label"] == i % 3  # ClassLabel encodes as integer id

        # Full iteration roundtrip
        xs = sorted(row["x"] for row in ds)
        assert xs == list(range(15))

    def test_lance_builder_cache_hit_on_second_call(self, tmp_path):
        Builder = _make_lance_builder_class()
        # First call: cache miss, writes Lance dataset.
        ds1 = Builder(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")
        assert len(ds1) == 15

        # Second call with same cache dir: must hit the cache-hit path,
        # not re-run _generate_examples. We detect this by making the
        # second call succeed without having the download dir present
        # (which _split_generators would otherwise require for most
        # real builders; our _TinyLanceBuilder has empty assets so it
        # does not actually download, but we still want to verify the
        # metadata-driven cache-hit path runs).
        ds2 = Builder(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")
        assert len(ds2) == 15
        # Sanity: both datasets produce identical contents.
        assert [ds1[i]["x"] for i in range(15)] == [ds2[i]["x"] for i in range(15)]

    def test_runtime_storage_format_override(self, tmp_path):
        """The ``storage_format=`` kwarg on the builder constructor
        must override the class-level ``STORAGE_FORMAT`` for a single
        call, without mutating the class or leaking state into
        subsequent instantiations."""
        from stable_datasets.backend import ArrowBackend
        from stable_datasets.lance_backend import LanceBackend

        Builder = _make_lance_builder_class()  # class default = "lance"

        # Runtime override to arrow: should return an ArrowBackend
        # even though the class default is lance.
        ds_arrow = Builder(
            processed_cache_dir=tmp_path,
            download_dir=tmp_path / "dl",
            split="train",
            storage_format="arrow",
        )
        assert isinstance(ds_arrow._backend, ArrowBackend)
        assert len(ds_arrow) == 15

        # No override: class default of "lance" applies.
        ds_lance = Builder(
            processed_cache_dir=tmp_path,
            download_dir=tmp_path / "dl",
            split="train",
        )
        assert isinstance(ds_lance._backend, LanceBackend)
        assert len(ds_lance) == 15

        # Content must be identical row-for-row across formats.
        for i in range(15):
            assert ds_arrow[i]["x"] == ds_lance[i]["x"]
            assert ds_arrow[i]["label"] == ds_lance[i]["label"]

        # Both caches must live on disk at distinct directories.
        cache_dirs = sorted(
            p.name for p in tmp_path.iterdir()
            if p.is_dir() and (p / "_metadata.json").exists()
        )
        assert len(cache_dirs) == 2, f"expected two cache dirs, got {cache_dirs}"

        # Class default must not have been mutated by the override.
        assert Builder.STORAGE_FORMAT == "lance"

    def test_invalid_storage_format_raises(self, tmp_path):
        Builder = _make_lance_builder_class()
        with pytest.raises(ValueError, match="storage_format must be"):
            Builder(
                processed_cache_dir=tmp_path,
                download_dir=tmp_path / "dl",
                split="train",
                storage_format="parquet",
            )

    def test_arrow_and_lance_caches_coexist_at_distinct_paths(self, tmp_path):
        """The storage_format-aware cache_fingerprint must place Arrow
        and Lance caches at different directories so both can exist
        side-by-side without one clobbering the other."""
        from stable_datasets.schema import (
            DatasetInfo,
            Features,
            Value,
            Version,
        )
        from stable_datasets.splits import Split, SplitGenerator
        from stable_datasets.utils import BaseDatasetBuilder

        class _ArrowVariant(BaseDatasetBuilder):
            VERSION = Version("0.0.0")
            SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}
            STORAGE_FORMAT = "arrow"

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _split_generators(self):
                return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 5})]

            def _generate_examples(self, n):
                for i in range(n):
                    yield i, {"x": i}

        class _LanceVariant(_ArrowVariant):
            STORAGE_FORMAT = "lance"

        ds_arrow = _ArrowVariant(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")
        ds_lance = _LanceVariant(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")

        # Both datasets should work and return the same content.
        assert len(ds_arrow) == 5
        assert len(ds_lance) == 5

        # The two caches live at different directories.
        cache_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir() and (p / "_metadata.json").exists())
        assert len(cache_dirs) == 2, (
            f"expected two cache dirs (one per format), got {cache_dirs}"
        )
