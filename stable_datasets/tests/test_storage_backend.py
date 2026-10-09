"""Cross-backend tests for the StorageBackend protocol.

Every test in this file runs twice via a parameterized fixture: once
with :class:`ArrowBackend` backed by a sharded Arrow IPC cache, once
with :class:`LanceBackend` backed by a native Lance cache written via
``write_lance_cache`` from the same generator. Any test that passes
for Arrow and fails for Lance is evidence of a protocol leak -- some
place where :class:`StableDataset` or its consumers reach past the
abstraction and depend on Arrow-specific behavior.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.backends.arrow_shards import ArrowBackend
from stable_datasets.backends.lance_rows import LanceBackend
from stable_datasets.cache import write_lance_cache, write_sharded_arrow_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value


BackendKind = str  # "arrow" | "lance"


@pytest.fixture(params=["arrow", "lance"], ids=["arrow", "lance"])
def make_ds(request, tmp_path) -> Callable[..., StableDataset]:
    """Factory fixture: returns a callable that builds a file-backed
    :class:`StableDataset` on the parameterized backend.

    Both variants consume the same in-memory generator, so the test
    content is identical row-for-row between Arrow and Lance runs.
    The Lance variant writes through :func:`write_lance_cache`
    directly, matching the path :class:`BaseDatasetBuilder` uses when
    ``STORAGE_FORMAT="lance"``.
    """
    kind: BackendKind = request.param

    def _make(n: int = 10, batch_size: int = 5) -> StableDataset:
        features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
        info = DatasetInfo(features=features)

        def gen():
            for i in range(n):
                yield i, {"x": i, "label": i % 2}

        if kind == "arrow":
            cache_dir = tmp_path / f"arrow_cache_{n}"
            meta = write_sharded_arrow_cache(gen(), features, cache_dir, batch_size=batch_size)
            backend = ArrowBackend(
                shard_paths=meta.shard_paths,
                shard_row_counts=meta.shard_row_counts,
                schema=features.to_arrow_schema(),
            )
            num_rows = meta.num_rows
        elif kind == "lance":
            lance_dir = tmp_path / f"lance_cache_{n}"
            lance_meta = write_lance_cache(gen(), features, lance_dir, batch_size=batch_size)
            backend = LanceBackend(uri=lance_dir)
            num_rows = lance_meta.num_rows
        else:
            raise AssertionError(f"unknown backend {kind}")

        return StableDataset(features=features, info=info, backend=backend, num_rows=num_rows)

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


class TestBatchedRowAccess:
    def test_numpy_integer_index(self, make_ds):
        ds = make_ds(n=10)
        assert ds[np.int64(3)]["x"] == 3
        assert ds[np.int32(-1)]["x"] == 9
        assert ds.select([5, 1])[np.uint8(1)]["x"] == 1
        with pytest.raises(TypeError, match="Unsupported index type"):
            ds[1.0]

    def _count_reads(self, ds, monkeypatch):
        calls = {"get_row": 0, "take": 0}
        backend = ds._backend
        get_row, take = backend.get_row, backend.take

        def counting_get_row(idx):
            calls["get_row"] += 1
            return get_row(idx)

        def counting_take(indices):
            calls["take"] += 1
            return take(indices)

        monkeypatch.setattr(backend, "get_row", counting_get_row)
        monkeypatch.setattr(backend, "take", counting_take)
        return calls

    def test_indexed_view_iteration_reads_chunks(self, make_ds, monkeypatch):
        view = make_ds(n=2100).select(np.arange(2099, -1, -1))
        calls = self._count_reads(view, monkeypatch)
        assert [r["x"] for r in view] == list(range(2099, -1, -1))
        assert [r["x"] for r in view.iter_epoch()] == list(range(2099, -1, -1))
        assert calls == {"get_row": 0, "take": 6}

    def test_unbatched_filter_reads_chunks(self, make_ds, monkeypatch):
        view = make_ds(n=2100).select(np.arange(2099, -1, -1))
        calls = self._count_reads(view, monkeypatch)
        kept = view.filter(lambda row: row["x"] % 3 == 0)
        assert calls == {"get_row": 0, "take": 3}
        assert [r["x"] for r in kept][:3] == [2097, 2094, 2091]

    def test_unbatched_map_reads_chunks(self, make_ds, monkeypatch, tmp_path):
        view = make_ds(n=2100).select(np.arange(2099, -1, -1))
        calls = self._count_reads(view, monkeypatch)
        mapped = view.map(
            lambda row, i: {"x": row["x"] + i, "label": row["label"]}, with_indices=True, cache_dir=tmp_path / "mapped"
        )
        assert calls["get_row"] == 1  # one probe row infers the output features
        assert calls["take"] == 3
        assert {r["x"] for r in mapped} == {2099}


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
        from stable_datasets.backends.protocol import StorageBackend

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

    def test_take_single_shard_preserves_order_and_duplicates(self, make_ds):
        ds = make_ds(n=8, batch_size=16)
        sub = ds._backend.take([6, 2, 6, 1])
        assert sub.column("x").to_pylist() == [6, 2, 6, 1]

    def test_take_multi_shard_preserves_order_and_duplicates(self, make_ds):
        ds = make_ds(n=12, batch_size=4)
        sub = ds._backend.take([8, 1, 8, 5, 0, 11])
        assert sub.column("x").to_pylist() == [8, 1, 8, 5, 0, 11]

    def test_take_contiguous_matches_slice(self, make_ds):
        ds = make_ds(n=12, batch_size=4)
        taken = ds._backend.take([4, 5, 6])
        sliced = ds._backend.slice(4, 3)
        assert taken.column("x").to_pylist() == sliced.column("x").to_pylist()
        assert taken.column("label").to_pylist() == sliced.column("label").to_pylist()

    def test_slice(self, make_ds):
        ds = make_ds(n=20)
        sub = ds._backend.slice(5, 3)
        assert sub.num_rows == 3
        assert sub.column("x").to_pylist() == [5, 6, 7]

    def test_iter_batches_row_count(self, make_ds):
        ds = make_ds(n=23, batch_size=7)
        total = sum(b.num_rows for b in ds._backend.iter_batches())
        assert total == 23


# ── Shallow-copy + fork-safety ────────────────────────────────────────────────
#
# ``_shallow_copy`` must not trigger backend access for ``num_rows``.
# For Lance, the underlying dataset handle remains unopened until use.


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


# ── BaseDatasetBuilder with STORAGE_FORMAT="lance" ──────────────────────────
#
# Integration test for the direct Lance writer. A tiny builder subclass
# opts into Lance storage via ``STORAGE_FORMAT = "lance"``; the round
# trip exercises ``write_lance_cache``, the ``_metadata.json`` format
# marker, the cache_fingerprint split by storage_format, and the read
# path through ``BaseDatasetBuilder.__new__`` -> ``LanceBackend``.


class _TinyLanceBuilder:
    """Module-level fixture for Lance builder integration tests.

    Defined as a module-level class so the Python pickle machinery can
    locate it during ``pickle.dumps(ds)``. Actual BaseDatasetBuilder
    inheritance is added inside the test setup.
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
            return DatasetInfo(features=Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b", "c"])}))

        def _split_generators(self):
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 15})]

        def _iter_examples(self, n):
            for i in range(n):
                yield i, {"x": i, "label": i % 3}

    return _LanceBuilder


class TestBuilderStorageFormat:
    def test_lance_builder_writes_lance_cache(self, tmp_path):
        import json

        from stable_datasets.backends.lance_rows import LanceBackend

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

        # Second call with the same cache dir must hit the metadata-driven
        # cache path without running _iter_examples.
        ds2 = Builder(processed_cache_dir=tmp_path, download_dir=tmp_path / "dl", split="train")
        assert len(ds2) == 15
        # Sanity: both datasets produce identical contents.
        assert [ds1[i]["x"] for i in range(15)] == [ds2[i]["x"] for i in range(15)]

    def test_runtime_storage_format_override(self, tmp_path):
        """The ``storage_format=`` kwarg on the builder constructor
        must override the class-level ``STORAGE_FORMAT`` for a single
        call, without mutating the class or leaking state into
        subsequent instantiations."""
        from stable_datasets.backends.arrow_shards import ArrowBackend
        from stable_datasets.backends.lance_rows import LanceBackend

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
        cache_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir() and (p / "_metadata.json").exists())
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

            def _iter_examples(self, n):
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
        assert len(cache_dirs) == 2, f"expected two cache dirs (one per format), got {cache_dirs}"


class TestLanceVersionPinning:
    def _write_two_versions(self, tmp_path):
        import lance

        features = Features({"x": Value("int32")})
        write_lance_cache(((i, {"x": i}) for i in range(4)), features, tmp_path / "lance")
        lance.write_dataset(pa.table({"x": pa.array([7, 8], type=pa.int32())}), str(tmp_path / "lance"), mode="append")
        return features

    def test_open_cache_pins_metadata_version_without_opening(self, tmp_path):
        from stable_datasets.cache import open_cache

        features = self._write_two_versions(tmp_path)
        backend = open_cache(tmp_path / "lance", features).backend
        worker_copy = pickle.loads(pickle.dumps(backend))

        assert backend._ds is None
        assert worker_copy.num_rows == 4
        assert worker_copy.version == 1
        assert worker_copy.table.column("x").to_pylist() == [0, 1, 2, 3]

    def test_pickled_copy_reads_the_version_the_parent_opened(self, tmp_path):
        import lance

        features = Features({"x": Value("int32")})
        write_lance_cache(((i, {"x": i}) for i in range(4)), features, tmp_path / "lance")
        backend = LanceBackend(uri=tmp_path / "lance")
        assert backend.num_rows == 4
        lance.write_dataset(pa.table({"x": pa.array([7], type=pa.int32())}), str(tmp_path / "lance"), mode="append")

        worker_copy = pickle.loads(pickle.dumps(backend))
        assert worker_copy.version == backend.version == 1
        assert worker_copy.num_rows == worker_copy._dataset.count_rows() == 4

    def test_refresh_checks_out_latest_version(self, tmp_path):
        from stable_datasets.cache import open_cache

        features = self._write_two_versions(tmp_path)
        backend = open_cache(tmp_path / "lance", features).backend
        assert backend.num_rows == 4

        latest = open_cache(tmp_path / "lance", features, backend_kwargs={"version": 2})
        assert latest.num_rows == latest.backend.num_rows == 6

        backend.refresh()
        assert backend.version == 2
        assert backend.num_rows == 6
        assert backend.shard_row_counts == [4, 2]
        assert pickle.loads(pickle.dumps(backend)).version == 2
