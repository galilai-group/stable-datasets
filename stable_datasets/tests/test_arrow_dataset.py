"""Tests for StableDataset lazy-mmap and pickle behaviour."""

import io
import pickle

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.cache import write_sharded_arrow_cache
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_sharded_cache(tmp_path, name="cache", n=10):
    """Write a small sharded Arrow cache and return (meta, features, info)."""
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    cache_dir = tmp_path / name
    meta = write_sharded_arrow_cache(gen(), features, cache_dir, batch_size=5)
    return meta, features, info


def _make_ds(tmp_path, **kw):
    """Shorthand: create a sharded cache and return a shard-backed StableDataset."""
    meta, features, info = _make_sharded_cache(tmp_path, **kw)
    return StableDataset(
        features=features,
        info=info,
        shard_dir=meta.cache_dir,
        shard_paths=meta.shard_paths,
        shard_row_counts=meta.shard_row_counts,
        num_rows=meta.num_rows,
    )


class _TinyBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 5})]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


# ── Lazy mmap ────────────────────────────────────────────────────────────────


class TestLazyMmap:
    def test_init_and_len_do_not_load_table(self, tmp_path):
        ds = _make_ds(tmp_path)
        assert ds._table is None
        assert len(ds) == 10
        assert ds._table is None

    def test_getitem_returns_correct_values(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        for i in range(5):
            assert ds[i]["x"] == i

    def test_negative_indexing(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        assert ds[-1]["x"] == 4
        assert ds[-5]["x"] == 0

    def test_index_out_of_range(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        with pytest.raises(IndexError):
            ds[5]
        with pytest.raises(IndexError):
            ds[-6]


# ── Pickle / DataLoader compatibility ────────────────────────────────────────


class TestPickle:
    def test_shard_backed_pickle_excludes_table(self, tmp_path):
        ds = _make_ds(tmp_path)
        _ = ds[0]  # trigger shard load
        state = ds.__getstate__()
        assert "table" not in state
        assert len(pickle.dumps(ds)) < 4096

    def test_unpickled_dataset_is_lazy_and_reads_correctly(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert ds2._table is None
        assert len(ds2) == 5
        assert ds2._table is None  # len uses cached num_rows
        for i in range(5):
            assert ds2[i]["x"] == i

    def test_indexed_slice_pickle_roundtrip(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        sub = ds[0:3]
        assert sub._indices is not None
        sub2 = pickle.loads(pickle.dumps(sub))
        assert len(sub2) == 3
        assert sub2[0]["x"] == 0

    def test_pickle_roundtrip_preserves_features(self, tmp_path):
        ds = _make_ds(tmp_path)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert isinstance(ds2.features["label"], ClassLabel)
        assert ds2.features["label"].names == ["a", "b"]


# ── Slice and train_test_split ───────────────────────────────────────────────


class TestSliceAndSplit:
    def test_slice_returns_indexed_view(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds[2:5]
        assert isinstance(sub, StableDataset)
        assert sub._indices is not None
        assert sub._table is None  # no materialization
        assert len(sub) == 3
        assert sub[0]["x"] == 2

    def test_train_test_split_is_disjoint_and_complete(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        splits = ds.train_test_split(test_size=0.3, seed=42)
        assert len(splits["train"]) + len(splits["test"]) == 10
        train_xs = {splits["train"][i]["x"] for i in range(len(splits["train"]))}
        test_xs = {splits["test"][i]["x"] for i in range(len(splits["test"]))}
        assert train_xs & test_xs == set()
        assert train_xs | test_xs == set(range(10))


# ── Iteration ────────────────────────────────────────────────────────────────


class TestIteration:
    def test_iter_yields_all_rows(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        rows = list(ds)
        assert len(rows) == 10
        assert [r["x"] for r in rows] == list(range(10))

    def test_iter_epoch_shuffled(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sequential = [r["x"] for r in ds]
        shuffled = [r["x"] for r in ds.iter_epoch(shuffle_shards=True, seed=42)]
        assert sorted(shuffled) == sorted(sequential)


# ── Integration: end-to-end through BaseDatasetBuilder ───────────────────────


class TestBuilderIntegration:
    def test_builder_produces_shard_backed_dataset(self, tmp_path):
        ds = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        assert isinstance(ds, StableDataset)
        assert ds._is_shard_backed
        assert ds._table is None
        assert len(ds) == 5
        assert len(pickle.dumps(ds)) < 4096

    def test_builder_warm_cache_skips_rebuild(self, tmp_path):
        _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        shard_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(shard_dirs) == 1
        mtimes = {f: f.stat().st_mtime for f in shard_dirs[0].iterdir()}
        ds2 = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        for f, mtime in mtimes.items():
            assert f.stat().st_mtime == mtime
        assert len(ds2) == 5


# ── Phase 1: Index Indirection ────────────────────────────────────────────────


class TestSelect:
    def test_select_returns_correct_rows(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds.select([3, 1, 4])
        assert len(sub) == 3
        assert sub[0]["x"] == 3
        assert sub[1]["x"] == 1
        assert sub[2]["x"] == 4

    def test_shuffle_preserves_all_rows(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        shuffled = ds.shuffle(seed=42)
        assert len(shuffled) == 10
        assert sorted(shuffled[i]["x"] for i in range(10)) == list(range(10))

    def test_filter_returns_matching_rows(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        evens = ds.filter(lambda row: row["x"] % 2 == 0)
        assert len(evens) == 5
        assert all(evens[i]["x"] % 2 == 0 for i in range(len(evens)))

    def test_train_test_split_no_materialization(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        splits = ds.train_test_split(test_size=0.3, seed=42)
        # Both splits should be indexed views, not materialized tables
        assert splits["train"]._table is None
        assert splits["test"]._table is None
        assert splits["train"]._indices is not None
        assert splits["test"]._indices is not None
        assert len(splits["train"]) + len(splits["test"]) == 10

    def test_indices_compose_through_select(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub1 = ds.select([3, 1, 4, 1, 5])
        sub2 = sub1.select([0, 2])
        assert len(sub2) == 2
        assert sub2[0]["x"] == 3
        assert sub2[1]["x"] == 4

    def test_indices_pickle_roundtrip(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds.select([3, 1, 4])
        sub2 = pickle.loads(pickle.dumps(sub))
        assert sub2._indices is not None
        assert len(sub2) == 3
        assert sub2[0]["x"] == 3
        assert sub2[1]["x"] == 1
        assert sub2[2]["x"] == 4

    def test_iter_epoch_with_indices(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds.select([3, 1, 4])
        rows = list(sub.iter_epoch(shuffle_shards=False))
        assert len(rows) == 3
        assert sorted(r["x"] for r in rows) == [1, 3, 4]

    def test_slice_returns_indexed_not_materialized(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds[2:5]
        assert sub._indices is not None
        assert sub._table is None
        assert len(sub) == 3
        assert sub[0]["x"] == 2


class TestColumnNamesAndNumRows:
    def test_column_names(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        assert ds.column_names == ["x", "label"]

    def test_num_rows(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        assert ds.num_rows == 5


# ── Phase 4: Format Control + Transform Pipeline ─────────────────────────────


def _make_image_cache(tmp_path, name="img_cache", n=5):
    """Write a small image sharded cache."""
    features = Features({"image": Image(), "label": Value("int32")})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            img = PILImage.new("RGB", (4, 4), color=(i * 10, i * 10, i * 10))
            yield i, {"image": img, "label": i}

    cache_dir = tmp_path / name
    meta = write_sharded_arrow_cache(gen(), features, cache_dir, batch_size=5)
    return StableDataset(
        features=features,
        info=info,
        shard_dir=meta.cache_dir,
        shard_paths=meta.shard_paths,
        shard_row_counts=meta.shard_row_counts,
        num_rows=meta.num_rows,
    )


class TestFormatAndTransform:
    def test_with_format_numpy_returns_arrays(self, tmp_path):
        ds = _make_image_cache(tmp_path)
        ds_np = ds.with_format("numpy")
        row = ds_np[0]
        assert isinstance(row["image"], np.ndarray)
        assert row["image"].shape == (4, 4, 3)

    def test_with_format_raw_skips_pil_decode(self, tmp_path):
        ds = _make_image_cache(tmp_path)
        ds_raw = ds.with_format("raw")
        row = ds_raw[0]
        assert isinstance(row["image"], bytes)

    def test_with_transform_applied(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)

        def add_one(row):
            row["x"] = row["x"] + 1
            return row

        ds_t = ds.with_transform(add_one)
        assert ds_t[0]["x"] == 1
        assert ds_t[4]["x"] == 5

    def test_format_preserved_through_select(self, tmp_path):
        ds = _make_image_cache(tmp_path).with_format("numpy")
        sub = ds.select([0, 2])
        row = sub[0]
        assert isinstance(row["image"], np.ndarray)

    def test_format_preserved_through_shuffle(self, tmp_path):
        ds = _make_image_cache(tmp_path).with_format("numpy")
        shuffled = ds.shuffle(seed=42)
        row = shuffled[0]
        assert isinstance(row["image"], np.ndarray)

    def test_with_format_torch_returns_tensors(self, tmp_path):
        torch = pytest.importorskip("torch")
        ds = _make_image_cache(tmp_path)
        ds_t = ds.with_format("torch")
        row = ds_t[0]
        assert isinstance(row["image"], torch.Tensor)
        assert row["image"].shape == (3, 4, 4)  # CHW
        assert isinstance(row["label"], torch.Tensor)
