"""Tests for StableDataset lazy-mmap and pickle behaviour."""

import pickle

import pytest

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.cache import write_sharded_arrow_cache
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value, Version
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

    def test_in_memory_slice_pickle_includes_table(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        sub = ds[0:3]
        assert sub._shard_paths is None
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
    def test_slice_returns_in_memory_dataset(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        sub = ds[2:5]
        assert isinstance(sub, StableDataset)
        assert sub._shard_paths is None
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
