"""Tests for StableDataset lazy-mmap and pickle behaviour."""

import pickle

import pytest

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.cache import write_arrow_cache
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_arrow_file(tmp_path, name="test.arrow", n=10):
    """Write a small Arrow IPC file and return (path, features, info, num_rows)."""
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    path = tmp_path / name
    table = write_arrow_cache(gen(), features, path)
    return path, features, info, table.num_rows


def _make_ds(tmp_path, **kw):
    """Shorthand: create an Arrow file and return a file-backed StableDataset."""
    path, features, info, n = _make_arrow_file(tmp_path, **kw)
    return StableDataset(path=path, features=features, info=info, num_rows=n)


class _TinyBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self, dl_manager=None):
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

    def test_getitem_triggers_mmap_and_returns_correct_values(self, tmp_path):
        ds = _make_ds(tmp_path, n=5)
        assert ds._table is None
        for i in range(5):
            assert ds[i]["x"] == i
        assert ds._table is not None

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

    def test_len_without_num_rows_triggers_mmap(self, tmp_path):
        path, features, info, _ = _make_arrow_file(tmp_path, n=7)
        ds = StableDataset(path=path, features=features, info=info)
        assert ds._table is None
        assert len(ds) == 7
        assert ds._table is not None


# ── Pickle / DataLoader compatibility ────────────────────────────────────────


class TestPickle:
    def test_file_backed_pickle_excludes_table(self, tmp_path):
        ds = _make_ds(tmp_path)
        _ = ds[0]  # load table into memory
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
        assert sub._path is None
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
        assert sub._path is None
        assert len(sub) == 3
        assert sub[0]["x"] == 2

    def test_train_test_split_is_disjoint_and_complete(self, tmp_path):
        ds = _make_ds(tmp_path, n=10)
        splits = ds.train_test_split(test_size=0.3, seed=42)
        assert splits["train"]._path is None
        assert len(splits["train"]) + len(splits["test"]) == 10
        train_xs = {splits["train"][i]["x"] for i in range(len(splits["train"]))}
        test_xs = {splits["test"][i]["x"] for i in range(len(splits["test"]))}
        assert train_xs & test_xs == set()
        assert train_xs | test_xs == set(range(10))


# ── Integration: end-to-end through BaseDatasetBuilder ───────────────────────


class TestBuilderIntegration:
    def test_builder_produces_lazy_file_backed_dataset(self, tmp_path):
        ds = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        assert isinstance(ds, StableDataset)
        assert ds._path is not None and ds._path.exists()
        assert ds._table is None
        assert len(ds) == 5
        assert len(pickle.dumps(ds)) < 4096

    def test_builder_warm_cache_skips_rebuild(self, tmp_path):
        _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        mtimes = {f: f.stat().st_mtime for f in tmp_path.glob("*.arrow")}
        ds2 = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        for f, mtime in mtimes.items():
            assert f.stat().st_mtime == mtime
        assert len(ds2) == 5
