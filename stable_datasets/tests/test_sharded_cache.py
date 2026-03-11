"""Tests for the sharded Arrow caching pipeline."""

import io
import json
import pickle

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.cache import (
    _encode_image,
    read_shard,
    read_sharded_cache_meta,
    validate_sharded_cache,
    write_sharded_arrow_cache,
)
from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    Features,
    Image,
    Value,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


# Helpers


def _simple_features():
    return Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})


def _simple_gen(n=100):
    for i in range(n):
        yield i, {"x": i, "label": i % 2}


def _image_features():
    return Features({"image": Image(), "label": Value("int32")})


def _image_gen(n=10):
    for i in range(n):
        img = PILImage.new("RGB", (4, 4), color=(i, i, i))
        yield i, {"image": img, "label": i}


def _write_shards(tmp_path, n=100, shard_size_bytes=1024, batch_size=10, features=None, gen=None):
    """Write a sharded cache and return the metadata."""
    if features is None:
        features = _simple_features()
    if gen is None:
        gen = _simple_gen(n)
    cache_dir = tmp_path / "sharded_cache"
    return write_sharded_arrow_cache(
        gen,
        features,
        cache_dir,
        shard_size_bytes=shard_size_bytes,
        batch_size=batch_size,
    )


# Sharded writer tests


class TestShardedWriter:
    def test_creates_directory_with_shards(self, tmp_path):
        meta = _write_shards(tmp_path, n=100, shard_size_bytes=1024)
        assert meta.cache_dir.is_dir()
        assert meta.num_shards >= 2, "Expected multiple shards with small shard_size_bytes"
        for fname in meta.shard_filenames:
            assert (meta.cache_dir / fname).exists()

    def test_metadata_file_exists_and_valid(self, tmp_path):
        meta = _write_shards(tmp_path, n=50)
        meta_path = meta.cache_dir / "_metadata.json"
        assert meta_path.exists()
        raw = json.loads(meta_path.read_text())
        assert raw["cache_format_version"] == 1
        assert raw["num_rows"] == 50
        assert raw["num_shards"] == meta.num_shards
        assert len(raw["shard_filenames"]) == meta.num_shards
        assert len(raw["shard_row_counts"]) == meta.num_shards

    def test_metadata_consistency(self, tmp_path):
        meta = _write_shards(tmp_path, n=100)
        assert meta.num_rows == 100
        assert sum(meta.shard_row_counts) == 100
        assert len(meta.shard_filenames) == meta.num_shards

    def test_single_shard_for_small_data(self, tmp_path):
        meta = _write_shards(tmp_path, n=5, shard_size_bytes=10 * 1024 * 1024)
        assert meta.num_shards == 1

    def test_empty_dataset(self, tmp_path):
        features = _simple_features()
        cache_dir = tmp_path / "empty_cache"
        meta = write_sharded_arrow_cache(iter([]), features, cache_dir)
        assert meta.num_rows == 0
        assert meta.num_shards == 0

    def test_empty_dataset_is_usable(self, tmp_path):
        features = _simple_features()
        info = DatasetInfo(features=features)
        cache_dir = tmp_path / "empty_ds"
        meta = write_sharded_arrow_cache(iter([]), features, cache_dir)
        ds = StableDataset(
            features=features,
            info=info,
            shard_dir=cache_dir,
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
            num_rows=meta.num_rows,
        )
        assert len(ds) == 0
        assert list(ds) == []
        with pytest.raises(IndexError):
            ds[0]
        assert ds.table.num_rows == 0

    def test_shard_data_readable(self, tmp_path):
        meta = _write_shards(tmp_path, n=20, shard_size_bytes=1024, batch_size=5)
        all_x = []
        for path in meta.shard_paths:
            tbl = read_shard(path)
            all_x.extend(tbl.column("x").to_pylist())
        assert sorted(all_x) == list(range(20))

    def test_atomic_publish_no_temp_dir_on_success(self, tmp_path):
        _write_shards(tmp_path, n=10)
        # No temp dirs should remain
        for child in tmp_path.iterdir():
            assert not child.name.startswith(".")

    def test_atomic_publish_cleans_up_on_failure(self, tmp_path):
        features = _simple_features()
        cache_dir = tmp_path / "fail_cache"

        def _bad_gen():
            yield 0, {"x": 0, "label": 0}
            raise RuntimeError("intentional failure")

        with pytest.raises(RuntimeError, match="intentional failure"):
            write_sharded_arrow_cache(_bad_gen(), features, cache_dir)

        # No temp dirs should remain
        for child in tmp_path.iterdir():
            assert not child.name.startswith(".")


# Image encoding tests


class TestImageEncoding:
    def test_jpeg_bytes_passthrough(self):
        # Create JPEG bytes
        img = PILImage.new("RGB", (4, 4), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        result = _encode_image(jpeg_bytes)
        assert result == jpeg_bytes
        assert result[:2] == b"\xff\xd8"  # JPEG magic

    def test_pil_jpeg_preserved(self, tmp_path):
        # Save a JPEG, reopen so .format is set
        img = PILImage.new("RGB", (4, 4), color=(128, 64, 32))
        path = tmp_path / "test.jpg"
        img.save(str(path), format="JPEG")
        reopened = PILImage.open(str(path))

        result = _encode_image(reopened)
        assert result[:2] == b"\xff\xd8"  # JPEG magic, not PNG

    def test_pil_png_for_rgba(self, tmp_path):
        # RGBA images should always use PNG even if source was JPEG
        img = PILImage.new("RGBA", (4, 4), color=(128, 64, 32, 255))
        result = _encode_image(img)
        assert result[:4] == b"\x89PNG"

    def test_numpy_array_to_png(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = _encode_image(arr)
        assert result[:4] == b"\x89PNG"

    def test_file_path_reads_raw_bytes(self, tmp_path):
        img = PILImage.new("RGB", (4, 4))
        path = tmp_path / "test.png"
        img.save(str(path))
        with open(path, "rb") as f:
            expected = f.read()
        result = _encode_image(str(path))
        assert result == expected

    def test_none_returns_none(self):
        assert _encode_image(None) is None


# Cache validation tests


class TestCacheValidation:
    def test_read_meta_valid(self, tmp_path):
        meta = _write_shards(tmp_path, n=20)
        loaded = read_sharded_cache_meta(meta.cache_dir)
        assert loaded.num_rows == 20
        assert loaded.num_shards == meta.num_shards
        assert loaded.shard_filenames == meta.shard_filenames

    def test_read_meta_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_sharded_cache_meta(tmp_path / "nonexistent")

    def test_validate_schema_mismatch(self, tmp_path):
        features1 = _simple_features()
        cache_dir = tmp_path / "cache"
        write_sharded_arrow_cache(_simple_gen(5), features1, cache_dir)

        features2 = Features({"y": Value("float64")})
        with pytest.raises(ValueError, match="Schema fingerprint mismatch"):
            validate_sharded_cache(cache_dir, features2)

    def test_validate_schema_match(self, tmp_path):
        features = _simple_features()
        cache_dir = tmp_path / "cache"
        write_sharded_arrow_cache(_simple_gen(5), features, cache_dir)
        meta = validate_sharded_cache(cache_dir, features)
        assert meta.num_rows == 5

    def test_missing_shard_file_detected(self, tmp_path):
        meta = _write_shards(tmp_path, n=50, shard_size_bytes=512)
        # Delete one shard
        (meta.cache_dir / meta.shard_filenames[-1]).unlink()
        with pytest.raises(ValueError, match="Missing shard file"):
            read_sharded_cache_meta(meta.cache_dir)


# Sharded StableDataset tests


def _make_sharded_ds(tmp_path, n=50, shard_size_bytes=512, batch_size=10):
    """Create a shard-backed StableDataset."""
    features = _simple_features()
    info = DatasetInfo(features=features)
    cache_dir = tmp_path / "ds_cache"
    meta = write_sharded_arrow_cache(
        _simple_gen(n),
        features,
        cache_dir,
        shard_size_bytes=shard_size_bytes,
        batch_size=batch_size,
    )
    return StableDataset(
        features=features,
        info=info,
        shard_dir=cache_dir,
        shard_paths=meta.shard_paths,
        shard_row_counts=meta.shard_row_counts,
        num_rows=meta.num_rows,
    ), meta


class TestShardedDataset:
    def test_len(self, tmp_path):
        ds, meta = _make_sharded_ds(tmp_path, n=50)
        assert len(ds) == 50

    def test_getitem_correctness(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=50)
        for i in range(50):
            row = ds[i]
            assert row["x"] == i
            assert row["label"] == i % 2

    def test_getitem_negative_index(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=20)
        assert ds[-1]["x"] == 19
        assert ds[-20]["x"] == 0

    def test_getitem_out_of_range(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=10)
        with pytest.raises(IndexError):
            ds[10]

    def test_getitem_loads_single_shard(self, tmp_path):
        ds, meta = _make_sharded_ds(tmp_path, n=200, shard_size_bytes=128, batch_size=5)
        assert meta.num_shards >= 2
        # Access first row (shard 0)
        ds[0]
        assert 0 in ds._shard_lru
        # Initially only shard 0 should be loaded
        assert len(ds._shard_lru) == 1

    def test_iter_yields_all(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=30)
        rows = list(ds)
        assert len(rows) == 30
        assert [r["x"] for r in rows] == list(range(30))

    def test_iter_epoch_shuffled(self, tmp_path):
        ds, meta = _make_sharded_ds(tmp_path, n=200, shard_size_bytes=128, batch_size=5)
        assert meta.num_shards >= 4
        sequential = [r["x"] for r in ds]
        shuffled = [r["x"] for r in ds.iter_epoch(shuffle_shards=True, seed=42)]
        # Same elements
        assert sorted(shuffled) == sorted(sequential)
        # Different order (shard boundaries reordered)
        assert shuffled != sequential

    def test_slice(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=20)
        sub = ds[2:5]
        assert isinstance(sub, StableDataset)
        assert len(sub) == 3
        assert sub[0]["x"] == 2
        assert sub[1]["x"] == 3
        assert sub[2]["x"] == 4

    def test_pickle_size_is_small(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=50)
        _ = ds[0]  # trigger shard load
        data = pickle.dumps(ds)
        assert len(data) < 4096

    def test_pickle_roundtrip(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=20)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert ds2._table is None
        assert ds2._shard_paths is not None
        assert len(ds2) == 20
        for i in range(20):
            assert ds2[i]["x"] == i

    def test_pickle_preserves_features(self, tmp_path):
        ds, _ = _make_sharded_ds(tmp_path, n=5)
        ds2 = pickle.loads(pickle.dumps(ds))
        assert isinstance(ds2.features["label"], ClassLabel)
        assert ds2.features["label"].names == ["a", "b"]

    def test_lru_eviction(self, tmp_path):
        ds, meta = _make_sharded_ds(tmp_path, n=200, shard_size_bytes=128, batch_size=5)
        assert meta.num_shards > 4  # Need more shards than LRU size
        # Access a row in each shard
        for i in range(meta.num_shards):
            offset = sum(meta.shard_row_counts[:i])
            if offset < 100:
                ds[offset]
        # LRU should have capped at max_open_shards (default 4)
        assert len(ds._shard_lru) <= 4


# Builder integration tests


class _TinyShardedBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 20})]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class TestBuilderShardedIntegration:
    def test_builder_creates_sharded_cache(self, tmp_path):
        ds = _TinyShardedBuilder(split="train", processed_cache_dir=str(tmp_path))
        assert isinstance(ds, StableDataset)
        assert ds._is_shard_backed
        assert len(ds) == 20
        for i in range(20):
            assert ds[i]["x"] == i

    def test_builder_warm_cache_skips_rebuild(self, tmp_path):
        _TinyShardedBuilder(split="train", processed_cache_dir=str(tmp_path))
        # Find the shard dir
        shard_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(shard_dirs) == 1
        mtimes = {f: f.stat().st_mtime for f in shard_dirs[0].iterdir()}

        ds2 = _TinyShardedBuilder(split="train", processed_cache_dir=str(tmp_path))
        for f, mtime in mtimes.items():
            assert f.stat().st_mtime == mtime
        assert len(ds2) == 20
