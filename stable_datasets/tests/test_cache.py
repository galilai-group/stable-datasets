"""Tests for the Lance caching pipeline: the writer, cache metadata, schema checks and locking."""

import io
import json

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.cache import (
    DEFAULT_SHARD_SIZE_BYTES,
    open_cache,
    read_lance_cache_meta,
    write_lance_cache,
    write_lance_table,
)
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    Features,
    Image,
    Value,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.tests._caches import cached_dataset
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


class TestLanceWriter:
    def test_writes_dataset_and_metadata(self, tmp_path):
        meta = write_lance_cache(_simple_gen(20), _simple_features(), tmp_path / "c")
        assert (tmp_path / "c" / "_versions").is_dir()
        raw = json.loads((tmp_path / "c" / "_metadata.json").read_text())
        assert raw["format"] == "lance" and raw["layout"] == "lance-rows"
        assert raw["num_rows"] == meta.num_rows == 20 == sum(raw["shard_row_counts"])

    def test_single_fragment_for_small_data(self, tmp_path):
        assert len(write_lance_cache(_simple_gen(5), _simple_features(), tmp_path / "c").shard_row_counts) == 1

    def test_fragments_follow_max_rows_per_file(self, tmp_path):
        meta = write_lance_cache(
            _simple_gen(50), _simple_features(), tmp_path / "c", batch_size=10, max_rows_per_file=10
        )
        assert meta.shard_row_counts == [10] * 5

    def test_empty_dataset_is_usable(self, tmp_path):
        features = _simple_features()
        write_lance_cache(iter([]), features, tmp_path / "empty")
        opened = open_cache(tmp_path / "empty", features)
        ds = StableDataset(features=features, info=DatasetInfo(features=features), backend=opened.backend, num_rows=0)
        assert len(ds) == 0
        assert list(ds) == []
        with pytest.raises(IndexError):
            ds[0]

    def test_rewrite_replaces_rows(self, tmp_path):
        features = _simple_features()
        write_lance_cache(_simple_gen(40), features, tmp_path / "c")
        write_lance_cache(_simple_gen(4), features, tmp_path / "c")
        assert open_cache(tmp_path / "c", features).num_rows == 4

    def test_rewrite_over_corrupt_metadata(self, tmp_path):
        features = _simple_features()
        write_lance_cache(_simple_gen(5), features, tmp_path / "c")
        (tmp_path / "c" / "_metadata.json").write_text("{")
        write_lance_cache(_simple_gen(3), features, tmp_path / "c")
        assert open_cache(tmp_path / "c", features).num_rows == 3

    def test_parallel_encode_matches_serial(self, tmp_path):
        features = _image_features()
        serial = cached_dataset(_image_gen(30), features, tmp_path / "serial", batch_size=10)
        write_lance_cache(_image_gen(30), features, tmp_path / "parallel", batch_size=10, num_encode_workers=2)
        parallel = open_cache(tmp_path / "parallel", features).backend
        assert [serial._backend.get_row(i) for i in range(30)] == [parallel.get_row(i) for i in range(30)]

    def test_write_lance_table_keeps_encoded_rows(self, tmp_path):
        features = _image_features()
        ds = cached_dataset(_image_gen(6), features, tmp_path / "src")
        write_lance_table(ds._backend.take([5, 1]), features, tmp_path / "copy")
        copy = open_cache(tmp_path / "copy", features).backend
        assert copy.get_row(0) == ds._backend.get_row(5) and copy.get_row(1) == ds._backend.get_row(1)


class TestImageEncoding:
    def test_jpeg_bytes_passthrough(self):
        # Create JPEG bytes
        img = PILImage.new("RGB", (4, 4), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        result = Image().encode(jpeg_bytes)
        assert result == jpeg_bytes
        assert result[:2] == b"\xff\xd8"  # JPEG magic

    def test_pil_jpeg_preserved(self, tmp_path):
        # Save a JPEG, reopen so .format is set
        img = PILImage.new("RGB", (4, 4), color=(128, 64, 32))
        path = tmp_path / "test.jpg"
        img.save(str(path), format="JPEG")
        reopened = PILImage.open(str(path))

        result = Image().encode(reopened)
        assert result[:2] == b"\xff\xd8"  # JPEG magic, not PNG

    def test_pil_png_for_rgba(self, tmp_path):
        # RGBA images should always use PNG even if source was JPEG
        img = PILImage.new("RGBA", (4, 4), color=(128, 64, 32, 255))
        result = Image().encode(img)
        assert result[:4] == b"\x89PNG"

    def test_numpy_array_to_png(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = Image().encode(arr)
        assert result[:4] == b"\x89PNG"

    def test_file_path_reads_raw_bytes(self, tmp_path):
        img = PILImage.new("RGB", (4, 4))
        path = tmp_path / "test.png"
        img.save(str(path))
        with open(path, "rb") as f:
            expected = f.read()
        result = Image().encode(str(path))
        assert result == expected

    def test_none_returns_none(self):
        assert Image().encode(None) is None


class TestCacheMetadata:
    def test_read_meta_valid(self, tmp_path):
        write_lance_cache(_simple_gen(20), _simple_features(), tmp_path / "c")
        assert read_lance_cache_meta(tmp_path / "c").num_rows == 20

    def test_read_meta_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_lance_cache_meta(tmp_path / "nonexistent")


class _TinyBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 20})]

    def _iter_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class TestBuilderIntegration:
    def test_builder_creates_lance_cache(self, tmp_path):
        ds = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        assert isinstance(ds, StableDataset)
        assert ds._backend.is_file_backed
        assert [ds[i]["x"] for i in range(20)] == list(range(20))

    def test_builder_warm_cache_skips_rebuild(self, tmp_path):
        _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        (cache_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
        mtimes = {f: f.stat().st_mtime for f in cache_dir.rglob("*") if f.is_file()}
        ds2 = _TinyBuilder(split="train", processed_cache_dir=str(tmp_path))
        assert all(f.stat().st_mtime == m for f, m in mtimes.items())
        assert len(ds2) == 20

    def test_fragment_target_is_finite(self):
        assert DEFAULT_SHARD_SIZE_BYTES == 256 * 1024 * 1024


class _SchemaBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}
    names = [f"class_{i}" for i in range(6)]

    def _info(self):
        return DatasetInfo(features=Features({"y": ClassLabel(names=self.names)}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={})]

    def _iter_examples(self):
        for i, name in enumerate(self.names):
            yield i, {"y": name}


class TestSchemaChanges:
    def test_feature_change_rebuilds_cache(self, tmp_path, monkeypatch):
        first = _SchemaBuilder(split="train", processed_cache_dir=tmp_path)
        monkeypatch.setattr(_SchemaBuilder, "names", [*_SchemaBuilder.names[:-1], "renamed"])
        second = _SchemaBuilder(split="train", processed_cache_dir=tmp_path)

        assert first.features["y"].names[-1] == "class_5"
        assert second.features["y"].names[-1] == "renamed"
        assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 2

    def test_lance_cache_rejects_other_features(self, tmp_path):
        from stable_datasets.cache import open_cache, write_lance_cache

        written = Features({"y": ClassLabel(names=[f"c{i}" for i in range(6)])})
        write_lance_cache(((i, {"y": i}) for i in range(3)), written, tmp_path / "lance")
        other = Features({"y": ClassLabel(names=[f"d{i}" for i in range(6)])})

        with pytest.raises(ValueError, match="Schema fingerprint mismatch"):
            open_cache(tmp_path / "lance", other)
        assert open_cache(tmp_path / "lance", written).num_rows == 3

    def test_video_frames_cache_rejects_other_features(self, tmp_path, monkeypatch):
        pytest.importorskip("lance")
        import stable_datasets.cache as cache
        from stable_datasets.cache import open_cache, write_lance_video_frames_cache
        from stable_datasets.schema import Video

        monkeypatch.setattr(
            cache, "_encode_one_video_frames", lambda task: ("ok", task[0], task[1], 1, 4, 4, [b"frame"])
        )
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")
        written = Features({"video": Video(storage="frames"), "label": Value("int32")})
        write_lance_video_frames_cache(
            iter([(0, {"video": video, "label": 1})]), written, tmp_path / "frames", workers=1
        )

        with pytest.raises(ValueError, match="Schema fingerprint mismatch"):
            open_cache(tmp_path / "frames", Features({"video": Video(storage="frames"), "label": Value("int64")}))
        assert open_cache(tmp_path / "frames", written).num_rows == 1

        meta_path = tmp_path / "frames" / "_metadata.json"
        meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "cache_format_version": 0}))
        with pytest.raises(ValueError, match="Unsupported cache format version 0"):
            open_cache(tmp_path / "frames", written)

    def test_flatten_indices_output_opens_with_open_cache(self, tmp_path):
        from stable_datasets.cache import open_cache

        ds = cached_dataset(_simple_gen(10), _simple_features(), tmp_path / "cache")
        flat = ds.select([4, 2, 0]).flatten_indices(cache_dir=tmp_path / "flat")
        opened = open_cache(tmp_path / "flat", flat.features)
        assert opened.num_rows == 3
        assert opened.backend.get_row(0)["x"] == ds[4]["x"]


class TestLocking:
    def test_lock_name_keeps_dotted_config_names(self, tmp_path):
        from stable_datasets.cache import _writer_lock

        a = _writer_lock(str(tmp_path / "cifar_v1.0_train_abc")).lock_file
        b = _writer_lock(str(tmp_path / "cifar_v1.5_train_abc")).lock_file
        assert a == str(tmp_path / "cifar_v1.0_train_abc.lock")
        assert a != b


def test_download_keeps_lock_file(tmp_path):
    from unittest.mock import patch

    from stable_datasets.utils import download

    class _Response:
        status_code = 200
        headers = {"content-length": "4"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([b"data"])

        class raw:  # download() streams the undecoded bytes
            @staticmethod
            def stream(chunk_size=8192, decode_content=True):
                return iter([b"data"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class _Session:
        headers = {}

        def get(self, url, **kwargs):
            return _Response()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("stable_datasets.utils.requests.Session", return_value=_Session()):
        dest = download("https://example.com/f.bin", dest_folder=tmp_path, progress_bar=False)
    assert dest.read_bytes() == b"data"
    assert dest.with_suffix(dest.suffix + ".lock").exists()
