import json
import subprocess
import sys
from dataclasses import dataclass

import pyarrow as pa
import pytest

from stable_datasets.backends import TableBackend
from stable_datasets.cache import _METADATA_FILE, CacheOpenResult, LanceCacheMeta, cache_fingerprint, open_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value, Version, Video
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.storage import (
    RowStorage,
    Storage,
    VideoFramesStorage,
    VideoStorage,
    layout_class,
    register_layout,
    storage_of,
)
from stable_datasets.storage.base import _LAYOUTS
from stable_datasets.utils import BaseDatasetBuilder


ROWS = Features({"x": Value("float32"), "y": ClassLabel(names=["a", "b"])})


def _video_features(video):
    return Features({"video": video, "label": Value("int32")})


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (ROWS, "tiny_default_train_7a96377f38dd92f1"),
        (_video_features(Video(storage="frames")), "tiny_default_train_9e6da54a52a5e80d"),
        (
            _video_features(Video(storage="mp4", settings=VideoStorage(fps=7.5, short_edge=96))),
            "tiny_default_train_48ce8c06463fde3e",
        ),
        (
            _video_features(Video(storage="jpeg", settings=VideoStorage(format="jpeg", quality=85))),
            "tiny_default_train_06ee695734a7ad6c",
        ),
    ],
)
def test_cache_keys_are_unchanged(features, expected):
    assert cache_fingerprint("Tiny", "1.0.0", "default", "train", features=features) == expected


def test_builtin_layouts_resolve_by_name():
    assert layout_class("lance-rows") is RowStorage
    assert layout_class("lance-video-frames") is VideoFramesStorage
    assert layout_class("framestore-video") is VideoStorage
    with pytest.raises(KeyError):
        layout_class("no-such-layout")


def test_coerce_aliases():
    assert Storage.coerce("rows") == RowStorage()
    assert Storage.coerce("frames") == VideoFramesStorage()
    assert Storage.coerce("mp4") == VideoStorage(format="mp4")
    assert Storage.coerce("jpeg") == VideoStorage(format="jpeg")
    vs = VideoStorage(fps=7.5)
    assert Storage.coerce(vs) is vs
    with pytest.raises(ValueError, match="Unknown storage"):
        Storage.coerce("webm")
    with pytest.raises(TypeError):
        Storage.coerce({"format": "mp4"})


def test_storage_of_reads_what_features_declare():
    assert storage_of(ROWS) == RowStorage()
    assert storage_of(_video_features(Video())) == RowStorage()
    assert storage_of(_video_features(Video(storage="frames"))) == VideoFramesStorage()
    vs = VideoStorage(format="jpeg", quality=80)
    assert storage_of(_video_features(Video(storage="jpeg", settings=vs))) == vs


def test_apply_declares_the_storage_in_the_features():
    features = _video_features(Video(allowed_extensions=(".mp4", ".avi")))
    vs = VideoStorage(fps=7.5)
    stored = vs.apply(features)
    assert stored["video"].storage == "mp4"
    assert stored["video"].settings == vs
    assert stored["video"].allowed_extensions == (".mp4", ".avi")
    assert stored["label"] == features["label"]
    assert storage_of(stored) == vs
    assert storage_of(VideoFramesStorage().apply(features)) == VideoFramesStorage()
    assert RowStorage().apply(features) is features


def test_loader_needs_a_layout_with_one():
    ds = StableDataset(features=ROWS, info=DatasetInfo(features=ROWS), table=pa.table({"x": [1.0], "y": [0]}))
    with pytest.raises(NotImplementedError, match="RowStorage has no batch loader"):
        ds.loader(4)


def test_register_layout_rejects_a_second_class_for_one_name():
    @dataclass(frozen=True)
    class Other(Storage):
        layout = "lance-rows"

    with pytest.raises(ValueError, match="already registered"):
        register_layout(Other)

    class Unnamed(Storage):
        pass

    with pytest.raises(TypeError, match="layout"):
        register_layout(Unnamed)


@pytest.mark.parametrize(
    "code",
    [
        "import stable_datasets.storage.video",
        "import stable_datasets.storage",
        "from stable_datasets.features import Video, VideoStorage",
        "from stable_datasets.storage import RowStorage, VideoFramesStorage",
    ],
)
def test_imports_in_any_order(code):
    subprocess.run([sys.executable, "-c", code], check=True)


@dataclass(frozen=True)
class JsonRows(Storage):
    """A layout outside the library: the rows as one JSON file."""

    layout = "test-json-rows"
    aliases = {"json-rows": {}}
    priority = -1

    def write(self, generator, features, cache_dir, *, lineage=None, storage_options=None):
        rows = [example for _key, example in generator]
        path = _layout_dir(cache_dir)
        (path / "rows.json").write_text(json.dumps(rows))
        meta = {"layout": self.layout, "num_rows": len(rows)}
        (path / _METADATA_FILE).write_text(json.dumps(meta))
        return LanceCacheMeta(cache_dir=cache_dir, num_rows=len(rows), schema_fingerprint="")

    @classmethod
    def open(cls, cache_dir, features, metadata, *, backend_kwargs, storage_options=None):
        rows = json.loads((_layout_dir(cache_dir) / "rows.json").read_text())
        table = pa.Table.from_pylist(rows, schema=features.to_arrow_schema())
        backend = TableBackend(table, schema=features.to_arrow_schema())
        return CacheOpenResult(backend=backend, num_rows=metadata["num_rows"], layout=cls.layout, metadata=metadata)


def _layout_dir(cache_dir):
    from pathlib import Path

    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _Tiny(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 5})]

    def _iter_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


@pytest.fixture
def json_rows():
    register_layout(JsonRows)
    yield JsonRows
    _LAYOUTS.pop(JsonRows.layout)


class _JsonTiny(_Tiny):
    STORAGE = "json-rows"


class _DeclaresJsonRows(JsonRows):
    """JsonRows chosen through the features: storage_of asks it before RowStorage."""

    layout = "test-json-rows-declared"
    priority = 5

    @classmethod
    def from_features(cls, features):
        return cls() if "json_x" in features else None


def test_a_registered_layout_is_written_and_opened_by_the_builder(tmp_path, json_rows):
    ds = _Tiny(split="train", processed_cache_dir=tmp_path, storage=JsonRows())
    assert [ds[i]["x"] for i in range(5)] == list(range(5))
    (cache_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert json.loads((cache_dir / _METADATA_FILE).read_text())["layout"] == "test-json-rows"
    reopened = open_cache(cache_dir, Features({"x": Value("int32")}))
    assert isinstance(reopened.backend, TableBackend) and reopened.num_rows == 5


def test_builder_storage_attribute_takes_an_alias(tmp_path, json_rows):
    ds = _JsonTiny(split="train", processed_cache_dir=tmp_path)
    assert len(ds) == 5
    (cache_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert (cache_dir / "rows.json").exists()


def test_storage_of_asks_layouts_by_priority(tmp_path, json_rows):
    register_layout(_DeclaresJsonRows)
    try:
        assert storage_of(Features({"json_x": Value("int32")})) == _DeclaresJsonRows()
        assert storage_of(ROWS) == RowStorage()
    finally:
        _LAYOUTS.pop(_DeclaresJsonRows.layout)


def test_unknown_layout_in_metadata(tmp_path):
    (tmp_path / _METADATA_FILE).write_text(json.dumps({"layout": "from-the-future"}))
    with pytest.raises(ValueError, match="Unknown cache layout 'from-the-future'"):
        open_cache(tmp_path, ROWS)
