"""Tests for cache locations that are URIs (``file://`` here, ``s3://`` in production)."""

import pickle
from pathlib import Path

import pyarrow.fs as pafs
import pytest

from stable_datasets import _uri
from stable_datasets.backends.lance_rows import LanceBackend
from stable_datasets.cache import open_cache, write_lance_cache, write_lance_video_frames_cache
from stable_datasets.schema import DatasetInfo, Features, Value, Version, Video
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


OPTIONS = {"allow_http": "true"}


def _file_uri(path: Path) -> str:
    return f"file://{path}"


def test_uri_helpers_keep_the_scheme():
    assert _uri.join("s3://bucket/prefix/", "a", "_metadata.json") == "s3://bucket/prefix/a/_metadata.json"
    assert _uri.local_path("s3://bucket/prefix") is None
    assert _uri.local_path("file:///data/cache") == Path("/data/cache")
    assert _uri.local_path("/data/cache") == Path("/data/cache")


def test_s3_storage_options_reach_the_metadata_filesystem():
    options = {"aws_access_key_id": "a", "aws_secret_access_key": "b", "aws_region": "eu-west-1"}
    fs, path = _uri._filesystem("s3://bucket/prefix/_metadata.json", options)
    assert isinstance(fs, pafs.S3FileSystem)
    assert fs.region == "eu-west-1"
    assert path == "bucket/prefix/_metadata.json"


def test_azure_storage_options_reach_the_metadata_filesystem():
    fs, path = _uri._filesystem(
        "az://container/prefix/_metadata.json", {"account_name": "acct", "account_key": "YQ=="}
    )
    assert isinstance(fs, pafs.AzureFileSystem)
    assert path == "container/prefix/_metadata.json"
    with pytest.raises(ValueError, match="Azure account name"):
        _uri._filesystem("az://container/prefix", None)


def test_lance_backend_keeps_remote_uri_as_string():
    backend = LanceBackend(uri="s3://bucket/cache", shard_row_counts=[3], storage_options=OPTIONS)
    restored = pickle.loads(pickle.dumps(backend))
    assert restored._uri == "s3://bucket/cache"
    assert restored._storage_options == OPTIONS
    assert restored.cache_dir is None
    assert restored.num_rows == 3


def test_lance_cache_roundtrip_through_file_uri(tmp_path):
    features = Features({"x": Value("int32")})
    uri = _file_uri(tmp_path / "lance")
    meta = write_lance_cache(((i, {"x": i}) for i in range(5)), features, uri, storage_options=OPTIONS)
    assert meta.cache_dir == uri

    opened = open_cache(uri, features, backend_kwargs={"batch_readahead": 2}, storage_options=OPTIONS)
    backend = opened.backend
    assert backend._uri == uri
    assert backend._storage_options == OPTIONS
    assert backend._batch_readahead == 2
    assert backend.cache_dir == tmp_path / "lance"
    assert backend.take([4, 0]).column("x").to_pylist() == [4, 0]

    overridden = open_cache(uri, features, backend_kwargs={"shard_row_counts": [2, 3]}).backend
    assert overridden.shard_row_counts == [2, 3]


def test_lance_rewrite_publishes_a_new_commit(tmp_path):
    import lance

    features = Features({"x": Value("int32")})
    uri = _file_uri(tmp_path / "lance")
    write_lance_cache(((i, {"x": i}) for i in range(5)), features, uri)
    write_lance_cache(((i, {"x": i * 10}) for i in range(3)), features, uri)

    dataset = lance.dataset(uri)
    assert dataset.version == 2
    assert dataset.to_table().column("x").to_pylist() == [0, 10, 20]
    assert open_cache(uri, features).num_rows == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lance", "lance.lock"]


def test_video_frames_cache_through_file_uri(tmp_path, monkeypatch):
    import stable_datasets.cache as cache

    monkeypatch.setattr(
        cache, "_encode_one_video_frames", lambda task: ("ok", task[0], task[1], 2, 4, 4, [b"a", b"b"])
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    features = Features({"video": Video(storage="frames")})
    uri = _file_uri(tmp_path / "frames")
    write_lance_video_frames_cache(iter([(0, {"video": video})]), features, uri, workers=1)

    opened = open_cache(uri, features, backend_kwargs={"window_length": 2})
    assert opened.backend._uri == uri
    assert opened.num_rows == 1
    assert opened.backend.get_row(0)["video"] == [b"a", b"b"]


class _UriBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={})]

    def _generate_examples(self):
        for i in range(4):
            yield i, {"x": i}


def test_builder_with_file_uri_cache_dir(tmp_path):
    uri = _file_uri(tmp_path / "processed")
    first = _UriBuilder(split="train", processed_cache_dir=uri, storage_format="lance", storage_options=OPTIONS)
    second = _UriBuilder(split="train", processed_cache_dir=uri, storage_format="lance", storage_options=OPTIONS)

    assert [row["x"] for row in second] == [0, 1, 2, 3]
    assert second._backend._uri.startswith(uri)
    assert second._backend._storage_options == OPTIONS
    assert first._stable_datasets_processed_cache_dir == tmp_path / "processed"


def test_builder_rejects_remote_arrow_cache():
    with pytest.raises(ValueError, match="Arrow caches must be local"):
        _UriBuilder(split="train", processed_cache_dir="s3://bucket/processed", storage_format="arrow")
