from pathlib import Path

import pytest

from stable_datasets.backends.arrow_shards import ArrowBackend
from stable_datasets.cache import write_lance_cache, write_sharded_arrow_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import DatasetInfo, Features, Value, Video, VideoRef


def _video_file(tmp_path: Path, name: str = "clip.mp4", data: bytes = b"fake mp4 bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_video_path_rejects_non_paths(tmp_path):
    features = Features({"video": Video(storage="path")})
    with pytest.raises(TypeError, match="Video\\(storage='path'\\)"):
        features["video"].encode(b"not a path", cache_dir=tmp_path)


def test_video_path_validates_extension(tmp_path):
    path = _video_file(tmp_path, name="clip.txt")
    features = Features({"video": Video(storage="path")})
    with pytest.raises(ValueError, match="Unsupported video extension"):
        features["video"].encode(path, cache_dir=tmp_path)


def test_video_path_stages_cache_owned_asset_and_survives_source_deletion(tmp_path):
    source = _video_file(tmp_path, data=b"stable video")
    features = Features({"video": Video(storage="path"), "label": Value("int32")})
    info = DatasetInfo(features=features)

    def gen():
        yield 0, {"video": source, "label": 1}

    cache_dir = tmp_path / "cache"
    meta = write_sharded_arrow_cache(gen(), features, cache_dir)
    source.unlink()

    ds = StableDataset(
        features=features,
        info=info,
        backend=ArrowBackend(
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
            schema=features.to_arrow_schema(),
        ),
        num_rows=meta.num_rows,
    )
    row = ds[0]
    assert isinstance(row["video"], VideoRef)
    assert row["video"].path is not None
    assert row["video"].path.exists()
    assert row["video"].bytes == b"stable video"
    assert row["video"].extension == ".mp4"


def test_video_bytes_mode_accepts_raw_bytes(tmp_path):
    features = Features({"video": Video(storage="bytes")})
    cell = features["video"].encode(b"abc123", cache_dir=tmp_path)
    assert cell["mode"] == "bytes"
    assert cell["bytes"] == b"abc123"
    assert cell["extension"] == ".mp4"
    assert cell["size"] == 6


def test_video_format_contracts(tmp_path):
    source = _video_file(tmp_path)
    features = Features({"video": Video(storage="path")})
    cell = features["video"].encode(source, cache_dir=tmp_path)

    assert isinstance(features["video"].format(cell, format_type="default", cache_dir=tmp_path), VideoRef)
    assert isinstance(features["video"].format(cell, format_type="numpy", cache_dir=tmp_path), VideoRef)
    assert isinstance(features["video"].format(cell, format_type="torch", cache_dir=tmp_path), VideoRef)
    assert features["video"].format(cell, format_type="raw", cache_dir=tmp_path) == cell


def test_video_roundtrip_matches_arrow_and_lance(tmp_path):
    pytest.importorskip("lance")
    source = _video_file(tmp_path, data=b"same content")
    features = Features({"video": Video(storage="path"), "label": Value("int32")})
    info = DatasetInfo(features=features)

    def gen():
        yield 0, {"video": source, "label": 7}

    arrow_meta = write_sharded_arrow_cache(gen(), features, tmp_path / "arrow")
    lance_meta = write_lance_cache(gen(), features, tmp_path / "lance")

    arrow_ds = StableDataset(
        features=features,
        info=info,
        backend=ArrowBackend(
            shard_paths=arrow_meta.shard_paths,
            shard_row_counts=arrow_meta.shard_row_counts,
            schema=features.to_arrow_schema(),
        ),
        num_rows=arrow_meta.num_rows,
    )
    from stable_datasets.backends.lance_rows import LanceBackend

    lance_ds = StableDataset(
        features=features,
        info=info,
        backend=LanceBackend(uri=lance_meta.cache_dir),
        num_rows=lance_meta.num_rows,
    )

    assert arrow_ds[0]["video"].bytes == lance_ds[0]["video"].bytes == b"same content"
    assert arrow_ds[0]["label"] == lance_ds[0]["label"] == 7
