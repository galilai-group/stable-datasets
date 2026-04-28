from pathlib import Path

import pytest

from stable_datasets.backends.arrow_shards import ArrowBackend
from stable_datasets.cache import write_lance_cache, write_sharded_arrow_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import DatasetInfo, Features, Value, Video, VideoDecodeConfig, VideoRef


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


def test_video_ref_is_storage_only(tmp_path):
    source = _video_file(tmp_path)
    ref = Video(storage="path").format(
        Video(storage="path").encode(source, cache_dir=tmp_path),
        format_type="default",
        cache_dir=tmp_path,
    )

    assert isinstance(ref, VideoRef)
    assert not hasattr(ref, "open")
    assert not hasattr(ref, "get_frame_at")


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


def _video_dataset(tmp_path: Path, n: int = 3) -> StableDataset:
    features = Features({"video": Video(storage="path"), "label": Value("int32")})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            source = _video_file(tmp_path, name=f"clip_{i}.mp4", data=f"video-{i}".encode())
            yield i, {"video": source, "label": i}

    meta = write_sharded_arrow_cache(gen(), features, tmp_path / "decode_cache")
    return StableDataset(
        features=features,
        info=info,
        backend=ArrowBackend(
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
            schema=features.to_arrow_schema(),
        ),
        num_rows=meta.num_rows,
    )


def test_video_decode_config_validation():
    assert VideoDecodeConfig(num_frames=1).layout == "TCHW"
    with pytest.raises(ValueError, match="num_frames"):
        VideoDecodeConfig(num_frames=0)
    with pytest.raises(ValueError, match="frame_stride"):
        VideoDecodeConfig(num_frames=1, frame_stride=0)
    with pytest.raises(ValueError, match="sampling"):
        VideoDecodeConfig(num_frames=1, sampling="middle")
    with pytest.raises(ValueError, match="zero_one"):
        VideoDecodeConfig(num_frames=1, dtype="uint8", scale="zero_one")


def test_set_video_decode_decodes_without_mutating_original(tmp_path):
    ds = _video_dataset(tmp_path)
    calls = []

    def decode_fn(ref, config, *, row=None, sample_index=None):
        calls.append((sample_index, row["label"]))
        return f"{sample_index}:{ref.bytes.decode()}"

    decoded = ds.set_video_decode(VideoDecodeConfig(num_frames=1, decode_fn=decode_fn))

    assert isinstance(ds[0]["video"], VideoRef)
    row = decoded[0]
    assert row["video"] == "0:video-0"
    assert row["label"] == 0
    assert calls == [(0, 0)]


def test_set_video_decode_none_disables_decoding(tmp_path):
    ds = _video_dataset(tmp_path)

    def decode_fn(ref, config, *, row=None, sample_index=None):
        return "decoded"

    decoded = ds.set_video_decode(VideoDecodeConfig(num_frames=1, decode_fn=decode_fn))
    restored = decoded.set_video_decode(None)

    assert decoded[0]["video"] == "decoded"
    assert isinstance(restored[0]["video"], VideoRef)


def test_set_video_decode_raw_format_wraps_raw_struct(tmp_path):
    ds = _video_dataset(tmp_path).with_format("raw")

    def decode_fn(ref, config, *, row=None, sample_index=None):
        return ref.bytes.decode()

    decoded = ds.set_video_decode(num_frames=1, decode_fn=decode_fn)

    assert isinstance(ds[0]["video"], dict)
    assert decoded[0]["video"] == "video-0"


def test_set_video_decode_uses_batched_hook_before_transform(tmp_path):
    ds = _video_dataset(tmp_path)
    calls = []

    def decode_fn_batched(refs, config, *, rows=None, sample_indices=None):
        calls.append((len(refs), list(sample_indices), [row["label"] for row in rows]))
        return [f"batch:{idx}:{ref.bytes.decode()}" for idx, ref in zip(sample_indices, refs)]

    def transform(row):
        row = dict(row)
        row["video"] = f"transform:{row['video']}"
        return row

    decoded = ds.set_video_decode(
        VideoDecodeConfig(num_frames=1, decode_fn_batched=decode_fn_batched)
    ).with_transform(transform)
    rows = decoded.__getitems__([2, 0])

    assert [row["video"] for row in rows] == [
        "transform:batch:2:video-2",
        "transform:batch:0:video-0",
    ]
    assert calls == [(2, [2, 0], [2, 0])]


def test_set_video_decode_falls_back_to_per_row_decode_for_batches(tmp_path):
    ds = _video_dataset(tmp_path)
    calls = []

    def decode_fn(ref, config, *, row=None, sample_index=None):
        calls.append(sample_index)
        return f"row:{sample_index}:{ref.bytes.decode()}"

    decoded = ds.set_video_decode(num_frames=1, decode_fn=decode_fn)
    rows = decoded.__getitems__([1, 2])

    assert [row["video"] for row in rows] == ["row:1:video-1", "row:2:video-2"]
    assert calls == [1, 2]


def test_set_video_decode_rejects_non_video_column(tmp_path):
    ds = _video_dataset(tmp_path)
    with pytest.raises(ValueError, match="must exist and be a Video feature"):
        ds.set_video_decode(num_frames=1, column="label")
