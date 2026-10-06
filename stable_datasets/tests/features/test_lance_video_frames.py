from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.cache import write_lance_video_frames_cache
from stable_datasets.schema import DatasetInfo, DatasetSource, Features, Value, Version, Video
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


def test_video_frames_encode_points_to_specialized_layout(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake")
    with pytest.raises(NotImplementedError, match="specialized Lance video-frames"):
        Video(storage="frames").encode(path, cache_dir=tmp_path)


def test_arrow_storage_rejects_video_frames_layout(tmp_path):
    class _FramesBuilder(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = DatasetSource(homepage="https://example.com", assets={}, citation="TBD")

        def _info(self):
            return DatasetInfo(features=Features({"video": Video(storage="frames")}))

        def _split_generators(self):
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={})]

        def _generate_examples(self):
            yield 0, {"video": "unused.mp4"}

    with pytest.raises(ValueError, match="requires storage_format='lance'"):
        _FramesBuilder(split="train", processed_cache_dir=tmp_path, storage_format="arrow")


def _write_tiny_video(path, frames: int = 5, size: int = 12):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (size, size),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create an mp4v test video in this environment")
    for idx in range(frames):
        frame = np.full((size, size, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_lance_video_frames_cache_segment_roundtrip(tmp_path):
    pytest.importorskip("lance")
    pytest.importorskip("cv2")

    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, frames=5, size=12)
    features = Features({"video": Video(storage="frames"), "label": Value("int32")})

    def gen():
        yield "tiny", {"video": video_path, "label": 3}

    meta = write_lance_video_frames_cache(gen(), features, tmp_path / "frames", workers=1)

    from stable_datasets.backends.lance_video_frames import LanceVideoFramesBackend
    from stable_datasets.dataset import StableDataset

    backend = LanceVideoFramesBackend(uri=meta.cache_dir, window_length=3, hop_size=2)
    assert backend.num_rows == 2
    row = backend.get_row(0)
    assert len(row["video"]) == 3
    assert row["frame_indices"] == [0, 1, 2]
    assert row["label"] == 3

    ds = StableDataset(
        features=features, info=DatasetInfo(features=features), backend=backend, num_rows=backend.num_rows
    )
    sample = ds[0]
    assert sample["video"].shape == (3, 12, 12, 3)
    assert list(ds)[1]["frame_indices"] == [2, 3, 4]


def _frames_dataset(tmp_path, frame_counts=(5, 8), column="video", **backend_kwargs):
    pytest.importorskip("lance")
    pytest.importorskip("cv2")
    from stable_datasets.backends.lance_video_frames import LanceVideoFramesBackend
    from stable_datasets.dataset import StableDataset

    paths = []
    for i, frames in enumerate(frame_counts):
        path = tmp_path / f"clip{i}.mp4"
        _write_tiny_video(path, frames=frames, size=12)
        paths.append(path)
    features = Features({column: Video(storage="frames"), "label": Value("int32")})
    gen = ((f"clip{i}", {column: path, "label": i}) for i, path in enumerate(paths))
    meta = write_lance_video_frames_cache(gen, features, tmp_path / "frames", workers=1)
    backend = LanceVideoFramesBackend(uri=meta.cache_dir, **backend_kwargs)
    return StableDataset(features=features, info=DatasetInfo(features=features), backend=backend)


def test_pickle_keeps_segment_plan(tmp_path):
    import pickle

    ds = _frames_dataset(tmp_path, min_video_frames=6)
    assert len(ds) == 8
    restored = pickle.loads(pickle.dumps(ds._backend))
    assert restored.num_rows == 8
    assert restored.min_video_frames == 6


def test_iteration_reads_one_take_per_batch_and_matches_random_access(tmp_path):
    ds = _frames_dataset(tmp_path, window_length=2, frame_skip=1)
    expected = [ds[i] for i in range(len(ds))]

    lance_ds = ds._backend._dataset
    calls = []

    class _CountingDataset:
        def take(self, *args, **kwargs):
            calls.append(args)
            return lance_ds.take(*args, **kwargs)

    ds._backend._ds = _CountingDataset()
    rows = list(ds)
    assert len(calls) == ds._backend.num_shards == 2
    assert [r["sample_idx"] for r in rows] == [r["sample_idx"] for r in expected]
    for row, ref in zip(rows, expected):
        np.testing.assert_array_equal(row["video"], ref["video"])
        assert row["video"].shape == (2, 12, 12, 3)


def test_shards_follow_videos_and_workers_read_each_window_once(tmp_path):
    from torch.utils.data import DataLoader

    ds = _frames_dataset(tmp_path, frame_counts=(5, 8, 6), window_length=3)
    assert ds._backend.num_shards == 3
    assert ds._backend.shard_row_counts == [3, 6, 4]
    # The parent process already used Lance, so forked workers are not safe.
    loader = DataLoader(ds.as_iterable(), batch_size=None, num_workers=2, multiprocessing_context="spawn")
    assert sorted(int(row["sample_idx"]) for row in loader) == list(range(13))


def test_row_key_uses_feature_column_name(tmp_path):
    ds = _frames_dataset(tmp_path, column="clip", window_length=2)
    row = ds[0]
    assert "video" not in row
    assert row["clip"].shape == (2, 12, 12, 3)


def test_set_video_decode_maps_config_onto_windows(tmp_path):
    torch = pytest.importorskip("torch")
    ds = _frames_dataset(tmp_path)
    decoded = ds.set_video_decode(num_frames=2, frame_stride=2)
    assert len(decoded) == (5 - 2) + (8 - 2)
    sample = decoded[0]
    assert sample["frame_indices"] == [0, 2]
    assert isinstance(sample["video"], torch.Tensor)
    assert sample["video"].shape == (2, 3, 12, 12)
    assert float(sample["video"].max()) <= 1.0

    numpy_view = ds.set_video_decode(num_frames=3, output="numpy", layout="THWC", dtype="uint8", scale="none")
    assert numpy_view[0]["video"].shape == (3, 12, 12, 3)
    assert numpy_view[0]["video"].dtype == np.uint8

    with pytest.raises(ValueError, match="default sampling"):
        ds.set_video_decode(num_frames=2, sampling="random")


def test_builder_decode_video_with_frames_storage(tmp_path):
    pytest.importorskip("lance")
    pytest.importorskip("cv2")
    video_path = tmp_path / "clip.mp4"
    _write_tiny_video(video_path, frames=6, size=12)

    class _FramesBuilder(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = DatasetSource(homepage="https://example.com", assets={}, citation="TBD")

        def _info(self):
            return DatasetInfo(features=Features({"video": Video(storage="frames")}))

        def _split_generators(self):
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={})]

        def _generate_examples(self):
            yield 0, {"video": video_path}

    ds = _FramesBuilder(
        split="train",
        processed_cache_dir=tmp_path / "processed",
        storage_format="lance",
        decode_video={"num_frames": 4, "output": "numpy"},
    )
    assert len(ds) == 3
    assert ds[2]["video"].shape == (4, 3, 12, 12)


def test_schema_comes_from_metadata_with_stable_metadata_types(tmp_path):
    ds = _frames_dataset(tmp_path, frame_counts=(3, 4), window_length=2)
    backend = ds._backend
    backend._videos[0]["metadata"]["note"] = None
    backend._videos[1]["metadata"]["note"] = "x"
    backend._metadata_keys.append("note")

    schema = backend.schema
    assert backend._ds is None
    assert schema.field("note").type == pa.string()
    assert backend.take([0]).schema == backend.take([3]).schema == schema


def test_video_frames_writer_writes_each_video_when_encoded(tmp_path, monkeypatch):
    lance = pytest.importorskip("lance")
    import stable_datasets.cache as cache

    encoded = []

    def fake_encode(task):
        video_id, path_str, _, _ = task
        encoded.append(video_id)
        return ("ok", video_id, path_str, 2, 4, 4, [b"a", b"b"])

    encoded_at_write = []

    def fake_write_dataset(reader, uri, **kwargs):
        for _ in reader:
            encoded_at_write.append(len(encoded))
        return SimpleNamespace(version=1)

    monkeypatch.setattr(cache, "_encode_one_video_frames", fake_encode)
    monkeypatch.setattr(lance, "write_dataset", fake_write_dataset)

    paths = []
    for i in range(3):
        path = tmp_path / f"clip{i}.mp4"
        path.write_bytes(b"video")
        paths.append(path)
    features = Features({"video": Video(storage="frames")})
    gen = ((i, {"video": path}) for i, path in enumerate(paths))
    meta = write_lance_video_frames_cache(gen, features, tmp_path / "frames", workers=1)

    assert encoded_at_write == [1, 2, 3]
    assert meta.num_rows == 6


def test_video_frames_writer_with_worker_pool(tmp_path):
    pytest.importorskip("lance")
    pytest.importorskip("cv2")
    paths = []
    for i, frames in enumerate((3, 4)):
        path = tmp_path / f"clip{i}.mp4"
        _write_tiny_video(path, frames=frames, size=12)
        paths.append(path)
    features = Features({"video": Video(storage="frames")})
    gen = ((i, {"video": path}) for i, path in enumerate(paths))
    meta = write_lance_video_frames_cache(gen, features, tmp_path / "frames", workers=2)
    assert meta.num_rows == 7
