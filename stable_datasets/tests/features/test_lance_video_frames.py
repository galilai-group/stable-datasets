import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.cache import write_lance_video_frames_cache
from stable_datasets.schema import DatasetInfo, DatasetSource, Features, Value, Version, Video
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


def _require_lance_dataset_api():
    lance = pytest.importorskip("lance")
    if not hasattr(lance, "dataset") or not hasattr(lance, "write_dataset"):
        pytest.skip("Installed 'lance' module does not expose dataset/write_dataset APIs")


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
    _require_lance_dataset_api()
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
    assert row["video"].shape == (3, 12, 12, 3)
    assert row["frame_indices"] == [0, 1, 2]
    assert row["label"] == 3

    ds = StableDataset(
        features=features, info=DatasetInfo(features=features), backend=backend, num_rows=backend.num_rows
    )
    sample = ds[0]
    assert sample["video"].shape == (3, 12, 12, 3)
    assert list(ds)[1]["frame_indices"] == [2, 3, 4]


class _FakeTakeResult:
    def __init__(self, rows, blobs):
        self._rows = list(rows)
        self._blobs = blobs

    def column(self, name):
        assert name == "bytes"
        selected = [self._blobs[row] for row in self._rows]
        return pa.array(selected, type=pa.large_binary())


class _FakeDataset:
    def __init__(self, blobs):
        self._blobs = blobs
        self.take_calls = []

    def take(self, rows, columns=None):
        assert columns == ["bytes"]
        self.take_calls.append(list(rows))
        return _FakeTakeResult(rows, self._blobs)


def _make_fake_backend():
    from stable_datasets.backends.lance_video_frames import LanceVideoFramesBackend

    backend = LanceVideoFramesBackend.__new__(LanceVideoFramesBackend)
    backend.window_length = 3
    backend.frame_skip = 0
    backend.hop_size = 1
    backend._stride = 1
    backend._span = 3
    backend._batch_readahead = 1
    backend._iter_batch_size = 2
    backend._uri = None
    backend._videos = [
        {
            "path": "v0.mp4",
            "start_row": 0,
            "H": 2,
            "W": 2,
            "metadata": {"label": 7},
        }
    ]
    backend._seg_vid = np.array([0, 0], dtype=np.int64)
    backend._seg_start = np.array([0, 1], dtype=np.int64)
    backend._per_video = [2]

    blobs = {
        0: b"f0",
        1: b"f1",
        2: b"f2",
        3: b"f3",
    }
    fake_ds = _FakeDataset(blobs)
    backend._ds = fake_ds

    def _fake_decode(blobs_in, height, width):
        del height, width
        decoded = np.empty((len(blobs_in), 2, 2, 3), dtype=np.uint8)
        for i, blob in enumerate(blobs_in):
            decoded[i] = int(blob[-1])
        return decoded

    backend._decode_blobs = _fake_decode
    return backend, fake_ds


def test_lance_video_frames_take_preserves_order_and_dedupes_overlaps():
    backend, fake_ds = _make_fake_backend()

    table = backend.take([1, 0, 1])
    rows = table.to_pylist()

    assert [row["sample_idx"] for row in rows] == [1, 0, 1]
    assert [row["frame_indices"] for row in rows] == [[1, 2, 3], [0, 1, 2], [1, 2, 3]]
    assert fake_ds.take_calls == [[1, 2, 3, 0]]


def test_lance_video_frames_iter_batches_uses_chunked_take():
    backend, fake_ds = _make_fake_backend()

    batches = list(backend.iter_batches())
    assert len(batches) == 1
    rows = pa.Table.from_batches(batches).to_pylist()
    assert [row["sample_idx"] for row in rows] == [0, 1]
    assert fake_ds.take_calls == [[0, 1, 2, 3]]


def test_lance_video_frames_take_prefetches_multi_video_rows_in_single_wave():
    from stable_datasets.backends.lance_video_frames import LanceVideoFramesBackend

    backend = LanceVideoFramesBackend.__new__(LanceVideoFramesBackend)
    backend.window_length = 2
    backend.frame_skip = 0
    backend.hop_size = 1
    backend._stride = 1
    backend._span = 2
    backend._batch_readahead = 1
    backend._iter_batch_size = 4
    backend._uri = None
    backend._videos = [
        {"path": "v0.mp4", "start_row": 0, "H": 2, "W": 2, "metadata": {"label": 7}},
        {"path": "v1.mp4", "start_row": 10, "H": 2, "W": 2, "metadata": {"label": 9}},
    ]
    backend._seg_vid = np.array([0, 1], dtype=np.int64)
    backend._seg_start = np.array([0, 0], dtype=np.int64)
    backend._per_video = [1, 1]

    blobs = {
        0: b"a0",
        1: b"a1",
        10: b"b0",
        11: b"b1",
    }
    fake_ds = _FakeDataset(blobs)
    backend._ds = fake_ds

    def _fake_decode(blobs_in, height, width):
        del height, width
        decoded = np.empty((len(blobs_in), 2, 2, 3), dtype=np.uint8)
        for i, blob in enumerate(blobs_in):
            decoded[i] = int(blob[-1])
        return decoded

    backend._decode_blobs = _fake_decode

    rows = backend.get_rows([1, 0])
    assert [row["video_idx"] for row in rows] == [1, 0]
    assert [row["frame_indices"] for row in rows] == [[0, 1], [0, 1]]
    assert fake_ds.take_calls == [[10, 11, 0, 1]]
