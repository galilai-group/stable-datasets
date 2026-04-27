import numpy as np
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
    assert row["video"].shape == (3, 12, 12, 3)
    assert row["frame_indices"] == [0, 1, 2]
    assert row["label"] == 3

    ds = StableDataset(features=features, info=DatasetInfo(features=features), backend=backend, num_rows=backend.num_rows)
    sample = ds[0]
    assert sample["video"].shape == (3, 12, 12, 3)
    assert list(ds)[1]["frame_indices"] == [2, 3, 4]
