"""video_storage=: a builder's video column stored as a framestore store (MP4 or JPEG), its samples clip windows.

pytest stable_datasets/tests/video/test_video_storage.py      (needs ffmpeg and torchcodec)
"""

import shutil
import subprocess

import numpy as np
import pytest

from stable_datasets.schema import DatasetInfo, Features, Value, Version, Video, VideoStorage
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


pytest.importorskip("torchcodec")
pytest.importorskip("lance")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

VIDEOS = []


class TinyVideos(BaseDatasetBuilder):
    VERSION = Version("0.1.0")
    SOURCE = {"homepage": "https://example.com", "assets": {}, "citation": "synthetic test videos"}

    def _info(self):
        return DatasetInfo(features=Features({"video": Video(), "label": Value("int64")}))

    def _split_generators(self):
        return [SplitGenerator(name="train", gen_kwargs={})]

    def _iter_examples(self):
        for i, p in enumerate(VIDEOS):
            yield str(i), {"video": str(p), "label": 10 + i}


@pytest.fixture(scope="module", autouse=True)
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    VIDEOS.clear()
    for i, (size, secs) in enumerate([("320x240", 3), ("240x320", 4)]):  # 23 and 30 frames at 7.5 fps
        p = d / f"v{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30", "-t",
             str(secs), "-pix_fmt", "yuv420p", str(p)],
            check=True,
        )  # fmt: skip
        VIDEOS.append(p)
    return VIDEOS


def build(tmp_path, storage):
    return TinyVideos(split="train", processed_cache_dir=tmp_path / "processed", video_storage=storage)


def test_mp4_store_windows(tmp_path):
    ds = build(tmp_path, VideoStorage(fps=7.5, short_edge=96))
    assert ds._features["video"].storage == "mp4"
    clips = ds.set_video_decode(num_frames=8, output="numpy", dtype="uint8", scale="none", layout="THWC")
    # windows start on keyframes (every 4 frames): (23 - 8) // 4 + 1 + (30 - 8) // 4 + 1
    assert len(clips) == 4 + 6
    first, last = clips[0], clips[len(clips) - 1]
    assert first["video"].shape == (8, 96, 128, 3) and first["video"].dtype == np.uint8
    assert last["video"].shape == (8, 128, 96, 3)  # the portrait video, read back portrait
    assert {first["label"], last["label"]} == {10, 11}
    assert clips[1]["start_frame"] - clips[0]["start_frame"] == 4
    with pytest.raises(ValueError, match="frame_skip must be 0"):
        ds.set_video_decode(num_frames=4, frame_stride=2)


def test_jpeg_store_windows(tmp_path):
    ds = build(tmp_path, {"format": "jpeg", "fps": 7.5, "short_edge": 96})
    clips = ds.set_video_decode(
        num_frames=4, frame_stride=2, output="numpy", dtype="uint8", scale="none", layout="THWC"
    )
    assert len(clips) == (23 - 6) + (30 - 6)  # any start (a window spans 7 frames); stride chosen when loading
    assert clips[0]["video"].shape[0] == 4


def test_storage_is_part_of_the_cache_key(tmp_path):
    a = build(tmp_path, VideoStorage(fps=7.5, short_edge=96))
    b = build(tmp_path, VideoStorage(fps=7.5, short_edge=96, crf=30))
    assert a._backend.store_path != b._backend.store_path
    c = build(tmp_path, VideoStorage(fps=7.5, short_edge=96))  # the same settings: the cache is reused
    assert c._backend.store_path == a._backend.store_path


def test_store_window_matches_the_store(tmp_path):
    from stable_datasets.framestore.mp4 import Mp4Store

    ds = build(tmp_path, VideoStorage(fps=7.5, short_edge=96)).set_video_decode(
        num_frames=8, output="numpy", dtype="uint8", scale="none", layout="THWC"
    )
    row = ds[2]
    store = Mp4Store(ds._backend.store_path)
    np.testing.assert_array_equal(row["video"], store.decode(row["video_idx"], row["start_frame"], 8))


def test_torchcodec_decodes_the_sampled_frames(tmp_path):
    """decoder="torchcodec" returns the frames at the sampled indices (it once passed seconds to get_frame_at, which
    takes an index)."""
    ds = build(tmp_path, None)
    clips = ds.set_video_decode(
        num_frames=4, frame_stride=3, sampling="start", decoder="torchcodec", output="numpy", dtype="uint8",
        scale="none", layout="THWC",
    )  # fmt: skip
    got = clips[0]["video"]
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(VIDEOS[0]), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    every = np.frombuffer(raw, np.uint8).reshape(-1, 240, 320, 3)
    assert got.shape == (4, 240, 320, 3)
    for k, idx in enumerate([0, 3, 6, 9]):
        err = np.abs(got[k].astype(int) - every[idx]).mean()
        assert err < 2, (k, idx, err)
        assert err < np.abs(got[k].astype(int) - every[idx + 1]).mean()  # and not a neighbouring frame
