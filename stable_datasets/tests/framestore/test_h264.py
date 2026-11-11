"""stable_datasets.framestore.h264, the reference decoder of the MP4 stores' H.264 subset, against ffmpeg: every frame's Y, U and V
equal, for intra and predicted frames, one slice per frame, per macroblock row and per macroblock, odd (cropped)
sizes, and a range of qualities.

    pytest tests/framestore/test_h264.py
"""

import shutil
import subprocess

import numpy as np
import pytest

from stable_datasets.framestore import h264


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def encode(tmp_path, w, h, frames, crf=18, keyint=4, slices="rows", source="testsrc2"):
    out = tmp_path / f"{source}_{w}x{h}_{crf}_{slices}.mp4"
    params = "ref=1" + {"rows": f":slices={(h + 15) // 16}", "one": "", "mb": ":slice-max-mbs=1"}[slices]
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"{source}=size={w}x{h}:rate=30",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", str(keyint),
         "-keyint_min", str(keyint), "-sc_threshold", "0", "-bf", "0", "-tune", "fastdecode", "-profile:v", "baseline",
         "-x264-params", params, "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip
    return out


def ffmpeg_frames(path, w, h):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
        check=True,
    ).stdout
    n = w * h * 3 // 2
    return [np.frombuffer(raw[i * n : (i + 1) * n], np.uint8) for i in range(len(raw) // n)]


@pytest.mark.parametrize(
    "w, h, frames, crf, slices, source",
    [
        (96, 64, 6, 18, "rows", "testsrc2"),
        (96, 64, 5, 18, "one", "testsrc2"),
        (48, 32, 3, 18, "mb", "testsrc2"),
        (120, 72, 5, 8, "rows", "testsrc2"),  # cropped to a size that is no multiple of 16; fine quantizer
        (112, 80, 5, 32, "rows", "mandelbrot"),
    ],
)
def test_matches_ffmpeg(tmp_path, w, h, frames, crf, slices, source):
    path = encode(tmp_path, w, h, frames, crf, slices=slices, source=source)
    ref = ffmpeg_frames(path, w, h)
    got = list(h264.decode_mp4(path.read_bytes()))
    assert len(got) == len(ref) == frames
    for k, (y, u, v) in enumerate(got):
        assert y.shape == (h, w) and u.shape == v.shape == (h // 2, w // 2)
        assert np.array_equal(np.concatenate([y.ravel(), u.ravel(), v.ravel()]), ref[k]), f"frame {k}"


def test_refuses_streams_outside_the_subset(tmp_path):
    out = tmp_path / "cabac.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30", "-frames:v", "2",
         "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip
    with pytest.raises(h264.UnsupportedStream):
        next(h264.decode_mp4(out.read_bytes()))
