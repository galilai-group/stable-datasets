"""stable_datasets.framestore.build on small synthetic videos (CPU): frames encoded with restart markers decode to Pillow's pixels,
a built store reads back through FrameStore with the right episodes, sizes and labels, corrupt videos are skipped,
and a build resumes but refuses other settings.

    pytest tests/framestore/test_build.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import io
import shutil
import subprocess

import numpy as np
import pytest


lance = pytest.importorskip("lance")
pytest.importorskip("numba")
Image = pytest.importorskip("PIL.Image")

from stable_datasets.framestore import codec
from stable_datasets.framestore.build import build_store, video_decoder
from stable_datasets.framestore.store import FrameStore


def test_encode_matches_pillow():
    yy, xx = np.mgrid[:100, :141]
    x = np.stack([np.stack([(xx * 2 + i) % 256, (yy * 3) % 256, ((xx + yy) * (i + 1)) % 256], -1) for i in range(3)])
    buf, off = codec.encode(x.astype(np.uint8), 90)
    info = codec.probe(buf, off)
    _, _, ok = codec.segment_lengths(buf, off, info)
    assert ok.all() and (codec.layout(info) == "4:2:0").all()
    for i in range(3):
        got = np.array(Image.open(io.BytesIO(buf[off[i] : off[i + 1]].tobytes())))
        b = io.BytesIO()
        Image.fromarray(x[i].astype(np.uint8)).save(b, "JPEG", quality=90)
        assert np.array_equal(got, np.array(Image.open(b)))


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    try:
        video_decoder()
    except ImportError:
        pytest.skip("needs torchcodec or decord")
    d = tmp_path_factory.mktemp("videos")
    out = []
    for i, (size, secs) in enumerate([("320x240", 2), ("320x240", 3), ("240x320", 2)]):
        p = d / f"{'ab'[i // 2]}" / f"v{i}.mp4"
        p.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
             "-t", str(secs), "-pix_fmt", "yuv420p", str(p)],
            check=True,
        )  # fmt: skip
        out.append(p)
    bad = d / "b" / "bad.mp4"
    bad.write_bytes(b"not a video")
    return out + [bad]


def test_build_reads_back(videos, tmp_path):
    out = str(tmp_path / "v.lance")
    labels = [0, 0, 1, 1]
    s = build_store(videos, out, short_edge=96, fps=15, columns={"label": labels}, cpu=2, nice=0, commit_gb=1e-5)
    assert s["videos"] == 3 and len(s["failed"]) == 1
    t = lance.dataset(out).to_table(columns=["episode_idx", "step_idx", "h", "w", "label"]).to_pandas()
    per = t.groupby("episode_idx").agg(
        n=("step_idx", "size"), h=("h", "first"), w=("w", "first"), label=("label", "first")
    )
    assert per.loc[0, "n"] == 30 and per.loc[1, "n"] == 45 and per.loc[2, "n"] == 30
    assert tuple(per.loc[0, ["h", "w"]]) == (96, 128) and tuple(per.loc[2, ["h", "w"]]) == (128, 96)
    assert list(per["label"]) == [0, 0, 1]
    store = FrameStore(out, index=True)
    assert store.format is not None and sorted(store.ep_len) == [30, 30, 45]


def test_build_resumes_and_refuses_other_settings(videos, tmp_path):
    out = str(tmp_path / "v.lance")
    build_store(videos[:2], out, short_edge=96, fps=15, cpu=2, nice=0)
    rows = lance.dataset(out).count_rows()
    s = build_store(videos[:2], out, short_edge=96, fps=15, cpu=2, nice=0)
    assert s["videos"] == 0 and lance.dataset(out).count_rows() == rows
    with pytest.raises(ValueError, match="was built with"):
        build_store(videos[:2], out, short_edge=128, fps=15, cpu=2, nice=0)
