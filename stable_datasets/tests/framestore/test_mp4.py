"""MP4 stores (stable_datasets.framestore.build --format mp4, stable_datasets.framestore.mp4) on small synthetic videos (CPU): a built store has a
row per video with its kept frames and even size, every video coded at one canvas size (portrait ones transposed) and
read back in its own size and orientation, clips start on keyframes and decode to the same pixels as the whole video
decoded front to back, close to the source frames, read through a memory map or pread alike.

    pytest tests/framestore/test_mp4.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import shutil
import subprocess

import numpy as np
import pytest


lance = pytest.importorskip("lance")
pytest.importorskip("numba")
pytest.importorskip("torchcodec")

from stable_datasets.framestore.build import build_store, video_decoder
from stable_datasets.framestore.mp4 import Mp4Store, encode, fit


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    d = tmp_path_factory.mktemp("videos")
    videos = []
    for i, (size, secs) in enumerate([("320x240", 3), ("240x320", 4), ("322x242", 2)]):
        p = d / f"v{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
             "-t", str(secs), "-pix_fmt", "yuv420p", str(p)],
            check=True,
        )  # fmt: skip
        videos.append(p)
    out = str(d / "v.lance")
    s = build_store(videos, out, short_edge=97, fps=7.5, format="mp4", cpu=2, nice=0, columns={"label": [3, 4, 5]})
    assert s["videos"] == 3 and not s["failed"]
    return out, videos


def test_rows(store):
    out, _ = store
    cols = ["episode_idx", "frames", "fps", "h", "w", "transposed", "label"]
    t = lance.dataset(out).to_table(columns=cols).sort_by("episode_idx")
    assert t.column("frames").to_pylist() == [23, 30, 15]  # every 4th of 90, 120, 60 frames
    assert t.column("fps").to_pylist() == [7.5] * 3
    assert t.column("label").to_pylist() == [3, 4, 5]
    h, w = t.column("h").to_numpy(), t.column("w").to_numpy()
    assert h.tolist() == [96] * 3 and w.tolist() == [128] * 3  # 97 cropped to even; the portrait one transposed
    assert t.column("transposed").to_pylist() == [False, True, False]
    s = Mp4Store(out)
    assert s.meta["canvas"] == [96, 172]  # 16:9 at the short edge, even
    v = int(np.flatnonzero(s.transposed)[0])
    assert s.decoder(v).get_frames_in_range(0, 1).data.shape[2:] == (96, 172)  # coded on the canvas
    assert s.decode(v, 0, 4).shape == (4, 128, 96, 3)  # read back portrait


def test_clips_start_on_keyframes(store):
    s = Mp4Store(store[0])
    v, start = s.clips(np.random.default_rng(0), 2000, frames=16)
    assert (start % s.keyint == 0).all() and (start + 16 <= s.n[v]).all()
    assert set(np.unique(v)) == set(np.flatnonzero(s.n >= 16))  # the 15-frame video holds no clip
    for k in np.flatnonzero(s.n >= 16):  # every keyframe start that fits is drawn
        assert set(start[v == k]) == set(range(0, s.n[k] - 15, s.keyint))


@pytest.mark.parametrize("read", ["mmap", "pread"])
def test_clip_matches_whole_video(store, read):
    out, videos = store
    s = Mp4Store(out, read=read)
    for v in range(len(s)):
        whole = s.decode(v, 0, int(s.n[v]))
        for start in range(0, int(s.n[v]) - 7, s.keyint):
            assert np.array_equal(s.decode(v, start, 8), whole[start : start + 8])
    ep = lance.dataset(out).to_table(columns=["episode_idx"]).column(0).to_numpy()
    v = int(np.flatnonzero(ep == 0)[0])
    d = video_decoder()(str(videos[0]), 97)
    src = d.frames(list(range(0, d.n, 4)))[:, : s.loc["h"][v], : s.loc["w"][v]].astype(np.float64)
    got = s.decode(v, 0, int(s.n[v])).astype(np.float64)

    def psnr(a, b):
        return 10 * np.log10(255**2 / ((a - b) ** 2).mean())

    # the kept source frames, in order (testsrc2's sharp colour edges lose to 4:2:0: ~23 dB even at crf 0)
    assert psnr(got, src) > 20 and psnr(got, src) > psnr(got[1:], src[:-1]) + 5
    assert psnr(got, src) > psnr(got[:-1], src[1:]) + 5


def test_encode_quality():
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    from torchcodec.decoders import VideoDecoder

    yy, xx = np.mgrid[:96, :128]
    x = np.stack(
        [np.stack([128 + 100 * np.sin((xx + 3 * t) / 17 + yy / 31), 128 + 90 * np.cos(yy / 13 + t / 5),
                   128 + 80 * np.sin((xx + yy) / 23)], -1) for t in range(12)]
    ).astype(np.uint8)  # fmt: skip
    got = VideoDecoder(encode([x], 96, 128, 7.5), dimension_order="NHWC").get_frames_in_range(0, 12).data.numpy()
    assert 10 * np.log10(255**2 / ((got.astype(np.float64) - x) ** 2).mean()) > 38


def test_portrait_reads_back_like_its_source(store):
    out, videos = store
    s = Mp4Store(out)
    ep = lance.dataset(out).to_table(columns=["episode_idx"]).column(0).to_numpy()
    v = int(np.flatnonzero(ep == 1)[0])
    d = video_decoder()(str(videos[1]), 97)
    src = d.frames(list(range(0, d.n, 4)))[:, :128, :96].astype(np.float64)
    got = s.decode(v, 0, int(s.n[v])).astype(np.float64)
    assert got.shape == src.shape and 10 * np.log10(255**2 / ((got - src) ** 2).mean()) > 20


def test_fit():
    x = np.arange(2 * 6 * 4 * 3, dtype=np.uint8).reshape(2, 6, 4, 3)  # portrait: transposed, then padded
    y, h, w, tr = fit(x, (4, 8))
    assert y.shape == (2, 4, 8, 3) and (h, w, tr) == (4, 6, True)
    assert np.array_equal(y[:, :4, :6], x.transpose(0, 2, 1, 3)) and (y[:, :, 6:] == y[:, :, 5:6]).all()
    y, h, w, tr = fit(np.zeros((1, 4, 16, 3), np.uint8), (4, 8))  # wider than the canvas: scaled to fit
    assert y.shape == (1, 4, 8, 3) and (h, w, tr) == (2, 8, False)


def test_decode_resized(store):
    s = Mp4Store(store[0])
    assert s.decode(1, 4, 16, size=(64, 80)).shape == (16, 64, 80, 3)


def test_encode_keyframes_every_keyint():
    if shutil.which("ffprobe") is None:
        pytest.skip("needs ffprobe")
    x = np.random.default_rng(0).integers(0, 255, (13, 32, 48, 3), np.uint8)
    data = encode([x[:5], x[5:]], 32, 48, 7.5, keyint=4)
    r = subprocess.run(
        ["ffprobe", "-loglevel", "error", "-select_streams", "v", "-show_entries", "frame=key_frame,pict_type",
         "-of", "csv=p=0", "-"],
        input=data, capture_output=True, check=True,
    )  # fmt: skip
    rows = [line.split(",")[:2] for line in r.stdout.decode().split()]
    assert [k for k, _ in rows] == ["1" if i % 4 == 0 else "0" for i in range(13)]
    assert {t for _, t in rows} <= {"I", "P"}  # no B-frames
