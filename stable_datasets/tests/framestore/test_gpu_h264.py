"""stable_datasets.framestore.gpu_h264, the GPU decoder of the MP4 stores' H.264 subset: its parse arrays equal stable_datasets.framestore.h264_parse's
and its frames equal ffmpeg's (Y, U, V), for clips of whole GOPs from several videos in one batch.

    pytest tests/framestore/test_gpu_h264.py      (needs a GPU and ffmpeg)
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import shutil
import subprocess

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg"),
]

from stable_datasets.framestore import gpu_h264, h264, h264_parse


def encode(path, w, h, frames, crf=18, keyint=4, source="testsrc2"):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"{source}=size={w}x{h}:rate=30",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-g", str(keyint),
         "-keyint_min", str(keyint), "-sc_threshold", "0", "-bf", "0", "-tune", "fastdecode", "-profile:v", "baseline",
         "-x264-params", f"ref=1:slices={(h + 15) // 16}", "-threads", "1", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        check=True,
    )  # fmt: skip
    return path.read_bytes()


def ffmpeg_yuv(data, w, h):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        input=data,
        capture_output=True,
        check=True,
    ).stdout
    n = w * h * 3 // 2
    out = []
    for i in range(len(raw) // n):
        f = np.frombuffer(raw[i * n : (i + 1) * n], np.uint8)
        out.append((f[: w * h].reshape(h, w), f[w * h : w * h * 5 // 4].reshape(h // 2, w // 2),
                    f[w * h * 5 // 4 :].reshape(h // 2, w // 2)))  # fmt: skip
    return out


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("h264")
    return [
        (encode(d / "a.mp4", 112, 64, 12, 18), 112, 64),
        (encode(d / "b.mp4", 112, 64, 8, 8, source="mandelbrot"), 112, 64),
        (encode(d / "c.mp4", 112, 64, 8, 30, source="testsrc"), 112, 64),
    ]


def test_parse_matches_numba(videos):
    data, w, h = videos[0]
    batch = gpu_h264.prepare([(data, 0, 8)])
    _, arrays = gpu_h264.decode(batch, clear=True)
    sps_nal, pps_nal, nal_len, samples = h264.mp4_samples(data)
    sps, pps = h264.parse_sps(h264.unescape(sps_nal)), h264.parse_pps(h264.unescape(pps_nal))
    for k in range(8):
        nals = [h264.unescape(n) for n in h264.sample_nals(data, *samples[k], nal_len) if n[0] & 31 in (1, 5)]
        ref = h264_parse.parse_frame(nals, sps, pps)
        for name in ("mbtype", "qp", "cbp", "i4", "mv", "coef"):
            got = arrays[name][k].cpu().numpy()
            assert np.array_equal(got, getattr(ref, name)), f"frame {k}: {name}"
        intra = ref.mbtype >= 2
        assert np.array_equal(arrays["modes"][k].cpu().numpy()[intra], ref.modes[intra]), f"frame {k}: modes"


def test_frames_match_ffmpeg(videos):
    clips = [(videos[0][0], 4, 8), (videos[1][0], 0, 8), (videos[2][0], 4, 4), (videos[0][0], 0, 4)]
    (Y, U, V), _ = gpu_h264.decode(gpu_h264.prepare(clips))
    f = 0
    for data, first, count in clips:
        ref = ffmpeg_yuv(data, 112, 64)
        for k in range(first, first + count):
            for name, got, want in zip("YUV", (Y[f], U[f], V[f]), ref[k]):
                g = got.cpu().numpy()[: want.shape[0], : want.shape[1]]
                assert np.array_equal(g, want), f"batch frame {f} (video frame {k}): {name}"
            f += 1


def test_store_clips_match_ffmpeg(tmp_path):
    """A store built for the GPU decoder: clips gathered by their slice index and prepared on the GPU (escape bytes
    removed there) decode to ffmpeg's frames."""
    from stable_datasets.framestore.build import build_store
    from stable_datasets.framestore.mp4 import Mp4Store

    src = []
    for i, size in enumerate(["160x120", "120x160"]):
        p = tmp_path / f"s{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30", "-t", "3",
             "-pix_fmt", "yuv420p", str(p)],
            check=True,
        )  # fmt: skip
        src.append(p)
    out = str(tmp_path / "s.lance")
    build_store(src, out, short_edge=96, fps=7.5, format="mp4", cpu=2, nice=0)
    store = Mp4Store(out)
    clips = [(v, s, 8) for v in range(len(store)) for s in (0, 4) if s + 8 <= store.n[v]]
    raw, table, geom = gpu_h264.gather(store, clips)
    (Y, U, V), _ = gpu_h264.decode(gpu_h264.prepare_device(raw, table, geom))
    H, W = 16 * geom[0], 16 * geom[1]
    f = 0
    for v, s, n in clips:
        f0, pos, size = (int(store.loc[k][v]) for k in ("fid", "pos", "size"))
        ref = ffmpeg_yuv(bytes(store.bufs[f0][pos : pos + size]), 170, 96)
        for k in range(s, s + n):
            assert np.array_equal(Y[f, :96, :170].cpu().numpy(), ref[k][0]), (v, k)
            assert np.array_equal(U[f, :48, :85].cpu().numpy(), ref[k][1]), (v, k)
            f += 1
    assert (H, W) == (96, 176)
