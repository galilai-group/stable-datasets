"""stable_datasets.framestore.gpu_jpeg: every MCU of restart-coded frames decodes, on its own, to the quantized DCT coefficients
libjpeg decodes from the whole frame (coefs_ref.c), for 4:2:0, 4:2:2, 4:4:4 and gray frames at low to high quality,
including optimized (non-standard) Huffman tables; the IDCT and colour stages then give libjpeg-turbo's pixels exactly
(islow IDCT, plain upsampling: simplejpeg's fastdct=False, fastupsample=True); and the Triton kernels equal the numpy
references stage by stage (GPU, skipped without CUDA).

    pytest tests/framestore/test_gpu_jpeg.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import ctypes
import io
import os

import numpy as np
import pytest


Image = pytest.importorskip("PIL.Image")
pytest.importorskip("numba")

from stable_datasets.framestore import _native, codec
from stable_datasets.framestore.gpu_jpeg import (
    color_np,
    decode_mcus_np,
    frames_from_mcus,
    header,
    idct_matrix_np,
    idct_np,
    mcu_offsets,
    pixels_np,
)


@pytest.fixture(scope="module")
def ref():
    try:
        lib = _native.load(os.path.join(os.path.dirname(__file__), "coefs_ref.c"), libs=["jpeg"])
    except OSError as e:
        pytest.skip(f"needs a C compiler and libjpeg: {e}")
    lib.read_coefs.restype = ctypes.c_int64
    lib.read_coefs.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]

    def read(j):
        out = np.zeros(1 << 22, np.int16)
        k = lib.read_coefs(j.ctypes.data, len(j), out.ctypes.data, len(out))
        assert k > 0, k
        return out[: k * 64].reshape(k, 64)

    return read


def frames(n=3, h=136, w=200, seed=0):
    """Photo-like frames: gradients, hard edges and noise rising frame to frame (long codes, stuffed 0xFF bytes)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    out = []
    for i in range(n):
        x = np.stack([(xx * (i + 1) + yy) % 256, (yy * 2 + 40 * i) % 256, ((xx // 16 + yy // 16) % 2) * 200], -1)
        out.append(np.clip(x + rng.normal(0, 12 * i, x.shape), 0, 255).astype(np.uint8))
    return out


def restart_coded(imgs, gray=False, **save):
    """Pillow JPEGs re-coded with a restart marker after every MCU, and their MCU segment lengths."""
    blobs = []
    for x in imgs:
        b = io.BytesIO()
        Image.fromarray(x[..., 0] if gray else x).save(b, "JPEG", **save)
        blobs.append(np.frombuffer(b.getvalue(), np.uint8))
    off = np.concatenate(([0], np.cumsum([len(b) for b in blobs]))).astype(np.int64)
    buf, off, status = codec.to_restart(np.concatenate(blobs), off)
    assert (status >= 0).all()
    info = codec.probe(buf, off)
    seg, seg_off, ok = codec.segment_lengths(buf, off, info)
    assert ok.all()
    return [(np.asarray(buf[off[f] : off[f + 1]]), seg[seg_off[f] : seg_off[f + 1]]) for f in range(len(imgs))]


CASES = {
    "4:2:0 q50": {"quality": 50, "subsampling": 2},
    "4:2:0 q90": {"quality": 90, "subsampling": 2},
    "4:2:0 q98": {"quality": 98, "subsampling": 2},
    "4:2:2 q90": {"quality": 90, "subsampling": 1},
    "4:4:4 q90": {"quality": 90, "subsampling": 0},
    "4:2:0 q90 optimized tables": {"quality": 90, "subsampling": 2, "optimize": True},
    "gray q90": {"quality": 90, "gray": True},
}


@pytest.mark.parametrize("case", CASES)
def test_each_mcu_decodes_to_libjpegs_coefficients(ref, case):
    save = dict(CASES[case])
    gray = save.pop("gray", False)
    for j, seg in restart_coded(frames(), gray=gray, **save):
        hdr = header(j.tobytes())
        got = decode_mcus_np(j, mcu_offsets(j, seg, hdr["sos_end"]), hdr)
        assert np.array_equal(got, ref(j).reshape(got.shape))


@pytest.mark.parametrize("case", CASES)
def test_frames_decode_to_libjpeg_turbos_pixels(case):
    simplejpeg = pytest.importorskip("simplejpeg")
    save = dict(CASES[case])
    gray = save.pop("gray", False)
    for j, seg in restart_coded(frames(), gray=gray, **save):
        hdr = header(j.tobytes())
        pix = color_np(idct_np(decode_mcus_np(j, mcu_offsets(j, seg, hdr["sos_end"]), hdr), hdr), hdr)
        got = frames_from_mcus(pix, *hdr["mcus"], *hdr["size"])
        want = simplejpeg.decode_jpeg(
            j.tobytes(), colorspace="GRAY" if gray else "RGB", fastdct=False, fastupsample=True
        )
        assert np.array_equal(got, want.reshape(got.shape))


@pytest.mark.parametrize("case", CASES)
def test_fused_formulation_is_exact(case):
    """The IDCT as integer matrices (IDCT_M), and chroma upsampled through row/column-selected matrices (the fused
    kernel's formulation), equal the butterfly IDCT and the gathered upsampling bit for bit."""
    save = dict(CASES[case])
    gray = save.pop("gray", False)
    for j, seg in restart_coded(frames(), gray=gray, **save):
        hdr = header(j.tobytes())
        c = decode_mcus_np(j, mcu_offsets(j, seg, hdr["sos_end"]), hdr)
        assert np.array_equal(idct_matrix_np(c, hdr), idct_np(c, hdr))
        assert np.array_equal(pixels_np(c, hdr), color_np(idct_np(c, hdr), hdr))


def test_mcus_decode_in_any_order(ref):
    """Each MCU depends only on its own bytes: decoding a shuffled subset gives those MCUs' coefficients."""
    (j, seg), *_ = restart_coded(frames(1, seed=3), quality=90)
    hdr = header(j.tobytes())
    off = mcu_offsets(j, seg, hdr["sos_end"])
    pick = np.random.default_rng(0).permutation(len(off))[: len(off) // 3]
    want = ref(j).reshape(len(off), hdr["nblocks"], 64)[pick]
    assert np.array_equal(decode_mcus_np(j, off[pick], hdr), want)


@pytest.mark.parametrize("case", CASES)
def test_triton_kernels_equal_the_references(case):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")
    from stable_datasets.framestore.gpu_jpeg import color, decode_mcus, decode_rgb, idct

    save = dict(CASES[case])
    gray = save.pop("gray", False)
    for j, seg in restart_coded(frames(), gray=gray, **save):
        hdr = header(j.tobytes())
        off = mcu_offsets(j, seg, hdr["sos_end"])
        data = torch.from_numpy(np.concatenate([j, np.zeros(16, np.uint8)])).cuda()
        coefs = decode_mcus(data, torch.from_numpy(off).cuda(), hdr)
        want = decode_mcus_np(j, off, hdr)
        assert np.array_equal(coefs.cpu().numpy(), want)
        samples = idct(coefs, hdr)
        assert np.array_equal(samples.cpu().numpy(), idct_np(want, hdr))
        assert np.array_equal(color(samples, hdr).cpu().numpy(), color_np(idct_np(want, hdr), hdr))
        assert np.array_equal(decode_rgb(data, torch.from_numpy(off).cuda(), hdr).cpu().numpy(), pixels_np(want, hdr))


def test_decode_buffer_whole_frames():
    """Frames of different sizes and tables, back to back, decode on the GPU to libjpeg-turbo's pixels (MCU starts
    found from the restart markers; one launch per table set)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")
    simplejpeg = pytest.importorskip("simplejpeg")
    from stable_datasets.framestore.decoders import Decoder

    blobs = [j for j, _ in restart_coded(frames(3), quality=90)]
    blobs += [j for j, _ in restart_coded(frames(2, h=72, w=120, seed=5), quality=50)]
    blobs += [j for j, _ in restart_coded(frames(2, h=100, w=64, seed=7), quality=90, subsampling=0)]
    got = Decoder("triton")(blobs)
    for j, g in zip(blobs, got):
        want = simplejpeg.decode_jpeg(j.tobytes(), colorspace="RGB", fastdct=False, fastupsample=True)
        assert np.array_equal(g.permute(1, 2, 0).cpu().numpy(), want)
