"""Differential fuzzing of codec.to_restart's fast translator (framestore/csrc/jpeg_restart.c, fast_one) against libjpeg's
re-code (restart_one, the same bytes as jpegtran -restart 1B): for every input, valid or corrupt, both give the
same status and the same bytes (or, where they differ, in padding blocks only: the same pixels), and a re-coded frame
decodes to the same pixels as the original.

    pytest tests/framestore/test_restart_fuzz.py                          # 200 examples per test
    FUZZ_EXAMPLES=20000 pytest tests/framestore/test_restart_fuzz.py      # a longer run
    # memory errors in the C code (macOS; on Linux preload libasan.so instead):
    NATIVE_CFLAGS="-O1 -g -fsanitize=address -fno-omit-frame-pointer" \\
    DYLD_INSERT_LIBRARIES=$(cc -print-file-name=libclang_rt.asan_osx_dynamic.dylib) pytest tests/framestore/test_restart_fuzz.py
"""

import io
import os
import shutil
import subprocess

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from stable_datasets.framestore import codec


PIL = pytest.importorskip("PIL.Image")


def frames(blobs):
    off = np.concatenate(([0], np.cumsum([len(b) for b in blobs]))).astype(np.int64)
    return np.frombuffer(b"".join(blobs), np.uint8).copy(), off


def recode(blobs, fast):
    out, off, status = codec.to_restart(*frames(blobs), fast=fast)
    return [out[off[i] : off[i + 1]].tobytes() for i in range(len(blobs))], status.tolist()


def assert_same(blobs):
    fast, fs = recode(blobs, True)
    slow, ss = recode(blobs, False)
    if (fs != ss or fast != slow) and os.environ.get("FUZZ_DUMP"):  # FUZZ_DUMP=dir: keep failing inputs
        os.makedirs(os.environ["FUZZ_DUMP"], exist_ok=True)
        for i, b in enumerate(blobs):
            open(os.path.join(os.environ["FUZZ_DUMP"], f"{hash(tuple(blobs)) & 0xFFFFFFFF:08x}-{i}.jpg"), "wb").write(
                b
            )
    assert fs == ss
    for i, (a, b) in enumerate(zip(fast, slow)):
        if a != b:  # allowed only in padding blocks (see framestore/csrc/jpeg_restart.c), which no decoder shows
            pa, pb = pixels_of(a), pixels_of(b)
            assert pa is not None and pa == pb, f"frame {i}: fast translator decodes differently from libjpeg's"
    return fast, fs


def pixels_of(blob):
    try:
        return np.asarray(PIL.open(io.BytesIO(blob)).convert("RGB")).tobytes()
    except Exception:
        return None


# ---- images ----------------------------------------------------------------------------------------------------
@st.composite
def pixels(draw):
    h, w = draw(st.integers(1, 72)), draw(st.integers(1, 72))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    kind = draw(st.sampled_from(["noise", "flat", "gradient", "blocks", "extremes"]))
    if kind == "noise":
        a = rng.integers(0, 256, (h, w, 3))
    elif kind == "flat":
        a = np.broadcast_to(rng.integers(0, 256, 3), (h, w, 3))
    elif kind == "gradient":
        y, x = np.mgrid[0:h, 0:w]
        a = np.stack([(x * rng.uniform(0, 8) + y * rng.uniform(0, 8) + c * 40) % 256 for c in range(3)], -1)
    elif kind == "blocks":  # 8x8 blocks alternating black and white: the largest DC differences
        y, x = np.mgrid[0:h, 0:w]
        a = np.repeat((((y // 8 + x // 8) % 2) * 255)[..., None], 3, -1)
    else:  # a checkerboard of single pixels: the largest AC coefficients
        y, x = np.mgrid[0:h, 0:w]
        a = np.repeat((((y + x) % 2) * 255)[..., None], 3, -1)
    return np.ascontiguousarray(a, np.uint8)


@st.composite
def jpegs(draw):
    a = draw(pixels())
    mode = draw(st.sampled_from(["RGB", "RGB", "RGB", "L", "CMYK", "YCbCr"]))
    kw = {"quality": draw(st.integers(1, 100))}
    if mode in ("RGB", "YCbCr"):
        kw["subsampling"] = draw(st.sampled_from([0, 1, 2]))  # 4:4:4, 4:2:2, 4:2:0
    if draw(st.booleans()):
        kw["optimize"] = True  # image-specific Huffman tables: libjpeg's path
    if draw(st.integers(0, 4)) == 0:
        kw["progressive"] = True
    if draw(st.booleans()):
        kw["comment"] = draw(st.binary(min_size=0, max_size=64))
    if draw(st.integers(0, 3)) == 0:
        kw["exif"] = b"Exif\x00\x00" + draw(st.binary(min_size=0, max_size=64))
    if draw(st.integers(0, 3)) == 0:
        kw["restart_marker_blocks"] = draw(st.integers(1, 8))  # already restart-coded, another interval
    b = io.BytesIO()
    PIL.fromarray(a).convert(mode).save(b, "JPEG", **kw)
    return b.getvalue()


def scan_start(blob):
    """Offset of the first SOS marker, found by walking the header's segments (a comment may hold FF DA)."""
    i = 2
    while blob[i + 1] != 0xDA:
        i += 2 + int.from_bytes(blob[i + 2 : i + 4], "big")
    return i


def decode(blob):
    return np.asarray(PIL.open(io.BytesIO(blob)).convert("RGB"))


@given(st.lists(jpegs(), min_size=1, max_size=4))
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_valid_frames(blobs):
    out, status = assert_same(blobs)
    for src, dst, s in zip(blobs, out, status):
        assert s == 0
        assert decode(dst).tobytes() == decode(src).tobytes(), "re-coding changed the pixels"
        assert b"\xff\xdd\x00\x04\x00\x01" in dst[: scan_start(dst)], "no DRI 1 before the scan"


# ---- corrupt frames ----------------------------------------------------------------------------------------------
@st.composite
def mutated(draw):
    b = bytearray(draw(jpegs()))
    for _ in range(draw(st.integers(1, 4))):
        op = draw(st.sampled_from(["flip", "byte", "truncate", "insert", "delete", "marker", "length"]))
        i = draw(st.integers(0, len(b) - 1)) if b else 0
        if not b:
            break
        if op == "flip":
            b[i] ^= 1 << draw(st.integers(0, 7))
        elif op == "byte":
            b[i] = draw(st.integers(0, 255))
        elif op == "truncate":
            del b[i:]
        elif op == "insert":
            b[i:i] = draw(st.binary(min_size=1, max_size=8))
        elif op == "delete":
            del b[i : i + draw(st.integers(1, 8))]
        elif op == "marker":  # a stray marker: RSTn, EOI, SOS, DHT, fill bytes
            b[i:i] = bytes([0xFF, draw(st.sampled_from([0xD0, 0xD3, 0xD7, 0xD9, 0xDA, 0xC4, 0xFF, 0x00, 0x01]))])
        else:  # a segment length, which the parser trusts to skip
            j = bytes(b).find(b"\xff", i)
            if 0 <= j < len(b) - 3:
                b[j + 2], b[j + 3] = draw(st.integers(0, 255)), draw(st.integers(0, 255))
    return bytes(b)


@given(st.lists(mutated(), min_size=1, max_size=4))
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_corrupt_frames(blobs):
    assert_same(blobs)


@given(st.binary(min_size=0, max_size=512))
@example(b"")
@example(b"\xff\xd8")
@example(b"\xff\xd8\xff\xd9")
@example(b"\xff\xd8\xff\xda\x00\x02")
def test_arbitrary_bytes(blob):
    assert_same([blob])


# ---- encoders other than Pillow's ---------------------------------------------------------------------------------
@pytest.mark.skipif(not shutil.which("jpegtran"), reason="needs jpegtran (libjpeg-turbo)")
@given(
    jpegs(),
    st.sampled_from(
        [
            ["-arithmetic"],
            ["-restart", "1"],
            ["-restart", "2B"],
            ["-optimize"],
            ["-progressive"],
            ["-grayscale"],
            ["-copy", "all"],
            ["-rotate", "90"],
            ["-crop", "16x16+8+8"],
        ]
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_jpegtran_variants(blob, args):
    r = subprocess.run(["jpegtran", *args], input=blob, capture_output=True)
    assume(r.returncode == 0 and r.stdout)
    assert_same([r.stdout])


# ---- the fast path is taken -------------------------------------------------------------------------------------
@given(pixels(), st.sampled_from(["RGB", "L"]), st.integers(1, 100), st.sampled_from([0, 1, 2]), st.booleans())
def test_fast_path_handles_standard_frames(a, mode, quality, subsampling, comment):
    """Baseline frames with the standard tables (most encoders' default) are re-coded by the fast path itself:
    a fast path that sent everything to libjpeg would pass the tests above too."""
    b = io.BytesIO()
    kw = {"quality": quality, "subsampling": subsampling} if mode == "RGB" else {"quality": quality}
    PIL.fromarray(a).convert(mode).save(b, "JPEG", **kw, **({"comment": b"note"} if comment else {}))
    buf, off = frames([b.getvalue()])
    _, _, status = codec._many(codec._restart().fast_restart_many, buf, off)
    assert status.tolist() == [0]


# ---- without the C re-coder ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not shutil.which("jpegtran"), reason="needs jpegtran (libjpeg-turbo)")
@given(st.lists(jpegs(), min_size=1, max_size=3))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_jpegtran_fallback_matches(blobs):
    """Where the C re-coder cannot be built, to_restart runs jpegtran: the same bytes and status as libjpeg's path."""
    from unittest import mock

    want = recode(blobs, fast=False)

    def unbuildable():
        raise OSError("no C compiler")

    with mock.patch.object(codec, "_restart", unbuildable), pytest.warns(RuntimeWarning, match="jpegtran"):
        from stable_datasets.framestore import _native

        _native._warned.clear()
        got = recode(blobs, fast=True)
    assert got == want
