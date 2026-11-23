import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stable_datasets.codecs import Chunks, Codec, RestartJpeg, Units, get, grid_cover
from stable_datasets.codecs.chunks import chunk_boxes, encode


def _brute_cover(shape, unit, box):
    ndim = len(shape)
    ids = []
    for cid, c in enumerate(chunk_boxes(shape, unit)):
        lo, hi = np.maximum(c[:ndim], box[:ndim]), np.minimum(c[ndim:], box[ndim:])
        if (hi > lo).all():
            ids.append(cid)
    return ids


@st.composite
def grid_case(draw):
    ndim = draw(st.integers(1, 3))
    shape = [draw(st.integers(1, 40)) for _ in range(ndim)]
    unit = [draw(st.integers(1, 12)) for _ in range(ndim)]
    lo = [draw(st.integers(-5, s + 5)) for s in shape]
    hi = [draw(st.integers(-5, s + 5)) for s in shape]
    return shape, unit, lo + hi


@settings(max_examples=300, deadline=None)
@given(st.lists(grid_case(), min_size=1, max_size=4).filter(lambda cs: len({len(c[0]) for c in cs}) == 1))
def test_grid_cover_matches_brute_force(cases):
    shape = np.array([c[0] for c in cases])
    unit = np.array([c[1] for c in cases])
    boxes = np.array([c[2] for c in cases])
    units = grid_cover(shape, unit, boxes)
    assert len(units) == len(cases)
    for i, (s, u, b) in enumerate(cases):
        assert units.of(i).tolist() == _brute_cover(s, u, np.array(b))


def test_units_csr_helpers():
    u = Units.from_lists([[3, 4], [], [7]])
    assert u.offsets.tolist() == [0, 2, 2, 3] and u.sample().tolist() == [0, 0, 2] and u.of(2).tolist() == [7]


def test_registry():
    assert isinstance(get("jpeg-restart"), RestartJpeg) and isinstance(get("chunks"), Chunks)
    assert isinstance(get("chunks"), Codec)
    with pytest.raises(KeyError):
        get("png")


@pytest.mark.parametrize("compression", ["none", "zlib", "zstd"])
@pytest.mark.parametrize("dtype", ["int16", "uint8", "float32"])
def test_chunks_round_trip_any_region(compression, dtype):
    if compression == "zstd":
        pytest.importorskip("zstandard")
    rng = np.random.default_rng(0)
    vol = rng.integers(-1000, 1000, (37, 50, 23)).astype(dtype)
    blobs, meta = encode(vol, (16, 16, 8), compression=compression)
    assert len(blobs) == 3 * 4 * 3
    codec = Chunks()
    meta_arr = {"shape": np.array([meta["shape"]]), "chunk": np.array([meta["chunk"]])}
    for _ in range(30):
        lo = rng.integers(-3, [37, 50, 23])
        hi = lo + rng.integers(0, 30, 3)
        box = np.concatenate([lo, hi])
        units = codec.cover(meta_arr, [0], box[None])
        region = Chunks.decode_region({i: blobs[i] for i in units.ids}, meta, box)
        want = vol[tuple(slice(max(a, 0), max(min(b, s), 0)) for a, b, s in zip(lo, hi, vol.shape))]
        assert region.dtype == vol.dtype
        assert np.array_equal(region, want.reshape(region.shape))


def test_chunks_missing_chunk_and_bad_chunk_shape():
    blobs, meta = encode(np.zeros((8, 8), np.uint8), (4, 4), compression="none")
    with pytest.raises(KeyError, match="missing"):
        Chunks.decode_region({0: blobs[0]}, meta, (0, 0, 8, 8))
    with pytest.raises(ValueError):
        encode(np.zeros((8, 8)), (4,))
    with pytest.raises(ValueError, match="compression"):
        encode(np.zeros((8, 8)), (4, 4), compression="lz77")


@pytest.fixture(scope="module")
def jpegs():
    pytest.importorskip("numba")
    pytest.importorskip("PIL")
    from stable_datasets.tests.framestore.test_gpu_jpeg import CASES, frames, restart_coded

    out = {}
    for name, save in CASES.items():
        save = dict(save)
        gray = save.pop("gray", False)
        out[name] = restart_coded(frames(n=2, h=77, w=133), gray=gray, **save)
    return out


def _full(j, seg):
    from stable_datasets.framestore.gpu_jpeg import (
        color_np,
        decode_mcus_np,
        frames_from_mcus,
        header,
        idct_np,
        mcu_offsets,
    )

    hdr = header(j.tobytes())
    pix = color_np(idct_np(decode_mcus_np(j, mcu_offsets(j, seg, hdr["sos_end"]), hdr), hdr), hdr)
    return frames_from_mcus(pix, *hdr["mcus"], *hdr["size"])


BOXES = [(0, 0, 77, 133), (5, 9, 40, 70), (16, 16, 32, 32), (70, 120, 77, 133), (-4, -4, 3, 200), (30, 30, 31, 31)]


def test_jpeg_region_decode_equals_the_whole_frames_crop(jpegs):
    for name, frames_ in jpegs.items():
        for j, seg in frames_:
            full = _full(j, seg)
            for box in BOXES:
                y0, x0, y1, x1 = box
                want = full[max(y0, 0) : y1, max(x0, 0) : x1]
                got = RestartJpeg.decode_region(j, seg, box)
                assert np.array_equal(got, want), (name, box)


def test_jpeg_cover_and_unit_ranges_address_the_mcus(jpegs):
    from stable_datasets.framestore import codec as fcodec
    from stable_datasets.framestore.gpu_jpeg import header, mcu_offsets

    frames_ = jpegs["4:2:0 q90"] + jpegs["gray q90"]
    buf = np.concatenate([j for j, _ in frames_])
    off = np.concatenate(([0], np.cumsum([len(j) for j, _ in frames_])))
    info = fcodec.probe(buf, off)
    seg, seg_off, ok = fcodec.segment_lengths(buf, off, info)
    assert ok.all()
    meta = {k: info[k] for k in ("h", "w", "mcu_h", "mcu_w")}
    meta["sos_end"] = info["sos_end"] - off[:-1]  # probe reports it in the batch buffer
    record = np.arange(len(frames_)).repeat(len(BOXES))
    boxes = np.array(BOXES * len(frames_))
    units = RestartJpeg().cover(meta, record, boxes)
    start, size = RestartJpeg.unit_ranges(meta, seg, seg_off, record, units)
    for i, r in enumerate(record):
        j, s = frames_[r]
        hdr = header(j.tobytes())
        rows, cols = hdr["mcus"]
        mh, mw = int(meta["mcu_h"][r]), int(meta["mcu_w"][r])
        y0, x0, y1, x1 = boxes[i]
        want = [
            rr * cols + cc
            for rr in range(rows)
            for cc in range(cols)
            if max(rr * mh, y0) < min((rr + 1) * mh, y1, hdr["size"][0])
            and max(cc * mw, x0) < min((cc + 1) * mw, x1, hdr["size"][1])
        ]
        assert units.of(i).tolist() == want
        sl = slice(units.offsets[i], units.offsets[i + 1])
        assert np.array_equal(start[sl], mcu_offsets(j, s, hdr["sos_end"])[units.of(i)])
        assert np.array_equal(size[sl], s[units.of(i)])


def test_jpeg_whole_decode(jpegs):
    j, _ = jpegs["4:4:4 q90"][0]
    assert RestartJpeg.decode(j).shape == (77, 133, 3)
    g, _ = jpegs["gray q90"][0]
    assert RestartJpeg.decode(g).shape == (77, 133, 1)


def test_jpeg_decode_from_fetched_units_only(jpegs):
    from stable_datasets.framestore import codec as fcodec

    for name in ("4:2:0 q90", "4:2:2 q90", "gray q90"):
        j, seg = jpegs[name][0]
        off = np.array([0, len(j)])
        info = fcodec.probe(j, off)
        meta = {k: info[k] for k in ("h", "w", "mcu_h", "mcu_w", "sos_end")}
        full = _full(j, seg)
        for box in BOXES[:4]:
            units = RestartJpeg().cover(meta, [0], np.array([box]))
            start, size = RestartJpeg.unit_ranges(meta, seg, [0, len(seg)], [0], units)
            pieces = [j[a : a + n] for a, n in zip(start, size)]
            head = j[: RestartJpeg.header_size(meta, [0])[0]]
            row = {k: v[0] for k, v in meta.items()}
            got = RestartJpeg.decode_units(head, pieces, units.ids, row, box)
            y0, x0, y1, x1 = box
            assert np.array_equal(got, full[y0:y1, x0:x1]), (name, box)


def test_chunks_decode_from_one_blob_per_record():
    vol = np.arange(20 * 9 * 7, dtype=np.int16).reshape(20, 9, 7)
    blobs, meta = encode(vol, (8, 4, 7), compression="zlib")
    blob = b"".join(blobs)
    sizes = np.array([len(b) for b in blobs])
    m = {"shape": np.array([meta["shape"]]), "chunk": np.array([meta["chunk"]])}
    box = np.array([3, 2, 0, 17, 9, 5])
    units = Chunks().cover(m, [0], box[None])
    start, size = Chunks.unit_ranges(m, sizes, [0, len(sizes)], [0], units)
    pieces = [blob[a : a + n] for a, n in zip(start, size)]
    got = Chunks.decode_units(b"", pieces, units.ids, meta, box)
    assert np.array_equal(got, vol[3:17, 2:9, 0:5])
