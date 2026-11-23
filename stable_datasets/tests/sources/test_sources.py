import multiprocessing as mp
import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stable_datasets.sources import ByteSource, LocalSource, coalesce


ranges = st.lists(st.tuples(st.integers(0, 3), st.integers(0, 5000), st.integers(0, 700)), min_size=0, max_size=60)


@settings(max_examples=300, deadline=None)
@given(ranges, st.integers(0, 2000))
def test_coalesced_reads_cover_every_range(rs, gap):
    fid, off, size = (np.array([r[i] for r in rs], np.int64) for i in range(3))
    c = coalesce(fid, off, size, gap)
    assert len(c.read) == len(rs)
    if not rs:
        assert len(c.fid) == 0
        return
    # each range lies inside its read, in the same file
    assert (c.fid[c.read] == fid).all()
    assert (c.delta >= 0).all()
    assert (c.offset[c.read] + c.delta == off).all()
    assert (c.delta + size <= c.size[c.read]).all()
    # reads of one file are disjoint and more than gap apart, so no read could be merged further
    order = np.lexsort((c.offset, c.fid))
    f, a, n = c.fid[order], c.offset[order], c.size[order]
    same = f[1:] == f[:-1]
    assert (a[1:][same] > (a[:-1] + n[:-1])[same] + gap).all()
    # every read starts at a range and ends at one
    for i in range(len(c.fid)):
        mine = c.read == i
        assert off[mine].min() == c.offset[i]
        assert (off[mine] + size[mine]).max() == c.offset[i] + c.size[i]


def test_coalesce_merges_touching_and_near_ranges():
    c = coalesce([0, 0, 0, 1], [0, 10, 25, 10], [10, 5, 5, 5])
    assert c.fid.tolist() == [0, 0, 1] and c.offset.tolist() == [0, 25, 10] and c.size.tolist() == [15, 5, 5]
    c = coalesce([0, 0, 0, 1], [0, 10, 25, 10], [10, 5, 5, 5], gap=10)
    assert c.size.tolist() == [30, 5]


def test_coalesce_rejects_bad_ranges():
    with pytest.raises(ValueError, match="unequal"):
        coalesce([0, 1], [0], [1])
    with pytest.raises(ValueError, match=">= 0"):
        coalesce([0], [-1], [1])


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    data = [rng.integers(0, 256, n, dtype=np.uint8) for n in (10_000, 3_000, 1)]
    paths = []
    for i, d in enumerate(data):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(d.tobytes())
        paths.append(p)
    return paths, data


def _expected(data, fid, off, size):
    return [data[f][o : o + n].tobytes() for f, o, n in zip(fid, off, size)]


@pytest.mark.parametrize("gap", [0, 100, 1 << 20])
def test_read_ranges_returns_each_range(files, gap):
    paths, data = files
    rng = np.random.default_rng(1)
    fid = rng.integers(0, 2, 200)
    size = rng.integers(0, 300, 200)
    off = np.array([rng.integers(0, len(data[f]) - n + 1) for f, n in zip(fid, size)])
    source = LocalSource(paths, threads=4)
    assert isinstance(source, ByteSource)
    buf, start = source.read_ranges(fid, off, size, gap=gap)
    got = [buf[s : s + n].tobytes() for s, n in zip(start, size)]
    assert got == _expected(data, fid, off, size)
    if gap == 1 << 20:
        assert len(buf) == len(data[0]) or len(buf) <= len(data[0]) + len(data[1])


def test_read_into_places_ranges(files):
    paths, data = files
    source = LocalSource(paths)
    buf = np.zeros(64, np.uint8)
    source.read_into([1, 0, 2], [5, 100, 0], [8, 8, 1], buf, [0, 20, 40])
    assert buf[0:8].tobytes() == data[1][5:13].tobytes()
    assert buf[20:28].tobytes() == data[0][100:108].tobytes()
    assert buf[40] == data[2][0]
    with pytest.raises(ValueError, match="outside the buffer"):
        source.read_into([0], [0], [8], buf, [60])
    source.read_ranges([], [], [])


def test_reading_past_the_end_fails(files):
    paths, _ = files
    with pytest.raises(OSError, match="ended early"):
        LocalSource(paths).read_ranges([2], [0], [5])


def _child_read(source, q):
    buf, start = source.read_ranges([0], [10], [4])
    q.put(buf[start[0] : start[0] + 4].tobytes())


def test_survives_pickle_and_fork(files):
    paths, data = files
    source = LocalSource(paths)
    source.read_ranges([0], [0], [4])  # opens descriptors in this process
    clone = pickle.loads(pickle.dumps(source))
    assert clone.read_ranges([1], [0], [3])[0].tobytes() == data[1][:3].tobytes()
    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(target=_child_read, args=(source, q))
    p.start()
    p.join(30)
    assert p.exitcode == 0
    assert q.get(timeout=5) == data[0][10:14].tobytes()
    source.close()
