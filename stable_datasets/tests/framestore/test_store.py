"""stable_datasets.framestore end to end on small stores (CPU): conversion is lossless and keeps every other column, every read
policy returns the same bytes, the per-node index matches lance, and an in-place conversion killed at any risky step
resumes to the same result.

    pytest tests/framestore/test_store.py
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest


lance = pytest.importorskip("lance")
pytest.importorskip("numba")
pytest.importorskip("simplejpeg")

from stable_datasets.framestore import codec
from stable_datasets.framestore.convert import convert_store
from stable_datasets.framestore.decoders import Decoder
from stable_datasets.framestore.store import FrameStore

from ._stores import make_store


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPTIONS = {"gpu": "none", "batch": 32, "memory": "2GB", "nice": 0}  # CPU verification; several batches per fragment


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """(path, original frames) of a converted 4:2:0 store."""
    path = str(tmp_path_factory.mktemp("store") / "frames.lance")
    blobs = make_store(path)
    convert_store(path, **OPTIONS)
    return path, blobs


def all_frames(store):
    fid, pos, size, _, _ = store.descriptors(np.arange(store.ds.count_rows()))
    return [store.frame(f, p, n).tobytes() for f, p, n in zip(fid, pos, size)]


def expected(blobs):
    out, off, st = codec.to_restart(*codec_frames(blobs))
    assert (st == 0).all()
    return [out[off[i] : off[i + 1]].tobytes() for i in range(len(blobs))]


def codec_frames(blobs):
    off = np.concatenate(([0], np.cumsum([len(b) for b in blobs]))).astype(np.int64)
    return np.frombuffer(b"".join(blobs), np.uint8).copy(), off


def test_conversion_is_lossless_and_keeps_columns(converted):
    path, blobs = converted
    ds = lance.dataset(path)
    assert ds.count_rows() == len(blobs) and "seg_len" in ds.schema.names
    t = ds.to_table(columns=["episode_idx", "h", "w"])
    assert t.column("episode_idx").to_numpy().tolist() == np.repeat(np.arange(6), 40).tolist()
    new = all_frames(FrameStore(path))
    assert new == expected(blobs), "stored bytes are not the restart re-coding of the originals"
    dec = Decoder("cpu")
    for a, b in zip(dec(blobs[::17]), dec(new[::17])):
        assert bool((a == b).all()), "a converted frame decodes differently"


@pytest.mark.parametrize("read", ["mmap", "pread", "spans", "ranges"])
def test_read_policies_agree(converted, read):
    path, blobs = converted
    want = expected(blobs)
    store = FrameStore(path, read=read)
    rows = store.windows(np.random.default_rng(0), clips=8, frames=16, stride=2)
    out = np.empty(sum(len(want[r]) for r in rows) + 16, np.uint8)
    off, _, _ = store.copy_frames(rows, out)
    assert [out[off[i] : off[i + 1]].tobytes() for i in range(len(rows))] == [want[r] for r in rows]


@pytest.mark.parametrize("kind", ["large_binary", "binary"])
def test_take_layout_reads_the_same(tmp_path, converted, kind):
    """A store whose frame column is not blob-encoded is read by lance take, into the same interface."""
    import pyarrow as pa

    path, blobs = converted
    plain = str(tmp_path / "plain.lance")
    lance.write_dataset(
        pa.table(
            {
                "episode_idx": pa.array(np.repeat(np.arange(6), 40), pa.int64()),
                "frame": pa.array(expected(blobs), getattr(pa, kind)()),
                "h": pa.array(np.full(len(blobs), 72), pa.int32()),
                "w": pa.array(np.full(len(blobs), 104), pa.int32()),
            }
        ),
        plain,
    )
    store = FrameStore(plain)
    assert store.read == "take"
    rows = np.array([3, 50, 51, 200])
    fid, pos, size, _, _ = store.descriptors(rows)
    assert [store.frame(f, p, n).tobytes() for f, p, n in zip(fid, pos, size)] == [expected(blobs)[r] for r in rows]


def test_shared_index_matches_lance(converted, tmp_path, monkeypatch):
    path, _ = converted
    monkeypatch.setenv("FRAMESTORE_INDEX_DIR", str(tmp_path / "index"))
    from stable_datasets.framestore.index import FrameIndex

    idx = FrameIndex.shared(path, workers=1)
    plain, indexed = FrameStore(path), FrameStore(path, index="shared")
    rows = np.random.default_rng(1).integers(0, len(idx), 100)
    a, b = plain.take_meta(rows), indexed.take_meta(rows)
    for k in ("fid", "pos", "size", "h", "w", "seg", "nseg"):  # fid: both index the store's files in one order
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_unsupported_frames_refused(tmp_path):
    """A CMYK frame cannot be re-coded for splicing: refused, and nothing is written, unless keep_unsupported."""
    import io

    import pyarrow as pa
    from PIL import Image

    from stable_datasets.framestore.convert import ConvertError

    b = io.BytesIO()
    Image.new("CMYK", (32, 32), (10, 20, 30, 40)).save(b, "JPEG")
    path = str(tmp_path / "cmyk.lance")
    lance.write_dataset(
        pa.table(
            {
                "episode_idx": pa.array([0, 0], pa.int64()),
                "frame": pa.array([b.getvalue()] * 2, pa.large_binary()),
                "h": pa.array([32, 32], pa.int32()),
                "w": pa.array([32, 32], pa.int32()),
            }
        ),
        path,
    )
    with pytest.raises(ConvertError, match="cannot be converted"):
        convert_store(path, **OPTIONS)
    assert "seg_len" not in lance.dataset(path).schema.names
    convert_store(path, keep_unsupported=True, **OPTIONS)
    assert lance.dataset(path).to_table(columns=["seg_len"]).column(0).null_count == 2


CRASH = """
import os, shutil, sys
sys.path.insert(0, {root!r})
import lance
from stable_datasets.framestore import convert
when, n = {when!r}, [0]
real_write, real_remove, real_rmtree = lance.write_dataset, os.remove, shutil.rmtree
def write(reader, uri, *a, **k):
    n[0] += 1
    if when == "create" and n[0] == 1:  # the directory exists, nothing committed
        os.makedirs(os.path.join(uri, "data"))
        open(os.path.join(uri, "data", "x.lance"), "w").write("partial")
        os._exit(1)
    if when == "before" and n[0] == 3:
        os._exit(1)
    r = real_write(reader, uri, *a, **k)
    if when == "after" and n[0] == 2:
        os._exit(1)
    return r
def remove(p, *a, **k):
    if when == "progress" and p.endswith(".progress.json"):
        os._exit(1)
    return real_remove(p, *a, **k)
def rmtree(p, *a, **k):
    if when == "rmtree" and p == {store!r}:
        real_rmtree(os.path.join(p, "_versions"))
        os._exit(1)
    return real_rmtree(p, *a, **k)
lance.write_dataset, os.remove, shutil.rmtree = write, remove, rmtree
convert.main([{store!r}, "--gpu", "none", "--batch", "32", "--memory", "2GB", "--nice", "0"])
"""


@pytest.mark.parametrize("when", ["create", "after", "before", "rmtree", "progress"])
def test_crash_and_resume(tmp_path, converted, when):
    """Killed at a risky step of an in-place conversion, then run again: the same store as a clean conversion, and
    nothing left over (the cases tla/ConvertResume.tla checks)."""
    ref, _ = converted
    path = str(tmp_path / "frames.lance")
    make_store(path)
    r = subprocess.run(
        [sys.executable, "-c", CRASH.format(root=ROOT, when=when, store=path)], capture_output=True, text=True
    )
    assert r.returncode == 1, r.stdout + r.stderr
    convert_store(path, **OPTIONS)
    cols = ["episode_idx", "h", "w", "seg_len"]
    assert lance.dataset(path).to_table(columns=cols).equals(lance.dataset(ref).to_table(columns=cols))
    assert all_frames(FrameStore(path)) == all_frames(FrameStore(ref))
    assert not [p for p in (path + ".converting", path + ".converting.progress.json") if os.path.exists(p)]
    shutil.rmtree(path)
