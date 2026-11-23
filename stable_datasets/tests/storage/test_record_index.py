import json
import multiprocessing as mp
import os

import numpy as np
import pytest

from stable_datasets.storage import record_index
from stable_datasets.storage.record_index import RecordIndex


def _columns(n=10):
    return {"fid": np.arange(n) % 3, "pos": np.arange(n) * 100, "h": np.full(n, 7, np.int32)}


def test_write_open_lookup(tmp_path):
    root = str(tmp_path / "idx")
    off = np.array([0, 2, 2, 5, 6, 6, 6, 9, 10, 10, 12])
    values = np.arange(12, dtype=np.uint16) * 3
    RecordIndex.write(root, ["a", "b", "c"], _columns(), ragged=("seg", off, values), store="s")
    idx = RecordIndex(root)
    assert len(idx) == 10 and idx.files == ["a", "b", "c"] and idx.meta["store"] == "s"
    assert idx.pos[4] == 400
    c = idx.lookup([3, 0, 1, 7])
    assert c["fid"].tolist() == [0, 0, 1, 1] and c["n"].tolist() == [1, 2, 0, 1]
    assert c["seg"].tolist() == [15, 0, 3, 27] and c["seg_start"].tolist() == [0, 1, 3, 3]
    c = idx.lookup([3, 7], gather=False)
    assert c["seg"][c["seg_start"]].tolist() == [15, 27]
    assert not any(p.name.startswith("idx.tmp") for p in tmp_path.iterdir())


def test_write_rejects_bad_columns(tmp_path):
    with pytest.raises(ValueError, match="different lengths"):
        RecordIndex.write(str(tmp_path / "a"), [], {"x": np.zeros(2), "y": np.zeros(3)})
    with pytest.raises(ValueError, match="rows \\+ 1"):
        RecordIndex.write(str(tmp_path / "b"), [], {"x": np.zeros(2)}, ragged=("r", [0, 1], np.zeros(1)))
    assert not list(tmp_path.iterdir())


def test_ragged_values_from_a_file(tmp_path):
    path = tmp_path / "values.bin"
    np.arange(4, dtype=np.uint16).tofile(path)
    root = str(tmp_path / "idx")
    RecordIndex.write(root, [], {"x": np.zeros(2)}, ragged=("seg", [0, 1, 4], str(path)), ragged_dtype="<u2")
    assert RecordIndex(root).lookup([1])["seg"].tolist() == [1, 2, 3]
    assert not path.exists()


def _build_counted(root, counter):
    with open(counter, "a") as f:
        f.write("x")
    RecordIndex.write(root, ["f"], _columns(4))


def _open_shared(base, counter, q):
    idx = RecordIndex.shared("key-1", lambda root: _build_counted(root, counter), 1000, base=base, owner="store")
    q.put(int(idx.pos[3]))


def test_shared_is_built_once_by_concurrent_processes(tmp_path):
    base, counter = str(tmp_path / "base"), str(tmp_path / "count")
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_open_shared, args=(base, counter, q)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0] * 6
    assert sorted(q.get(timeout=5) for _ in procs) == [300] * 6
    assert open(counter).read() == "x"
    (idx_dir,) = [d for d in os.listdir(base) if not d.endswith(".lock")]
    meta = json.load(open(f"{base}/{idx_dir}/meta.json"))
    assert meta["owner"] == "store" and "build_s" in meta


def test_a_new_index_of_the_owner_removes_the_old_one(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_INDEX_DIR", str(tmp_path))
    a = RecordIndex.shared("v1", lambda r: RecordIndex.write(r, [], _columns(2)), 10, owner="store")
    b = RecordIndex.shared("v2", lambda r: RecordIndex.write(r, [], _columns(3)), 10, owner="store")
    assert not os.path.exists(a.root) and len(b) == 3
    again = RecordIndex.shared("v2", lambda r: pytest.fail("built twice"), 10, owner="store")
    assert again.root == b.root


def test_an_index_of_another_format_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_INDEX_DIR", str(tmp_path))
    first = RecordIndex.shared("k", lambda r: RecordIndex.write(r, [], _columns(2)), 10)
    meta = json.load(open(f"{first.root}/meta.json"))
    meta["version"] = record_index.VERSION - 1
    json.dump(meta, open(f"{first.root}/meta.json", "w"))
    with pytest.raises(ValueError, match="format"):
        RecordIndex(first.root)
    rebuilt = RecordIndex.shared("k", lambda r: RecordIndex.write(r, [], _columns(5)), 10)
    assert rebuilt.root == first.root and len(rebuilt) == 5
