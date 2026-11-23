"""The Lance blob layout that lance_files.py assumes: if a Lance release changes it, this fails."""

import numpy as np
import pytest


lance = pytest.importorskip("lance")
pa = pytest.importorskip("pyarrow")

from stable_datasets.sources import LocalSource  # noqa: E402
from stable_datasets.storage.lance_files import blob_files, blob_locations  # noqa: E402


def test_blob_bytes_are_where_the_layout_says(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [10, 70_000, 5_000_000, 0, 3_000, 200_000, 64 * 1024, 1]  # inline, packed side, own file, empty
    blobs = [rng.integers(0, 256, n, dtype=np.uint8).tobytes() for n in sizes] * 3
    field = pa.field("blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"})
    schema = pa.schema([pa.field("i", pa.int64()), field])
    table = pa.table({"i": np.arange(len(blobs)), "blob": pa.array(blobs, pa.large_binary())}, schema=schema)
    path = str(tmp_path / "blobs.lance")
    lance.write_dataset(table, path, max_rows_per_file=10)
    ds = lance.dataset(path)
    frags = ds.get_fragments()
    assert len(frags) == 3
    files, data_fid, side_fid = blob_files(path, frags)
    assert len(files) > len(frags), "large blobs should live in side files"
    rows = np.arange(len(blobs))
    d = ds.take(rows, columns=["blob"]).column("blob").combine_chunks()
    assert pa.types.is_struct(d.type), "a blob column's take should return descriptors"
    frag = np.searchsorted(np.cumsum([0] + [f.count_rows() for f in frags]), rows, side="right") - 1
    fid, pos, size = blob_locations(d, frag, data_fid, side_fid)
    assert size.tolist() == [len(b) for b in blobs]
    buf, start = LocalSource(files).read_ranges(fid, pos, size)
    assert [buf[s : s + n].tobytes() for s, n in zip(start, size)] == blobs
