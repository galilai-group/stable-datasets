from __future__ import annotations

import numpy as np
import pyarrow as pa

from stable_datasets.backends._indexed_arrow_table import IndexedArrowTable


def _make_chunked_table() -> pa.Table:
    batches = [
        pa.record_batch({"x": [0, 1, 2], "y": ["a", "b", "c"]}),
        pa.record_batch({"x": [3, 4], "y": ["d", "e"]}),
        pa.record_batch({"x": [5, 6, 7], "y": ["f", "g", "h"]}),
    ]
    return pa.Table.from_batches(batches)


def test_fast_gather_preserves_order_and_duplicates_single_batch():
    table = pa.Table.from_batches([pa.record_batch({"x": [0, 1, 2, 3]})])
    indexed = IndexedArrowTable(table)

    gathered = indexed.fast_gather(np.array([3, 1, 3, 0], dtype=np.int64))

    assert gathered.column("x").to_pylist() == [3, 1, 3, 0]


def test_fast_gather_preserves_order_across_batches():
    indexed = IndexedArrowTable(_make_chunked_table())

    gathered = indexed.fast_gather([6, 1, 4, 0, 7, 3])

    assert gathered.column("x").to_pylist() == [6, 1, 4, 0, 7, 3]
    assert gathered.column("y").to_pylist() == ["g", "b", "e", "a", "h", "d"]


def test_fast_slice_matches_expected_rows():
    indexed = IndexedArrowTable(_make_chunked_table())

    sliced = indexed.fast_slice(2, 4)

    assert sliced.column("x").to_pylist() == [2, 3, 4, 5]
    assert sliced.column("y").to_pylist() == ["c", "d", "e", "f"]


def test_fast_gather_empty_preserves_schema():
    indexed = IndexedArrowTable(_make_chunked_table())

    gathered = indexed.fast_gather([])

    assert gathered.num_rows == 0
    assert gathered.schema == indexed.schema
