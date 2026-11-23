"""Small tables of numpy arrays in Lance (sampling indexes, plans), so code outside storage needs no Arrow."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def write_arrays(path: str, columns: Mapping[str, np.ndarray]) -> None:
    """One Lance dataset at ``path`` from equal-length arrays; 2-D arrays become fixed-size list columns."""
    import lance
    import pyarrow as pa

    cols = {}
    for name, values in columns.items():
        values = np.asarray(values)
        if values.ndim == 2:
            cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])
        else:
            cols[name] = pa.array(values)
    lance.write_dataset(pa.table(cols), path, mode="overwrite")


def read_arrays(path: str) -> dict[str, np.ndarray]:
    """The columns of a dataset written by :func:`write_arrays`, as numpy arrays."""
    import lance
    import pyarrow as pa

    table = lance.dataset(path).to_table()
    out = {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            out[name] = col.values.to_numpy(zero_copy_only=False).reshape(len(col), col.type.list_size)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    return out
