"""Where a Lance dataset keeps a blob column's bytes on disk: the only module that knows Lance's file layout.

Readers that skip Lance on the hot path (memory maps, preads, byte sources) locate blobs through this module. A
Lance release that changes the layout breaks test_lance_files, not the readers.

Layout (pylance 0.3x, blob encoding): every fragment has one data file, data/<uuid>.lance, holding blobs under
64 KB (descriptor kind 0); larger blobs live in its side files data/<uuid>/<blob_id>.blob (kind 1: packed; kind 2:
one blob per file), the blob_id written bit-reversed in binary in the file name.
"""

from __future__ import annotations

import glob
import os

import numpy as np


def blob_files(path, frags) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The files of a dataset's blobs in a fixed order: per fragment its data file, then its side files by blob_id.
    Returns (files, data file id per fragment, side file id [fragment, blob_id], -1 where there is none)."""
    files, data_fid, side = [], [], {}
    for i, frag in enumerate(frags):
        (df,) = frag.data_files()
        data_fid.append(len(files))
        files.append(f"{path}/data/{df.path}")
        found = {
            int(os.path.basename(b)[:-5][::-1], 2): b  # blob_id, written bit-reversed in the file name
            for b in glob.glob(f"{path}/data/{os.path.splitext(df.path)[0]}/*.blob")
        }
        for bid in sorted(found):
            side[i, bid] = len(files)
            files.append(found[bid])
    side_fid = np.full((len(frags), max([k[1] for k in side] + [0]) + 1), -1, np.int64)
    for (i, bid), f in side.items():
        side_fid[i, bid] = f
    return files, np.array(data_fid, np.int64), side_fid


def blob_locations(descriptors, frag, data_fid, side_fid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(file id, position, size) of blobs from their descriptors (a struct array of kind, position, size,
    blob_id, as a blob column's take or scan returns) and the fragment of each."""
    kind = descriptors.field("kind").to_numpy(zero_copy_only=False)
    if kind.max(initial=0) > 2:
        raise ValueError("unexpected blob descriptor kind: the Lance blob layout changed")
    frag = np.asarray(frag, np.int64)
    blob_id = descriptors.field("blob_id").to_numpy(zero_copy_only=False).astype(np.int64)
    side = side_fid[frag, np.minimum(blob_id, side_fid.shape[1] - 1)]
    fid = np.where(kind == 0, data_fid[frag], np.where(blob_id < side_fid.shape[1], side, -1))
    if fid.min(initial=0) < 0:
        raise ValueError("a blob is in a side file the dataset does not have")
    pos = descriptors.field("position").to_numpy(zero_copy_only=False).astype(np.int64)
    size = descriptors.field("size").to_numpy(zero_copy_only=False).astype(np.int64)
    return fid, pos, size
