"""Byte ranges as arrays: one entry per range, (file id, offset, size)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Coalesced:
    """Reads that cover a set of ranges: read ``i`` is bytes ``[offset[i], offset[i] + size[i])`` of file
    ``fid[i]``; range ``j`` starts ``delta[j]`` bytes into read ``read[j]``."""

    fid: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    read: np.ndarray
    delta: np.ndarray


def as_ranges(fid, offset, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three columns as int64 arrays of one length, sizes not negative."""
    fid, offset, size = (np.ascontiguousarray(x, np.int64).reshape(-1) for x in (fid, offset, size))
    if not len(fid) == len(offset) == len(size):
        raise ValueError(f"ranges of unequal length: {len(fid)}, {len(offset)}, {len(size)}")
    if len(size) and (size.min() < 0 or offset.min() < 0 or fid.min() < 0):
        raise ValueError("byte ranges need file id, offset and size >= 0")
    return fid, offset, size


def coalesce(fid, offset, size, gap: int = 0) -> Coalesced:
    """Merge ranges of one file that overlap, touch, or lie at most ``gap`` bytes apart into one read.

    A larger ``gap`` gives fewer, larger reads that also fetch the bytes between ranges. Object stores, where each
    request costs latency and money, want a gap of about a megabyte; a local disk wants 0 or a few pages.
    """
    fid, offset, size = as_ranges(fid, offset, size)
    n = len(fid)
    if n == 0:
        empty = np.zeros(0, np.int64)
        return Coalesced(empty, empty, empty, empty, empty)
    order = np.lexsort((offset, fid))
    f, a = fid[order], offset[order]
    b = a + size[order]
    # running end of the reads so far within each file: a range starts a new read when it begins past it + gap
    end = _file_running_max(f, b)
    new = np.r_[True, (f[1:] != f[:-1]) | (a[1:] > end[:-1] + gap)]
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    g_fid, g_off = f[starts], a[starts]
    g_end = np.maximum.reduceat(b, starts)
    read = np.empty(n, np.int64)
    read[order] = group
    delta = offset - g_off[read]
    return Coalesced(g_fid, g_off, g_end - g_off, read, delta)


def _file_running_max(f, b):
    """Running maximum of b that restarts where f changes (f sorted, b >= 0)."""
    span = int(b.max()) + 1
    if int(f.max()) < np.iinfo(np.int64).max // span:  # earlier files' keys are all below this file's
        key = f * span + b
        return np.maximum.accumulate(key) - f * span
    out = b.copy()
    bounds = np.r_[np.flatnonzero(np.r_[True, f[1:] != f[:-1]]), len(f)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        np.maximum.accumulate(out[s:e], out=out[s:e])
    return out
