"""The Codec interface: what a stored record's bytes mean, and which of its units a region needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Units:
    """Unit ids per sample, in CSR form: sample i needs ``ids[offsets[i] : offsets[i + 1]]`` (sorted, unique)."""

    offsets: np.ndarray
    ids: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def of(self, i: int) -> np.ndarray:
        return self.ids[self.offsets[i] : self.offsets[i + 1]]

    def sample(self) -> np.ndarray:
        """The sample of every entry of ``ids``."""
        return np.repeat(np.arange(len(self)), np.diff(self.offsets))

    @staticmethod
    def from_lists(lists) -> Units:
        sizes = [len(x) for x in lists]
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        ids = np.concatenate([np.asarray(x, np.int64) for x in lists]) if lists else np.zeros(0, np.int64)
        return Units(offsets, ids.astype(np.int64))


@runtime_checkable
class Codec(Protocol):
    """A record is stored as units: pieces that decode on their own (an MCU of a restart-coded JPEG, a chunk of an
    array, the frames of a GOP). ``cover`` says which units a region of each record needs, dependencies included,
    so a reader fetches and decodes only those. Formats without random access have one unit per record."""

    name: str
    channels_last: bool

    def shapes(self, meta: dict[str, np.ndarray], record: np.ndarray) -> np.ndarray:
        """(n, ndim) spatial shape of each record."""
        ...

    def header_size(self, meta: dict[str, np.ndarray], record: np.ndarray) -> np.ndarray:
        """Bytes at the start of each record that every partial decode needs (0: none)."""
        ...

    def unit_ranges(self, meta, ragged, ragged_off, record, units: Units) -> tuple[np.ndarray, np.ndarray]:
        """(start, size) inside the record's bytes of every unit ``units`` names."""
        ...

    def decode_units(self, header, pieces, ids, meta_row: dict, box) -> np.ndarray:
        """The pixels (voxels) of ``box`` from the record's header bytes and the fetched units ``ids``."""
        ...

    def cover(self, meta: dict[str, np.ndarray], record: np.ndarray, boxes: np.ndarray) -> Units:
        """Units of ``record[i]`` that ``boxes[i]`` needs. ``meta`` holds per-record arrays; a box is
        ``(start_0, ..., start_k, stop_0, ..., stop_k)`` in the record's pixel (or voxel, or frame) coordinates."""
        ...


def grid_cover(shape: np.ndarray, unit: np.ndarray, boxes: np.ndarray) -> Units:
    """Ids of the cells of a regular grid that each box overlaps, cells numbered in row-major order.

    shape, unit: (n, ndim) record sizes and cell sizes; boxes: (n, 2 * ndim) starts then stops, clipped to the
    record. An empty box needs no cells.
    """
    shape, unit, boxes = (np.asarray(x, np.int64) for x in (shape, unit, boxes))
    n, ndim = shape.shape
    if boxes.shape != (n, 2 * ndim):
        raise ValueError(f"boxes must be (n, {2 * ndim}), got {boxes.shape}")
    lo = np.clip(boxes[:, :ndim], 0, shape)
    hi = np.clip(boxes[:, ndim:], 0, shape)
    cells = -(-shape // unit)
    first, last = lo // unit, np.where(hi > lo, (hi - 1) // unit, lo // unit - 1)
    lists = []
    for i in range(n):
        if (hi[i] <= lo[i]).any():
            lists.append(np.zeros(0, np.int64))
            continue
        axes = [np.arange(first[i, d], last[i, d] + 1) for d in range(ndim)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, ndim)
        strides = np.r_[np.cumprod(cells[i, 1:][::-1])[::-1], 1]
        lists.append(grid @ strides)
    return Units.from_lists(lists)
