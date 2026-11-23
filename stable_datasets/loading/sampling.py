"""What a sample is: rows of a sampling index, drawn by a sampler.

A sampling index row names a record and, optionally, a box of it (a tissue tile of a slide, a patch centred on a
tumour voxel, a clip window of a video) with a label and a weight. Transforms then draw inside that box. An index
with no boxes samples whole records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingIndex:
    """record (n,), and optionally box (n, 2 * ndim) starts then stops, label (n,) and weight (n,)."""

    record: np.ndarray
    box: np.ndarray | None = None
    label: np.ndarray | None = None
    weight: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.record)
        for name in ("box", "label", "weight"):
            v = getattr(self, name)
            if v is not None and len(v) != n:
                raise ValueError(f"sampling index column {name} has {len(v)} rows, record has {n}")
        if self.weight is not None and (np.asarray(self.weight) < 0).any():
            raise ValueError("sampling weights must be >= 0")

    def __len__(self) -> int:
        return len(self.record)

    @staticmethod
    def whole(n_records: int) -> SamplingIndex:
        return SamplingIndex(np.arange(n_records, dtype=np.int64))

    def boxes(self, rows, shapes) -> np.ndarray:
        """Box of each row: its own, else the whole record (``shapes``: each row's record shape)."""
        shapes = np.asarray(shapes, np.int64)
        if self.box is None:
            return np.concatenate([np.zeros_like(shapes), shapes], 1)
        return np.asarray(self.box, np.int64)[np.asarray(rows)]

    def write_lance(self, path: str) -> None:
        from ..storage.lance_arrays import write_arrays

        cols = {"record": np.asarray(self.record, np.int64)}
        if self.box is not None:
            cols["box"] = np.asarray(self.box, np.int64)
        if self.label is not None:
            cols["label"] = np.asarray(self.label)
        if self.weight is not None:
            cols["weight"] = np.asarray(self.weight, np.float64)
        write_arrays(path, cols)

    @staticmethod
    def read_lance(path: str) -> SamplingIndex:
        from ..storage.lance_arrays import read_arrays

        c = read_arrays(path)
        return SamplingIndex(c["record"], box=c.get("box"), label=c.get("label"), weight=c.get("weight"))


class Sampler:
    """Draws rows of a sampling index. Seeded: the same seed gives the same rows, so a loader can plan reads ahead."""

    def draw(self, rng: np.random.Generator, index: SamplingIndex, n: int) -> np.ndarray:
        raise NotImplementedError


class Uniform(Sampler):
    """Rows uniformly at random, with replacement."""

    def draw(self, rng, index, n):
        return rng.integers(0, len(index), n)


class Weighted(Sampler):
    """Rows in proportion to their weight (the index's weight column)."""

    def draw(self, rng, index, n):
        if index.weight is None:
            raise ValueError("Weighted needs a sampling index with a weight column")
        w = np.asarray(index.weight, np.float64)
        total = w.sum()
        if total <= 0:
            raise ValueError("every sampling weight is 0")
        return np.minimum(np.searchsorted(np.cumsum(w), rng.random(n) * total, side="right"), len(w) - 1)


class ClassBalanced(Sampler):
    """A label uniformly among the labels present, then a row of it uniformly."""

    def draw(self, rng, index, n):
        if index.label is None:
            raise ValueError("ClassBalanced needs a sampling index with a label column")
        labels, inverse = np.unique(np.asarray(index.label), return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[order], np.arange(len(labels)))
        counts = np.bincount(inverse, minlength=len(labels))
        pick = rng.integers(0, len(labels), n)
        return order[starts[pick] + np.floor(rng.random(n) * counts[pick]).astype(np.int64)]


def clip_windows(lengths, frames: int, stride: int = 1) -> SamplingIndex:
    """Every window of ``frames`` frames, ``stride`` apart, of records of the given lengths: rows (record, box
    (start, stop)), one per possible start, so Uniform draws a window uniformly among all windows. Records shorter
    than a window have none."""
    lengths = np.asarray(lengths, np.int64)
    span = (frames - 1) * stride + 1
    starts = np.maximum(lengths - span + 1, 0)
    record = np.repeat(np.arange(len(lengths)), starts)
    first = np.repeat(np.cumsum(starts) - starts, starts)
    start = np.arange(int(starts.sum())) - first
    return SamplingIndex(record, box=np.stack([start, start + span], 1))
