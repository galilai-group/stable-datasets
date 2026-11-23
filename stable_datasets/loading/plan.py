"""Plans: a batch of samples turned into the exact byte ranges it needs, before any byte is read.

    planner = Planner(codec, records, chain, index=SamplingIndex.whole(n), sampler=Uniform())
    plan = planner.plan(rng, batch_size)       # rows, drawn params, boxes, units, coalesced reads: arrays only
    pixels = fetch_and_decode(plan, planner, source)    # the reference execution

The steps: the sampler picks sampling-index rows (a record and a box of it); the chain draws its parameters inside
each box and names the region it reads; the codec names the units of the record that cover the region and their
byte ranges; ranges of one file close together are merged into one read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..codecs.base import Units
from ..sources.ranges import Coalesced, coalesce
from ..transforms import Chain
from .sampling import Sampler, SamplingIndex, Uniform


@dataclass(frozen=True)
class Records:
    """Where each record's bytes are and what its codec needs to address them.

    fid, pos: each record's blob (file id into ``files``, byte offset); meta: per-record arrays the codec reads
    (e.g. h, w, mcu_h, mcu_w, sos_end); ragged, ragged_off: every record's unit sizes back to back, and where each
    record's start; static: meta shared by every record (e.g. dtype, compression).
    """

    files: list[str]
    fid: np.ndarray
    pos: np.ndarray
    meta: dict
    ragged: np.ndarray
    ragged_off: np.ndarray
    static: dict

    def __len__(self) -> int:
        return len(self.fid)

    def row(self, r: int) -> dict:
        out = dict(self.static)
        for k, v in self.meta.items():
            out[k] = v[r]
        return out

    @staticmethod
    def from_index(index, meta_columns, static=None) -> Records:
        """From a RecordIndex with columns fid, pos and ``meta_columns`` and one ragged column of unit sizes."""
        meta = {k: np.asarray(index.columns[k]) for k in meta_columns}
        return Records(list(index.files), np.asarray(index.columns["fid"], np.int64),
                       np.asarray(index.columns["pos"], np.int64), meta, np.asarray(index.ragged),
                       np.asarray(index.ragged_off, np.int64), dict(static or {}))  # fmt: skip


@dataclass(frozen=True)
class Plan:
    """One batch. Per sample: rows (sampling-index rows), record, label (or None), base (the sampling box in record
    coordinates), params (one dict per chain op, drawn inside base), boxes (the region read, record coordinates),
    units (CSR unit ids). Per piece (each sample's header, then each unit): piece_sample, piece_unit (-1 for a
    header), fid, offset, size. reads: the coalesced reads covering every piece."""

    rows: np.ndarray
    record: np.ndarray
    label: np.ndarray | None
    base: np.ndarray
    params: list
    boxes: np.ndarray
    units: Units
    piece_sample: np.ndarray
    piece_unit: np.ndarray
    fid: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    reads: Coalesced

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def origin(self) -> np.ndarray:
        """Where each sample's region starts inside its base box: the chain's ``origin``."""
        nd = self.base.shape[1] // 2
        return self.boxes[:, :nd] - self.base[:, :nd]


class Planner:
    def __init__(self, codec, records: Records, chain: Chain, *, index: SamplingIndex | None = None,
                 sampler: Sampler | None = None, gap: int = 0):  # fmt: skip
        self.codec, self.records, self.chain = codec, records, chain
        self.index = index if index is not None else SamplingIndex.whole(len(records))
        self.sampler = sampler or Uniform()
        self.gap = int(gap)

    def plan(self, rng: np.random.Generator, n: int, rows=None) -> Plan:
        """A plan for ``n`` samples drawn by the sampler (or for the given sampling-index rows)."""
        rec, codec = self.records, self.codec
        rows = self.sampler.draw(rng, self.index, n) if rows is None else np.asarray(rows, np.int64)
        record = np.asarray(self.index.record, np.int64)[rows]
        full = codec.shapes(rec.meta, record)
        nd = full.shape[1]
        base = self.index.boxes(rows, full)
        base = np.concatenate([np.clip(base[:, :nd], 0, full), np.clip(base[:, nd:], 0, full)], 1)
        params = self.chain.draw(rng, base[:, nd:] - base[:, :nd])
        local = self.chain.region(params)
        boxes = local + np.tile(base[:, :nd], 2)
        units = codec.cover(rec.meta, record, boxes)
        ustart, usize = codec.unit_ranges(rec.meta, rec.ragged, rec.ragged_off, record, units)
        hsize = codec.header_size(rec.meta, record)
        k = len(rows)
        piece_sample = np.concatenate([np.arange(k), units.sample()])
        piece_unit = np.concatenate([np.full(k, -1), units.ids])
        start = np.concatenate([np.zeros(k, np.int64), ustart])
        size = np.concatenate([hsize, usize]).astype(np.int64)
        fid = rec.fid[record][piece_sample]
        offset = rec.pos[record][piece_sample] + start
        label = None if self.index.label is None else np.asarray(self.index.label)[rows]
        return Plan(rows, record, label, base, params, boxes, units, piece_sample, piece_unit, fid, offset, size,
                    coalesce(fid, offset, size, self.gap))  # fmt: skip


def fetch(plan: Plan, source) -> tuple[np.ndarray, np.ndarray]:
    """Every piece's bytes in one buffer and each piece's start in it, from the plan's coalesced reads."""
    r = plan.reads
    at = np.concatenate(([0], np.cumsum(r.size)))
    buf = np.empty(int(at[-1]), np.uint8)
    if hasattr(source, "read_into"):
        source.read_into(r.fid, r.offset, r.size, buf, at[:-1])
    else:
        got, start = source.read_ranges(r.fid, r.offset, r.size)
        for a, s, n in zip(at[:-1], start, r.size):
            buf[a : a + n] = got[s : s + n]
    return buf, at[r.read] + r.delta


def decode_sample(plan: Plan, planner: Planner, buf, piece_start, i: int):
    """Sample i's region decoded by the codec's reference decoder, as a torch tensor with channels first (images)
    or the array's own axes (volumes)."""
    import torch

    mine = np.flatnonzero(plan.piece_sample == i)
    head_piece = mine[plan.piece_unit[mine] == -1][0]
    units = mine[plan.piece_unit[mine] >= 0]
    header = buf[piece_start[head_piece] : piece_start[head_piece] + plan.size[head_piece]]
    pieces = [buf[piece_start[p] : piece_start[p] + plan.size[p]] for p in units]
    r = int(plan.record[i])
    x = planner.codec.decode_units(header, pieces, plan.piece_unit[units], planner.records.row(r), plan.boxes[i])
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.permute(2, 0, 1) if planner.codec.channels_last else t[None]


def fetch_and_decode(plan: Plan, planner: Planner, source) -> list:
    """The reference execution of a plan: read, decode each sample's region, apply the chain."""
    buf, piece_start = fetch(plan, source)
    origin = plan.origin
    return [
        planner.chain.apply(decode_sample(plan, planner, buf, piece_start, i), plan.params, i, origin=origin[i])
        for i in range(len(plan))
    ]
