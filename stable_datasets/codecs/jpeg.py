"""Restart-coded JPEG (a restart marker after every MCU): each MCU is a unit that decodes alone.

Per-record meta: h, w, mcu_h, mcu_w (from framestore.codec.probe) and, to address units, sos_end (where the scan's
data starts, from the start of the record's JPEG) with the MCU segment lengths (framestore.codec.segment_lengths).
"""

from __future__ import annotations

import numpy as np

from .base import Units, grid_cover


class RestartJpeg:
    name = "jpeg-restart"
    channels_last = True  # decode_units returns (h, w, channels)

    @staticmethod
    def shapes(meta, record) -> np.ndarray:
        record = np.asarray(record, np.int64)
        return np.stack([np.asarray(meta["h"])[record], np.asarray(meta["w"])[record]], 1).astype(np.int64)

    def cover(self, meta, record, boxes) -> Units:
        """MCUs of each record a box ``(y0, x0, y1, x1)`` overlaps, in raster order."""
        record = np.asarray(record, np.int64)
        shape = np.stack([meta["h"][record], meta["w"][record]], 1)
        unit = np.stack([meta["mcu_h"][record], meta["mcu_w"][record]], 1)
        return grid_cover(shape, unit, boxes)

    @staticmethod
    def header_size(meta, record) -> np.ndarray:
        """Bytes at the start of each record that every decode needs: the JPEG headers up to the scan's data."""
        return np.asarray(meta["sos_end"], np.int64)[np.asarray(record, np.int64)]

    @staticmethod
    def unit_ranges(meta, ragged, ragged_off, record, units: Units) -> tuple[np.ndarray, np.ndarray]:
        """Byte ranges inside each record's JPEG of the MCU segments ``units`` names: (start, size) per entry of
        ``units.ids``. ``ragged``/``ragged_off``: every record's MCU segment lengths and where each record's
        start. Each segment is followed by its 2-byte restart marker (not included)."""
        record = np.asarray(record, np.int64)
        rec = record[units.sample()]
        first = np.asarray(ragged_off, np.int64)[rec]
        seg = np.asarray(ragged, np.int64)
        csum = np.concatenate(([0], np.cumsum(seg + 2)))
        start = np.asarray(meta["sos_end"], np.int64)[rec] + csum[first + units.ids] - csum[first]
        return start, seg[first + units.ids]

    @staticmethod
    def decode_units(header, pieces, ids, meta_row, box) -> np.ndarray:
        """``box`` (y0, x0, y1, x1) of a record from its header bytes and the MCU segments ``ids`` covers (bytes
        each, in the order of ids): the reference decode of fetched units."""
        from ..framestore.gpu_jpeg import color_np, decode_mcus_np, idct_np
        from ..framestore.gpu_jpeg import header as parse

        head = np.asarray(header, np.uint8)
        hdr = parse(head.tobytes() + b"\xff\xd9")
        sizes = np.array([len(p) for p in pieces], np.int64)
        data = np.concatenate([head] + [np.asarray(p, np.uint8) for p in pieces]) if len(pieces) else head
        offs = len(head) + np.concatenate(([0], np.cumsum(sizes)[:-1]))
        tiles = color_np(idct_np(decode_mcus_np(data, offs, hdr), hdr), hdr)
        mh, mw = int(meta_row["mcu_h"]), int(meta_row["mcu_w"])
        return _assemble(tiles, np.asarray(ids), int(meta_row["w"]), mh, mw, box)

    @staticmethod
    def decode(jpeg) -> np.ndarray:
        """The whole image, (h, w, channels) uint8, decoded by libjpeg-turbo (PIL)."""
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(bytes(np.asarray(jpeg, np.uint8))))
        x = np.asarray(img.convert("L" if img.mode == "L" else "RGB"))
        return x[..., None] if x.ndim == 2 else x

    @staticmethod
    def decode_region(jpeg, seg_len, box) -> np.ndarray:
        """Only the MCUs under ``box = (y0, x0, y1, x1)``, decoded and cropped to it: the reference for partial
        decoding (framestore.gpu_jpeg's numpy path; libjpeg-turbo's pixels with plain chroma upsampling)."""
        from ..framestore.gpu_jpeg import color_np, decode_mcus_np, header, idct_np, mcu_offsets

        data = np.asarray(jpeg, np.uint8)
        hdr = header(data)
        _, cols = hdr["mcus"]
        h, w = hdr["size"]
        mh, mw = 8 * max(c[1] for c in hdr["comps"]), 8 * max(c[0] for c in hdr["comps"])
        if len(hdr["comps"]) == 1:
            mh = mw = 8
        y0, x0, y1, x1 = (int(v) for v in box)
        y0, y1, x0, x1 = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
        r0, r1, c0, c1 = y0 // mh, -(-y1 // mh), x0 // mw, -(-x1 // mw)
        ids = (np.arange(r0, r1)[:, None] * cols + np.arange(c0, c1)[None]).reshape(-1)
        offs = mcu_offsets(data, seg_len, hdr["sos_end"])[ids]
        tiles = color_np(idct_np(decode_mcus_np(data, offs, hdr), hdr), hdr)
        return _assemble(tiles, ids, w, mh, mw, (y0, x0, y1, x1))


def _assemble(tiles, ids, w, mh, mw, box) -> np.ndarray:
    """MCU tiles (n, mh, mw, ch) of raster ids covering a box -> the box's pixels (h, w, ch)."""
    y0, x0, y1, x1 = (int(v) for v in box)
    ch = tiles.shape[-1]
    if len(ids) == 0:
        return np.zeros((max(y1 - y0, 0), max(x1 - x0, 0), ch), np.uint8)
    cols = -(-w // mw)
    r, c = ids // cols, ids % cols
    r0, c0 = int(r.min()), int(c.min())
    nr, nc = int(r.max()) - r0 + 1, int(c.max()) - c0 + 1
    grid = np.zeros((nr, nc, mh, mw, ch), np.uint8)
    grid[r - r0, c - c0] = tiles
    region = grid.transpose(0, 2, 1, 3, 4).reshape(nr * mh, nc * mw, ch)
    return region[y0 - r0 * mh : y1 - r0 * mh, x0 - c0 * mw : x1 - c0 * mw]
