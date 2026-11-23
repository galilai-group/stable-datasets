"""Restart-coded JPEG (a restart marker after every MCU): each MCU is a unit that decodes alone.

Per-record meta: h, w, mcu_h, mcu_w (from framestore.codec.probe) and, to address units, sos_end (where the scan's
data starts, from the start of the record's JPEG) with the MCU segment lengths (framestore.codec.segment_lengths).
"""

from __future__ import annotations

import numpy as np

from .base import Units, grid_cover


class RestartJpeg:
    name = "jpeg-restart"

    def cover(self, meta, record, boxes) -> Units:
        """MCUs of each record a box ``(y0, x0, y1, x1)`` overlaps, in raster order."""
        record = np.asarray(record, np.int64)
        shape = np.stack([meta["h"][record], meta["w"][record]], 1)
        unit = np.stack([meta["mcu_h"][record], meta["mcu_w"][record]], 1)
        return grid_cover(shape, unit, boxes)

    @staticmethod
    def unit_ranges(sos_end, seg, seg_off, record, units: Units) -> tuple[np.ndarray, np.ndarray]:
        """Byte ranges inside each record's JPEG of the MCU segments ``units`` names: (start, size) per entry of
        ``units.ids``, each segment followed by its 2-byte restart marker (not included)."""
        record = np.asarray(record, np.int64)
        rec = record[units.sample()]
        first = np.asarray(seg_off, np.int64)[rec]
        seg = np.asarray(seg, np.int64)
        # start of MCU k of a record: sos_end + sum over earlier MCUs of (segment + marker)
        csum = np.concatenate(([0], np.cumsum(seg + 2)))
        start = np.asarray(sos_end, np.int64)[rec] + csum[first + units.ids] - csum[first]
        return start, seg[first + units.ids]

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
        ch = tiles.shape[-1]
        region = tiles.reshape(r1 - r0, c1 - c0, mh, mw, ch).transpose(0, 2, 1, 3, 4)
        region = region.reshape((r1 - r0) * mh, (c1 - c0) * mw, ch)
        return region[y0 - r0 * mh : y1 - r0 * mh, x0 - c0 * mw : x1 - c0 * mw]
