"""N-D arrays as a grid of losslessly compressed chunks: each chunk is a unit that decodes alone.

    blobs, meta = encode(volume, chunk=(64, 64, 64), compression="zstd")
    units = Chunks().cover({"shape": ..., "chunk": ...}, record, boxes)
    patch = decode_region(blobs_by_id, meta, box)

Chunks are numbered in row-major order of the chunk grid; edge chunks are cut to the array (not padded). Masks and
other label arrays use the same codec on the same grid as their image, so one plan reads both.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence

import numpy as np

from .base import Units, grid_cover


COMPRESSIONS = ("zstd", "zlib", "none")


def _compressor(name: str, level: int | None):
    if name == "zstd":
        try:
            import zstandard
        except ImportError as e:
            raise ImportError("compression='zstd' needs the zstandard package (pip install zstandard)") from e
        c = zstandard.ZstdCompressor(level=3 if level is None else level)
        return c.compress
    if name == "zlib":
        return lambda b: zlib.compress(b, 6 if level is None else level)
    if name == "none":
        return bytes
    raise ValueError(f"compression must be one of {COMPRESSIONS}, got {name!r}")


def _decompressor(name: str):
    if name == "zstd":
        import zstandard

        d = zstandard.ZstdDecompressor()
        return d.decompress
    if name == "zlib":
        return zlib.decompress
    if name == "none":
        return bytes
    raise ValueError(f"compression must be one of {COMPRESSIONS}, got {name!r}")


def chunk_boxes(shape: Sequence[int], chunk: Sequence[int]) -> np.ndarray:
    """(n_chunks, 2 * ndim) starts then stops of every chunk, in row-major order."""
    shape, chunk = np.asarray(shape, np.int64), np.asarray(chunk, np.int64)
    axes = [np.arange(0, s, c) for s, c in zip(shape, chunk)]
    lo = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(shape))
    return np.concatenate([lo, np.minimum(lo + chunk, shape)], 1)


def encode(array: np.ndarray, chunk: Sequence[int], compression: str = "zstd", level: int | None = None):
    """An array as compressed chunks: (list of bytes in chunk order, meta)."""
    array = np.ascontiguousarray(array)
    if len(chunk) != array.ndim or min(chunk) < 1:
        raise ValueError(f"chunk {tuple(chunk)} does not fit an array of {array.ndim} dims")
    compress = _compressor(compression, level)
    ndim = array.ndim
    blobs = []
    for box in chunk_boxes(array.shape, chunk):
        piece = array[tuple(slice(a, b) for a, b in zip(box[:ndim], box[ndim:]))]
        blobs.append(compress(np.ascontiguousarray(piece).tobytes()))
    meta = {"shape": tuple(array.shape), "chunk": tuple(int(c) for c in chunk), "dtype": array.dtype.str,
            "compression": compression}  # fmt: skip
    return blobs, meta


class Chunks:
    name = "chunks"

    def cover(self, meta, record, boxes) -> Units:
        """Chunks of each record a box (starts then stops) overlaps."""
        record = np.asarray(record, np.int64)
        return grid_cover(np.asarray(meta["shape"])[record], np.asarray(meta["chunk"])[record], boxes)

    @staticmethod
    def decode_region(chunks: Mapping[int, bytes], meta: Mapping, box) -> np.ndarray:
        """The region ``box`` (starts then stops) of the array, from the chunks it covers (chunk id -> bytes)."""
        shape, chunk = np.asarray(meta["shape"], np.int64), np.asarray(meta["chunk"], np.int64)
        ndim, dtype = len(shape), np.dtype(meta["dtype"])
        box = np.asarray(box, np.int64)
        lo, hi = np.clip(box[:ndim], 0, shape), np.clip(box[ndim:], 0, shape)
        out = np.zeros(tuple(np.maximum(hi - lo, 0)), dtype)
        if (hi <= lo).any():
            return out
        decompress = _decompressor(meta["compression"])
        units = grid_cover(shape[None], chunk[None], np.concatenate([lo, hi])[None])
        every = chunk_boxes(shape, chunk)
        for cid in units.ids:
            c = every[cid]
            if cid not in chunks:
                raise KeyError(f"chunk {cid} of the region is missing")
            data = np.frombuffer(decompress(chunks[cid]), dtype).reshape(tuple(c[ndim:] - c[:ndim]))
            a, b = np.maximum(c[:ndim], lo), np.minimum(c[ndim:], hi)
            src = tuple(slice(x - s, y - s) for x, y, s in zip(a, b, c[:ndim]))
            dst = tuple(slice(x - s, y - s) for x, y, s in zip(a, b, lo))
            out[dst] = data[src]
        return out
