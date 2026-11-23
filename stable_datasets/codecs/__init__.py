"""Codecs: what a stored record's bytes mean, and which of its units a region needs.

    codec = get("jpeg-restart")
    units = codec.cover(meta, record, boxes)      # the MCUs (chunks, frames) each box needs, as CSR Units

A codec states its units so the planner never parses a format. Reference decoders here are plain numpy or library
calls; fast paths (GPU decoders, fused kernels) are checked against them.
"""

from .base import Codec, Units, grid_cover
from .chunks import Chunks
from .jpeg import RestartJpeg


_CODECS = {c.name: c for c in (RestartJpeg, Chunks)}


def get(name: str) -> Codec:
    """The codec registered under ``name``."""
    if name not in _CODECS:
        raise KeyError(f"unknown codec {name!r}; known: {sorted(_CODECS)}")
    return _CODECS[name]()


__all__ = ["Chunks", "Codec", "RestartJpeg", "Units", "get", "grid_cover"]
