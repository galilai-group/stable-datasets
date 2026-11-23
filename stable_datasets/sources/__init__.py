"""Byte sources: where the bytes of a cache come from.

    source = LocalSource(files)
    buf, start = source.read_ranges(fid, offset, size)    # range i: buf[start[i] : start[i] + size[i]]

Every read in stable_datasets that is not a Lance scan goes through a ByteSource, so the storage location (local
disk, NFS, an object store, a node cache) is one implementation of this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .local import LocalSource
from .ranges import Coalesced, coalesce


@runtime_checkable
class ByteSource(Protocol):
    """Reads byte ranges ``(file id, offset, size)`` of a fixed list of files."""

    files: list[str]

    def read_ranges(self, fid, offset, size, *, gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """The ranges' bytes in one uint8 buffer, and where each range starts in it."""
        ...


__all__ = ["ByteSource", "Coalesced", "LocalSource", "coalesce"]
