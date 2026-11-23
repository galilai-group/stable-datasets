"""Byte ranges of local files, read with pread from a pool of threads (C loop: framestore.preads)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from .ranges import as_ranges, coalesce


class LocalSource:
    """A ByteSource over local (or NFS-mounted) files.

    Each process opens its own descriptors on first use, so a source survives fork and pickling into DataLoader
    workers. ``direct=True`` opens the files O_DIRECT: byte-exact reads that bypass the page cache on NFS (local
    filesystems need aligned O_DIRECT reads, which this does not do).
    """

    def __init__(self, files: Sequence[str | os.PathLike], *, threads: int = 16, direct: bool = False):
        self.files = [os.fspath(f) for f in files]
        self.threads = int(threads)
        self.direct = bool(direct)
        self._pid = None
        self._fds = None
        self._preads = None
        self._buf = np.zeros(0, np.uint8)

    def __getstate__(self):
        return {"files": self.files, "threads": self.threads, "direct": self.direct}

    def __setstate__(self, state):
        self.__init__(state["files"], threads=state["threads"], direct=state["direct"])

    def _open(self):
        if self._pid != os.getpid():
            from ..framestore.preads import PreadMany

            flags = os.O_RDONLY | (os.O_DIRECT if self.direct else 0)
            self._fds = np.array([os.open(f, flags) for f in self.files], np.int64)
            self._preads = PreadMany(self.threads)
            self._pid = os.getpid()
        return self._fds

    def close(self):
        if self._pid == os.getpid() and self._fds is not None:
            for fd in self._fds:
                os.close(int(fd))
        self._pid = self._fds = self._preads = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_into(self, fid, offset, size, buf: np.ndarray, dst) -> None:
        """``buf[dst[i] : dst[i] + size[i]]`` = bytes ``[offset[i], offset[i] + size[i])`` of ``files[fid[i]]``:
        one pread per range, placed where the caller says."""
        fid, offset, size = as_ranges(fid, offset, size)
        dst = np.ascontiguousarray(dst, np.int64).reshape(-1)
        if len(dst) != len(fid):
            raise ValueError("read_into needs one destination per range")
        if not len(fid):
            return
        if dst.min() < 0 or int((dst + size).max()) > len(buf):
            raise ValueError("a range falls outside the buffer")
        fds = self._open()
        self._preads(fds[fid], offset, dst, size, buf)

    def read_ranges(self, fid, offset, size, *, gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """The ranges' bytes in one buffer, and where each range starts in it.

        Ranges of one file that lie at most ``gap`` bytes apart are read together (coalesce). The buffer is
        reused by the next call, so copy what must outlive it.
        """
        fid, offset, size = as_ranges(fid, offset, size)
        c = coalesce(fid, offset, size, gap)
        at = np.concatenate(([0], np.cumsum(c.size)))
        if len(self._buf) < at[-1]:
            self._buf = np.empty(int(at[-1]), np.uint8)
        self.read_into(c.fid, c.offset, c.size, self._buf, at[:-1])
        return self._buf[: at[-1]], at[c.read] + c.delta
