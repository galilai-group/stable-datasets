"""Many preads per call, from C, so a thread issues its share of a batch's reads without the GIL.

    reader = PreadMany(threads=16)
    reader(fds, src, dst, n, buf)   # buf[dst[i]:dst[i] + n[i]] = file fds[i] bytes src[i]:src[i] + n[i]

The C loop (csrc/pread_many.c) is compiled once (_native.py; needs a C compiler); without one, a Python loop does the
same reads.
"""

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _lib():
    from . import _native

    lib = _native.load("pread_many.c")
    lib.pread_many.restype = ctypes.c_int64
    lib.pread_many.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_void_p]
    return lib


class PreadMany:
    def __init__(self, threads=16):
        try:
            self.lib = _lib()
        except (OSError, subprocess.CalledProcessError) as e:
            from ._native import fallback

            fallback("preads run in a Python loop, holding the GIL between reads", str(e).splitlines()[0])
            self.lib = None
        self.pool = ThreadPoolExecutor(threads)
        self.threads = threads

    def __call__(self, fd, src, dst, n, buf):
        fd, src, dst, n = (np.ascontiguousarray(x, np.int64) for x in (fd, src, dst, n))
        parts = np.array_split(np.arange(len(fd)), min(self.threads, max(len(fd), 1)))

        def run(idx):
            if len(idx) == 0:
                return
            a, b = idx[0], idx[-1] + 1
            if self.lib is not None:
                r = self.lib.pread_many(
                    fd[a:].ctypes.data,
                    src[a:].ctypes.data,
                    dst[a:].ctypes.data,
                    n[a:].ctypes.data,
                    b - a,
                    buf.ctypes.data,
                )
                assert r >= 0, f"pread failed at read {a - 1 - r}"
                return
            mv = memoryview(buf)
            for i in range(a, b):
                d, p, e = dst[i], src[i], dst[i] + n[i]
                while d < e:
                    k = os.preadv(int(fd[i]), [mv[d:e]], int(p))
                    assert k > 0, "short read"
                    d, p = d + k, p + k

        list(self.pool.map(run, parts))
