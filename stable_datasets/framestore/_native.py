"""Small C helpers (csrc/*.c): compiled once into ~/.cache/framestore (needs a C compiler) and loaded with ctypes,
which releases the GIL for every call, so threads calling them run in parallel.

    lib = _native.load("pread_many.c")                      # cc -O2 -shared; rebuilt when the file changes
    lib = _native.load("jpeg_restart.c", libs=["jpeg"])     # linked against libjpeg(-turbo)

NATIVE_CFLAGS replaces -O2, e.g. "-O1 -g -fsanitize=address" for the fuzz tests (a separate build in the cache).
Where they cannot be built, callers fall back to slower paths and warn once (fallback()).
"""

import ctypes
import hashlib
import os
import subprocess
import warnings


def _flags(libs):
    flags = []
    for prefix in ("/opt/homebrew", "/usr/local"):  # macOS package managers keep headers outside the default path
        if libs and os.path.exists(f"{prefix}/include"):
            flags += [f"-I{prefix}/include", f"-L{prefix}/lib"]
    return flags + [f"-l{lib}" for lib in libs]


CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def load(source, libs=()):
    """The shared library built from csrc/<source> (or a path), compiled if this version of it was not yet."""
    path = source if os.path.isabs(source) else os.path.join(CSRC, source)
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        src = f.read()
    d = os.path.expanduser(os.environ.get("FRAMESTORE_CACHE", "~/.cache/framestore"))
    cflags = os.environ.get("NATIVE_CFLAGS", "-O2").split()
    key = hashlib.sha1((src + " ".join(libs) + " ".join(cflags)).encode()).hexdigest()[:12]
    so = f"{d}/{name}-{key}.so"
    if not os.path.exists(so):
        os.makedirs(d, exist_ok=True)
        tmp = f"{so}.{os.getpid()}"
        try:
            subprocess.run(
                ["cc", *cflags, "-shared", "-fPIC", "-o", tmp, path] + _flags(libs),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise OSError(f"building {name} failed:\n{e.stderr}") from None
        os.replace(tmp, so)  # atomic: concurrent processes may build at once
    return ctypes.CDLL(so)


HOW = (
    "for the fast path install a C compiler and libjpeg-turbo's headers (apt install build-essential "
    "libjpeg-turbo8-dev; brew install jpeg-turbo; conda install -c conda-forge compilers libjpeg-turbo), then run again"
)
_warned = set()


def fallback(what, err):
    """Warn once per process that `what` runs a slower path because a C helper could not be built (err)."""
    if what not in _warned:
        _warned.add(what)
        warnings.warn(f"stable_datasets.framestore: {what} ({err}); {HOW}", RuntimeWarning, stacklevel=3)


def usable_cpus():
    """CPUs this process may use: its affinity, capped by a cgroup CPU quota (containers often allow fewer CPUs than
    the host has)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    for read in (
        lambda: open("/sys/fs/cgroup/cpu.max").read().split(),  # cgroup v2
        lambda: [open(f"/sys/fs/cgroup/cpu/cpu.cfs_{k}_us").read().strip() for k in ("quota", "period")],
    ):
        try:
            quota, period = read()
        except (OSError, ValueError):
            continue
        if quota not in ("max", "-1"):
            n = min(n, max(1, int(quota) // int(period)))
        break
    return n
