"""Check an installed stable-datasets wheel's prebuilt framestore helpers, without its dependencies or a compiler:
both load as the wheel's own copies, and the re-coder (libjpeg-turbo linked in) re-codes a JPEG losslessly with a
restart marker after every MCU.

    pip install --no-deps dist/stable_datasets-*.whl numpy pillow && python .github/scripts/wheel_smoke.py
"""

import ctypes
import importlib.util
import io
import os
import sys

import numpy as np
from PIL import Image


pkg = importlib.util.find_spec("stable_datasets").submodule_search_locations[0]  # without importing the package
spec = importlib.util.spec_from_file_location("_native", os.path.join(pkg, "framestore", "_native.py"))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
csrc = os.path.join(pkg, "framestore", "csrc")

for name in ("pread_many", "jpeg_restart"):
    lib = native.prebuilt(name, os.path.join(csrc, f"{name}.c"))
    assert lib is not None, f"the wheel's _{name}.so is missing, or was built from other source"
    print(f"{name}: prebuilt, key {native.source_key(os.path.join(csrc, name + '.c'))}")

lib = native.prebuilt("jpeg_restart", os.path.join(csrc, "jpeg_restart.c"))
rng = np.random.default_rng(0)
img = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
b = io.BytesIO()
Image.fromarray(img).save(b, "JPEG", quality=90)
src = np.frombuffer(b.getvalue(), np.uint8)
off = np.array([0, len(src)], np.int64)
for fn, extra in (("fast_restart_many", ()), ("restart_many", (ctypes.c_int32(1),))):
    out, out_off, status = np.empty(4 * len(src) + 4096, np.uint8), np.zeros(2, np.int64), np.zeros(1, np.int32)
    if fn == "fast_restart_many":
        lib.fast_init()
    f = getattr(lib, fn)
    f.restype = ctypes.c_int64
    n = f(
        ctypes.c_void_p(src.ctypes.data),
        ctypes.c_void_p(off.ctypes.data),
        ctypes.c_int64(1),
        *extra,
        ctypes.c_void_p(out.ctypes.data),
        ctypes.c_int64(len(out)),
        ctypes.c_void_p(out_off.ctypes.data),
        ctypes.c_void_p(status.ctypes.data),
    )
    assert n > 0 and status[0] == 0, (fn, n, status)
    dst = out[:n].tobytes()
    assert b"\xff\xdd\x00\x04\x00\x01" in dst, f"{fn}: no restart interval of one MCU"
    same = np.array_equal(np.asarray(Image.open(io.BytesIO(dst))), np.asarray(Image.open(io.BytesIO(src.tobytes()))))
    assert same, f"{fn}: re-coding changed the pixels"
    print(f"{fn}: {len(src)} -> {n} bytes, pixels unchanged")
print(f"OK {sys.platform}")
