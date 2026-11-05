"""JPEG pieces shared by the token loader and the store converter, each defined once.

Frames travel as one byte buffer plus offsets (`buf[off[i]:off[i + 1]]` is frame i), the layout of an arrow binary
column, so batches go between lance, numba and C without copies.

    info = probe(buf, off)                    header facts per frame (numba, no GIL): SOF kind, precision,
                                              components and sampling, restart interval, quantization-table hash,
                                              size, MCU grid, end of the scan header
    layout(info)                              per frame: "4:2:0", "4:4:4", ..., "gray", "other", "unsupported"
    out, out_off, status = to_restart(buf, off)   lossless re-code: baseline, a restart marker after every MCU,
                                              standard Huffman tables, COM markers kept -- libjpeg in C, the same
                                              bytes as `jpegtran -restart 1B` (no GIL)
    seg, seg_off, ok = segment_lengths(buf, off, info)   per-MCU entropy-segment lengths of restart-coded frames
    frames_of(buf, off)                       frame i as a numpy view

Whole-frame decoders are in decoders.py.
"""

import ctypes
import shutil
import subprocess
import threading

import numpy as np


try:
    import numba
except ImportError:
    numba = None

SOF_BASELINE, SOF_EXTENDED, SOF_PROGRESSIVE = 0xC0, 0xC1, 0xC2
LAYOUTS = {0x22: "4:2:0", 0x21: "4:2:2", 0x12: "4:4:0", 0x11: "4:4:4"}  # luma sampling, chroma 1 x 1
INFO = (
    "ok",
    "sof",
    "precision",
    "ncomp",
    "hv0",
    "hv1",
    "hv2",
    "dri",
    "h",
    "w",
    "sof_pos",
    "sos_end",
    "mcu_w",
    "mcu_h",
    "rows",
    "cols",
)

# ---------------------------------------------------------------------------------------------------------------
# headers (numba)
# ---------------------------------------------------------------------------------------------------------------
if numba is not None:

    @numba.njit(cache=True, nogil=True)
    def _probe(data, off, out, qhash):
        for f in range(len(off) - 1):
            i, end = off[f], off[f + 1]
            r = out[f]
            r[:] = 0
            h = np.uint64(14695981039346656037)  # FNV-1a over the quantization tables (by id) and their use
            tq = np.zeros(4, np.int64)
            if end - i < 4 or data[i] != 0xFF or data[i + 1] != 0xD8:
                continue
            i += 2
            ok = False
            while i + 4 <= end:
                if data[i] != 0xFF:
                    break
                m = data[i + 1]
                if m == 0xFF:  # fill byte
                    i += 1
                    continue
                if m == 0x01 or (0xD0 <= m <= 0xD7):
                    i += 2
                    continue
                L = (np.int64(data[i + 2]) << 8) | np.int64(data[i + 3])
                if L < 2 or i + 2 + L > end:
                    break
                if 0xC0 <= m <= 0xCF and m != 0xC4 and m != 0xC8 and m != 0xCC:
                    r[1] = m
                    r[2] = data[i + 4]
                    r[8] = (np.int64(data[i + 5]) << 8) | np.int64(data[i + 6])
                    r[9] = (np.int64(data[i + 7]) << 8) | np.int64(data[i + 8])
                    nc = np.int64(data[i + 9])
                    r[3] = nc
                    r[10] = i
                    for c in range(min(nc, 3)):
                        r[4 + c] = data[i + 11 + 3 * c]
                        h = (h ^ np.uint64(data[i + 11 + 3 * c])) * np.uint64(1099511628211)
                        h = (h ^ np.uint64(data[i + 12 + 3 * c])) * np.uint64(1099511628211)
                elif m == 0xDB:  # DQT: one or more tables, each (precision << 4 | id) then 64 or 128 bytes
                    j = i + 4
                    while j < i + 2 + L:
                        pq, t = data[j] >> 4, data[j] & 15
                        n = 128 if pq else 64
                        th = np.uint64(14695981039346656037)
                        for k in range(j, min(j + 1 + n, end)):
                            th = (th ^ np.uint64(data[k])) * np.uint64(1099511628211)
                        if t < 4:
                            tq[t] = np.int64(th)
                        j += 1 + n
                elif m == 0xDD:
                    r[7] = (np.int64(data[i + 4]) << 8) | np.int64(data[i + 5])
                elif m == 0xDA:
                    r[11] = i + 2 + L
                    ok = r[1] != 0
                    break
                i += 2 + L
            for t in range(4):
                h = (h ^ np.uint64(tq[t])) * np.uint64(1099511628211)
            qhash[f] = np.int64(h)
            if not ok:
                continue
            r[0] = 1
            if r[3] == 1:  # one component: non-interleaved 8 x 8 blocks, whatever its sampling factors say
                mw, mh = 8, 8
            else:
                mw, mh = 8 * max(r[4] >> 4, max(r[5] >> 4, r[6] >> 4)), 8 * max(r[4] & 15, max(r[5] & 15, r[6] & 15))
            r[12] = mw
            r[13] = mh
            r[14] = (r[8] + mh - 1) // mh
            r[15] = (r[9] + mw - 1) // mw


def probe(buf, off):
    """Header facts of every frame: a dict of int64 arrays named by INFO, plus "qhash" (equal for frames whose
    quantization tables and their assignment to components are equal) and "mcus" (rows * cols)."""
    out = np.zeros((len(off) - 1, len(INFO)), np.int64)
    qhash = np.zeros(len(off) - 1, np.int64)
    _probe(buf, np.asarray(off, np.int64), out, qhash)
    info = {k: out[:, i] for i, k in enumerate(INFO)}
    info["qhash"], info["mcus"] = qhash, info["rows"] * info["cols"]
    return info


def layout(info):
    """Per frame: its chroma layout if the splicer handles it after re-coding ("4:2:0", "4:2:2", "4:4:0",
    "4:4:4"), else "gray" (one component), "other" (another 3-component sampling, e.g. 4:1:1) or "unsupported"
    (unparseable, arithmetic-coded, lossless, hierarchical, 12-bit, CMYK)."""
    out = np.full(len(info["ok"]), "unsupported", dtype=object)
    good = (
        (info["ok"] == 1)
        & np.isin(info["sof"], (SOF_BASELINE, SOF_EXTENDED, SOF_PROGRESSIVE))
        & (info["precision"] == 8)
    )
    out[good & (info["ncomp"] == 1)] = "gray"
    color = good & (info["ncomp"] == 3)
    out[color] = "other"
    chroma11 = color & (info["hv1"] == 0x11) & (info["hv2"] == 0x11)
    for hv, name in LAYOUTS.items():
        out[chroma11 & (info["hv0"] == hv)] = name
    return out


# ---------------------------------------------------------------------------------------------------------------
# lossless re-coding (C, libjpeg)
# ---------------------------------------------------------------------------------------------------------------
_restart_lib = None
_restart_err = None  # why the C re-coder could not be built: re-coding falls back to jpegtran
_restart_lock = threading.Lock()


def _restart():
    global _restart_lib, _restart_err
    with _restart_lock:
        if _restart_err is not None:
            raise _restart_err
        if _restart_lib is None:
            from . import _native

            try:
                lib = _native.load("jpeg_restart.c", libs=["jpeg"])
            except OSError as e:
                _restart_err = e
                raise
            lib.restart_many.restype = ctypes.c_int64
            lib.restart_many.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            lib.fast_restart_many.restype = ctypes.c_int64
            lib.fast_restart_many.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            lib.fast_init()
            _restart_lib = lib
    return _restart_lib


def _many(fn, buf, off, *extra):
    n = len(off) - 1
    out_off, status = np.zeros(n + 1, np.int64), np.zeros(n, np.int32)
    cap = int(1.5 * (off[-1] - off[0])) + 4096 * n  # restarts add ~2-3 bytes per MCU; retried if short
    while True:
        out = np.empty(cap, np.uint8)
        r = fn(
            buf.ctypes.data, off.ctypes.data, n, *extra, out.ctypes.data, cap, out_off.ctypes.data, status.ctypes.data
        )
        if r >= 0:
            return out[:r], out_off, status
        cap *= 2


def to_restart(buf, off, interval=1, fast=True):
    """Re-code frames losslessly with a restart marker every `interval` MCUs. Returns (out, out_off, status):
    status 0 ok, 1 ok but libjpeg warned (corrupt data), -1 failed (empty output). One call runs on the calling
    thread without the GIL; parallelism comes from calling it from several threads.

    fast: baseline frames with the standard Huffman tables (most encoders' default) skip libjpeg's coefficient
    round trip -- only each MCU's first DC differences are re-coded, the rest copied bit for bit -- with the same
    output bytes; every other frame goes through libjpeg."""
    buf, off = np.ascontiguousarray(buf, np.uint8), np.ascontiguousarray(off, np.int64)
    try:
        lib = _restart()
    except OSError as e:
        from ._native import HOW, fallback

        err = str(e).splitlines()[0]
        if not shutil.which("jpegtran"):
            raise OSError(
                f"stable_datasets.framestore cannot re-code frames: its C re-coder could not be built ({err}); {HOW}"
            ) from e
        fallback("re-coding runs jpegtran, one process per frame (several times slower)", err)
        return _jpegtran_many(buf, off, interval)
    if not (fast and interval == 1):
        return _many(lib.restart_many, buf, off, interval)
    out, out_off, status = _many(lib.fast_restart_many, buf, off)
    slow = np.flatnonzero(status == 2)
    if len(slow) == 0:
        return out, out_off, status
    sizes = np.diff(off)[slow]
    soff = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    sbuf = np.concatenate([buf[off[i] : off[i + 1]] for i in slow]) if len(slow) else np.zeros(0, np.uint8)
    sout, sout_off, sst = _many(lib.restart_many, sbuf, soff, interval)
    status[slow] = sst
    lens = np.diff(out_off)
    lens[slow] = np.diff(sout_off)
    parts, k = [], 0
    for i in range(len(lens)):  # merge in frame order
        if k < len(slow) and slow[k] == i:
            parts.append(sout[sout_off[k] : sout_off[k + 1]])
            k += 1
        else:
            parts.append(out[out_off[i] : out_off[i + 1]])
    return np.concatenate(parts) if parts else out[:0], np.concatenate(([0], np.cumsum(lens))).astype(np.int64), status


def _jpegtran_many(buf, off, interval):
    """to_restart without the C re-coder: jpegtran -restart <interval>B writes the same bytes as restart_many (both
    keep COM markers only); exit status 2 is libjpeg's warning."""
    n = len(off) - 1
    parts, status = [], np.zeros(n, np.int32)
    for i in range(n):
        r = subprocess.run(
            ["jpegtran", "-restart", f"{interval}B"], input=buf[off[i] : off[i + 1]].tobytes(), capture_output=True
        )
        ok = r.returncode in (0, 2) and len(r.stdout) > 0
        parts.append(np.frombuffer(r.stdout, np.uint8) if ok else np.zeros(0, np.uint8))
        status[i] = (1 if r.returncode == 2 else 0) if ok else -1
    lens = [len(p) for p in parts]
    out = np.concatenate(parts) if parts else np.zeros(0, np.uint8)
    return out, np.concatenate(([0], np.cumsum(lens))).astype(np.int64), status


# ---------------------------------------------------------------------------------------------------------------
# per-MCU segment lengths (numba)
# ---------------------------------------------------------------------------------------------------------------
if numba is not None:

    @numba.njit(cache=True, nogil=True)
    def _segments(data, off, sos_end, seg_off, out, found):
        for f in range(len(off) - 1):
            eoi = off[f + 1] - 2
            if sos_end[f] == 0 or data[eoi] != 0xFF or data[eoi + 1] != 0xD9:
                found[f] = -1
                continue
            i, start, j, cap = sos_end[f], sos_end[f], seg_off[f], seg_off[f + 1]
            while i < eoi:
                if data[i] == 0xFF and 0xD0 <= data[i + 1] <= 0xD7:
                    if j + 1 >= cap or i - start > 65535:
                        j = cap + 1
                        break
                    out[j] = i - start
                    j += 1
                    i += 2
                    start = i
                else:
                    i += 1
            if j < cap:
                out[j] = eoi - start
            found[f] = j - seg_off[f] + 1


def segment_lengths(buf, off, info):
    """Entropy-segment lengths of every MCU of restart-coded frames (bytes between markers, the marker not
    counted), uint16, all frames concatenated: frame i's are seg[seg_off[i]:seg_off[i + 1]]. ok[i]: frame i has
    exactly one segment per MCU."""
    seg_off = np.concatenate(([0], np.cumsum(info["mcus"]))).astype(np.int64)
    seg = np.zeros(seg_off[-1], np.uint16)
    found = np.zeros(len(off) - 1, np.int64)
    _segments(buf, np.asarray(off, np.int64), info["sos_end"], seg_off, seg, found)
    return seg, seg_off, found == info["mcus"]


def frames_of(buf, off):
    """Frame i of a buffer + offsets as a numpy view."""
    return [buf[off[i] : off[i + 1]] for i in range(len(off) - 1)]
