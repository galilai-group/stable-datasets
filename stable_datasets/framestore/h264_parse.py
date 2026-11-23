"""The parse stage of the MP4 stores' H.264 decoder, compiled (numba): a frame's slices -> per-macroblock arrays
(types, quantizers, intra modes, motion vectors per 4x4 block, coefficient levels) in the layout the reconstruction
stage takes (stable_datasets.framestore.gpu_h264 on the GPU; stable_datasets.framestore.h264.reconstruct_arrays on the CPU, for checking). Same
results as stable_datasets.framestore.h264's SliceParser (the reference), ~100x faster.

Arrays of one frame (H x W macroblocks):
    mbtype  int8 (H, W): 0 P_Skip, 1 inter, 2 intra 4x4, 3 intra 16x16
    qp      int8 (H, W): luma quantizer
    cbp     int8 (H, W): coded block pattern (luma bits 0-3, chroma 4-5)
    modes   int8 (H, W, 2): intra 16x16 mode, chroma intra mode
    i4      int8 (4H, 4W): intra 4x4 mode per 4x4 block (-1: not intra 4x4)
    mv      int16 (4H, 4W, 2): motion vector per 4x4 block (quarter samples; intra: 0)
    coef    int16 (H, W, 27, 16): levels, before dequantization, raster order in each 4x4 block: blocks 0-15 luma
            (block y * 4 + x), 16 the intra 16x16 luma DC (a 4x4 of the blocks' DCs), 17 and 18 Cb and Cr DC
            (first 4), 19-22 Cb AC and 23-26 Cr AC (block y * 2 + x)
"""

import numba
import numpy as np

from . import h264


def _lut(lens, bits, values, width):
    """A lookup table over `width`-bit windows: (length << 8) | value, 0 where no code starts."""
    t = np.zeros(1 << width, np.int32)
    for n, b, v in zip(lens, bits, values):
        if n:
            base = b << (width - n)
            t[base : base + (1 << (width - n))] = (n << 8) | v
    return t


COEFF_TOKEN_LUT = np.stack(
    [_lut(h264.COEFF_TOKEN_LEN[t], h264.COEFF_TOKEN_BITS[t], [(i // 4) << 2 | (i % 4) for i in range(68)], 16)
     for t in range(4)]
)  # fmt: skip
CHROMA_DC_COEFF_TOKEN_LUT = _lut(
    h264.CHROMA_DC_COEFF_TOKEN_LEN, h264.CHROMA_DC_COEFF_TOKEN_BITS, [(i // 4) << 2 | (i % 4) for i in range(20)], 8
)
TOTAL_ZEROS_LUT = np.stack(
    [_lut(n, b, range(len(n)), 9) for n, b in zip(h264.TOTAL_ZEROS_LEN, h264.TOTAL_ZEROS_BITS)]
)  # fmt: skip
CHROMA_DC_TOTAL_ZEROS_LUT = np.stack(
    [_lut(n, b, range(len(n)), 3) for n, b in zip(h264.CHROMA_DC_TOTAL_ZEROS_LEN, h264.CHROMA_DC_TOTAL_ZEROS_BITS)]
)
RUN_LUT = np.stack([_lut(n, b, range(len(n)), 11) for n, b in zip(h264.RUN_LEN, h264.RUN_BITS)])
ZIGZAG = np.array(h264.ZIGZAG, np.int64)
INTRA_CBP = np.array(h264.GOLOMB_TO_INTRA4X4_CBP, np.int64)
INTER_CBP = np.array(h264.GOLOMB_TO_INTER_CBP, np.int64)
# decoding order -> (x, y) of the 4x4 luma blocks
BLK_X = np.array([h264.BLK_XY[n][0] for n in range(16)], np.int64)
BLK_Y = np.array([h264.BLK_XY[n][1] for n in range(16)], np.int64)


PAD = 8  # zero bytes after a slice's data, so reads need no bounds checks


@numba.njit(cache=True, inline="always")
def _peek(d, pos, k):
    """The k (<= 32) bits at bit pos (d padded with PAD zero bytes)."""
    b = pos >> 3
    v = (
        (np.int64(d[b]) << 32)
        | (np.int64(d[b + 1]) << 24)
        | (np.int64(d[b + 2]) << 16)
        | (np.int64(d[b + 3]) << 8)
        | np.int64(d[b + 4])
    )
    return (v >> (40 - (pos & 7) - k)) & ((np.int64(1) << k) - 1)


@numba.njit(cache=True, inline="always")
def _zeros(d, pos):
    """Leading zero bits at pos (up to 32)."""
    v = _peek(d, pos, 32)
    z = 0
    while z < 32 and not (v >> (31 - z)) & 1:
        z += 1
    return z


@numba.njit(cache=True, inline="always")
def _ue(d, pos):
    z = _zeros(d, pos)
    return (1 << z) - 1 + _peek(d, pos + z + 1, z), pos + 2 * z + 1


@numba.njit(cache=True, inline="always")
def _se(d, pos):
    k, pos = _ue(d, pos)
    return ((k + 1) >> 1 if k & 1 else -(k >> 1)), pos


@numba.njit(cache=True)
def _residual(
    d, pos, nc, max_coeff, chroma_dc, out, start, ct_lut, cdc_lut, tz_lut, cdtz_lut, run_lut, zigzag, scan, levels
):
    """One CAVLC block into out[start + ...] (raster positions via zigzag if scan, else in order); returns
    (total_coeff, pos)."""
    if chroma_dc:
        e = cdc_lut[_peek(d, pos, 8)]
    else:
        e = ct_lut[0 if nc < 2 else 1 if nc < 4 else 2 if nc < 8 else 3, _peek(d, pos, 16)]
    pos += e >> 8
    total, t1 = (e >> 2) & 31, e & 3
    if total == 0:
        return 0, pos
    suffix = 1 if total > 10 and t1 < 3 else 0
    for i in range(total):
        if i < t1:
            levels[i] = 1 - 2 * _peek(d, pos, 1)
            pos += 1
            continue
        prefix = _zeros(d, pos)
        pos += prefix + 1
        size = 4 if prefix == 14 and suffix == 0 else (prefix - 3 if prefix >= 15 else suffix)
        code = (min(15, prefix) << suffix) + (_peek(d, pos, size) if size else 0)
        pos += size
        if prefix >= 15 and suffix == 0:
            code += 15
        if prefix >= 16:
            code += (1 << (prefix - 3)) - 4096
        if i == t1 and t1 < 3:
            code += 2
        level = (code + 2) >> 1 if code % 2 == 0 else (-code - 1) >> 1
        levels[i] = level
        if suffix == 0:
            suffix = 1
        if abs(level) > (3 << (suffix - 1)) and suffix < 6:
            suffix += 1
    zeros = 0
    if total < max_coeff:
        if chroma_dc:
            e = cdtz_lut[total - 1, _peek(d, pos, 3)]
        else:
            e = tz_lut[total - 1, _peek(d, pos, 9)]
        pos += e >> 8
        zeros = e & 255
    p = zeros + total - 1  # scan position of the highest-frequency coefficient
    off = 16 - max_coeff if max_coeff == 15 else 0  # AC blocks start at scan position 1
    for i in range(total):
        k = p + off
        out[start + (zigzag[k] if scan else k)] = levels[i]
        if i == total - 1:
            break
        r = 0
        if zeros > 0:
            e = run_lut[min(zeros, 7) - 1, _peek(d, pos, 11)]
            pos += e >> 8
            r = e & 255
            zeros -= r
        p -= 1 + r
    return total, pos


@numba.njit(cache=True, inline="always")
def _median(a, b, c):
    return max(min(a, b), min(max(a, b), c))


@numba.njit(cache=True)
def _parse_slice(
    d,
    pos,
    slice_type,
    qp,
    first,
    last,
    num,
    W,
    slice_of,
    mbtype,
    mbqp,
    cbps,
    modes,
    i4,
    mv,
    ref,
    nnz,
    nnzc,
    coef,
    ct_lut,
    cdc_lut,
    tz_lut,
    cdtz_lut,
    run_lut,
    zigzag,
    intra_cbp,
    inter_cbp,
    blk_x,
    blk_y,
):
    """Macroblocks first..last of one slice (slice number num) into the frame's arrays; returns the bit position
    after them, or -1 on a stream outside the subset."""
    done = np.zeros((4, 4), np.bool_)
    mb = first
    while mb <= last:
        if slice_type == 0:
            skip, pos = _ue(d, pos)
            for _ in range(skip):
                if mb > last:
                    return -1
                _start_mb(mb, W, num, qp, done, slice_of, mbtype, mbqp, cbps, i4, nnz, nnzc, coef)
                mby, mbx = mb // W, mb % W
                by, bx = mby * 4, mbx * 4
                ra, ma0, ma1 = _nb(by, bx - 1, mby, mbx, num, slice_of, done, ref, mv, W)
                rb, mb0, mb1 = _nb(by - 1, bx, mby, mbx, num, slice_of, done, ref, mv, W)
                if ra == -2 or rb == -2 or (ra == 0 and ma0 == 0 and ma1 == 0) or (rb == 0 and mb0 == 0 and mb1 == 0):
                    m0, m1 = 0, 0
                else:
                    m0, m1 = _pred_mv(by, bx, 4, 0, 0, mby, mbx, num, slice_of, done, ref, mv, W)
                _set_part(by, bx, 4, 4, m0, m1, mv, ref, done)
                mb += 1
            if mb > last:
                break
        _start_mb(mb, W, num, qp, done, slice_of, mbtype, mbqp, cbps, i4, nnz, nnzc, coef)
        pos, qp = _coded_mb(d, pos, slice_type, qp, mb, num, W, slice_of, mbtype, mbqp, cbps, modes, i4, mv, ref,
                            nnz, nnzc, coef, done, ct_lut, cdc_lut, tz_lut, cdtz_lut, run_lut, zigzag, intra_cbp,
                            inter_cbp, blk_x, blk_y)  # fmt: skip
        if pos < 0:
            return -1
        mb += 1
    return pos


@numba.njit(cache=True, inline="always")
def _start_mb(mb, W, num, qp, done, slice_of, mbtype, mbqp, cbps, i4, nnz, nnzc, coef):
    mby, mbx = mb // W, mb % W
    by, bx = mby * 4, mbx * 4
    slice_of[mby, mbx] = num
    done[:] = False
    nnz[by : by + 4, bx : bx + 4] = 0
    nnzc[:, mby * 2 : mby * 2 + 2, mbx * 2 : mbx * 2 + 2] = 0
    i4[by : by + 4, bx : bx + 4] = -1
    coef[mby, mbx] = 0
    mbqp[mby, mbx] = qp
    cbps[mby, mbx] = 0
    mbtype[mby, mbx] = 0


@numba.njit(cache=True)
def _coded_mb(
    d,
    pos,
    slice_type,
    qp,
    mb,
    num,
    W,
    slice_of,
    mbtype,
    mbqp,
    cbps,
    modes,
    i4,
    mv,
    ref,
    nnz,
    nnzc,
    coef,
    done,
    ct_lut,
    cdc_lut,
    tz_lut,
    cdtz_lut,
    run_lut,
    zigzag,
    intra_cbp,
    inter_cbp,
    blk_x,
    blk_y,
):
    """One macroblock that is not skipped; returns (bit position, quantizer) or (-1, qp) outside the subset."""
    levels = done_levels = np.empty(16, np.int64)  # noqa: F841
    mby, mbx = mb // W, mb % W
    by, bx = mby * 4, mbx * 4
    t, pos = _ue(d, pos)
    if slice_type == 0 and t < 5:
        mbtype[mby, mbx] = 1
        kind = 1
        pos = _parse_inter(d, pos, t, by, bx, mby, mbx, num, slice_of, done, ref, mv, W)
        if pos < 0:
            return -1, qp
        c, pos = _ue(d, pos)
        cbp = inter_cbp[c]
    else:
        if slice_type == 0:
            t -= 5
        ref[by : by + 4, bx : bx + 4] = -1
        mv[by : by + 4, bx : bx + 4] = 0
        if t == 25:
            return -1, qp  # I_PCM
        if t == 0:
            mbtype[mby, mbx] = 2
            kind = 2
            for n in range(16):
                x, y = blk_x[n], blk_y[n]
                left = _i4_nb(by + y, bx + x - 1, mby, mbx, num, slice_of, i4, W)
                top = _i4_nb(by + y - 1, bx + x, mby, mbx, num, slice_of, i4, W)
                pred = 2 if left == -2 or top == -2 else min(2 if left < 0 else left, 2 if top < 0 else top)
                if _peek(d, pos, 1):
                    pos += 1
                    mode = pred
                else:
                    r = _peek(d, pos + 1, 3)
                    pos += 4
                    mode = r + (1 if r >= pred else 0)
                i4[by + y, bx + x] = mode
        else:
            mbtype[mby, mbx] = 3
            kind = 3
            modes[mby, mbx, 0] = (t - 1) % 4
        cm, pos = _ue(d, pos)
        modes[mby, mbx, 1] = cm
        if kind == 2:
            c, pos = _ue(d, pos)
            cbp = intra_cbp[c]
        else:
            cbp = ((t - 1) // 4) % 3 * 16 + (15 if t >= 13 else 0)
    cbps[mby, mbx] = cbp
    if cbp or kind == 3:
        dq, pos = _se(d, pos)
        qp = (qp + dq + 52) % 52
        mbqp[mby, mbx] = qp
        out = coef[mby, mbx].reshape(-1)
        if kind == 3:
            nc = _nc(nnz, by, bx, mby, mbx, num, slice_of, 4, W)
            _, pos = _residual(d, pos, nc, 16, False, out, 16 * 16, ct_lut, cdc_lut, tz_lut, cdtz_lut, run_lut,
                               zigzag, True, levels)  # fmt: skip
        for n in range(16):
            if not (cbp >> (n // 4)) & 1:
                continue
            x, y = blk_x[n], blk_y[n]
            nc = _nc(nnz, by + y, bx + x, mby, mbx, num, slice_of, 4, W)
            total, pos = _residual(d, pos, nc, 15 if kind == 3 else 16, False, out, (y * 4 + x) * 16, ct_lut,
                                   cdc_lut, tz_lut, cdtz_lut, run_lut, zigzag, True, levels)  # fmt: skip
            nnz[by + y, bx + x] = total
        if cbp >> 4:
            for p in range(2):
                _, pos = _residual(d, pos, -1, 4, True, out, (17 + p) * 16, ct_lut, cdc_lut, tz_lut, cdtz_lut,
                                   run_lut, zigzag, False, levels)  # fmt: skip
        if cbp >> 5:
            for p in range(2):
                for n in range(4):
                    x, y = n & 1, n >> 1
                    nc = _nc(nnzc[p], mby * 2 + y, mbx * 2 + x, mby, mbx, num, slice_of, 2, W)
                    total, pos = _residual(d, pos, nc, 15, False, out, (19 + 4 * p + n) * 16, ct_lut, cdc_lut,
                                           tz_lut, cdtz_lut, run_lut, zigzag, True, levels)  # fmt: skip
                    nnzc[p, mby * 2 + y, mbx * 2 + x] = total
    return pos, qp


@numba.njit(cache=True, inline="always")
def _avail(yy, xx, mby, mbx, num, slice_of, done, scale, W, use_done):
    """Whether the block at (yy, xx) (grid of `scale` blocks per macroblock) is available to macroblock (mby, mbx);
    with use_done, a block of the current macroblock only once decoded (done: its decoded 4x4 luma blocks)."""
    if yy < 0 or xx < 0 or xx >= W * scale:
        return False
    my, mx = yy // scale, xx // scale
    if my == mby and mx == mbx:
        return done[yy % 4, xx % 4] if use_done else True
    if slice_of[my, mx] != num:
        return False
    return my < mby or (my == mby and mx < mbx)


@numba.njit(cache=True, inline="always")
def _nc(nnz, by, bx, mby, mbx, num, slice_of, scale, W):
    a = b = -1
    if _avail(by, bx - 1, mby, mbx, num, slice_of, slice_of, scale, W, False):
        a = nnz[by, bx - 1]
    if _avail(by - 1, bx, mby, mbx, num, slice_of, slice_of, scale, W, False):
        b = nnz[by - 1, bx]
    if a >= 0 and b >= 0:
        return (a + b + 1) >> 1
    return a if a >= 0 else (b if b >= 0 else 0)


@numba.njit(cache=True, inline="always")
def _i4_nb(yy, xx, mby, mbx, num, slice_of, i4, W):
    """Intra 4x4 mode of a neighbouring block: -2 unavailable, -1 not intra 4x4."""
    if not _avail(yy, xx, mby, mbx, num, slice_of, slice_of, 4, W, False):
        return -2
    return i4[yy, xx]


@numba.njit(cache=True, inline="always")
def _nb(yy, xx, mby, mbx, num, slice_of, done, ref, mv, W):
    """(reference, mv x, mv y) of a neighbouring 4x4 block: reference -2 if unavailable, -1 if intra."""
    if not _avail(yy, xx, mby, mbx, num, slice_of, done, 4, W, True):
        return -2, 0, 0
    return ref[yy, xx], mv[yy, xx, 0], mv[yy, xx, 1]


@numba.njit(cache=True)
def _pred_mv(by, bx, w, shape, part, mby, mbx, num, slice_of, done, ref, mv, W):
    """Predicted mv (8.4.1.3) of the partition at block (by, bx), w blocks wide; shape 1 16x8, 2 8x16."""
    ra, a0, a1 = _nb(by, bx - 1, mby, mbx, num, slice_of, done, ref, mv, W)
    rb, b0, b1 = _nb(by - 1, bx, mby, mbx, num, slice_of, done, ref, mv, W)
    rc, c0, c1 = _nb(by - 1, bx + w, mby, mbx, num, slice_of, done, ref, mv, W)
    if rc == -2:
        rc, c0, c1 = _nb(by - 1, bx - 1, mby, mbx, num, slice_of, done, ref, mv, W)
    if shape == 1:
        if part == 0 and rb == 0:
            return b0, b1
        if part == 1 and ra == 0:
            return a0, a1
    elif shape == 2:
        if part == 0 and ra == 0:
            return a0, a1
        if part == 1 and rc == 0:
            return c0, c1
    if rb == -2 and rc == -2 and ra != -2:
        rb, b0, b1, rc, c0, c1 = ra, a0, a1, ra, a0, a1
    n = (1 if ra == 0 else 0) + (1 if rb == 0 else 0) + (1 if rc == 0 else 0)
    if n == 1:
        if ra == 0:
            return a0, a1
        if rb == 0:
            return b0, b1
        return c0, c1
    return _median(a0, b0, c0), _median(a1, b1, c1)


@numba.njit(cache=True, inline="always")
def _set_part(by, bx, w, h, m0, m1, mv, ref, done):
    for y in range(by, by + h):
        for x in range(bx, bx + w):
            mv[y, x, 0] = m0
            mv[y, x, 1] = m1
            ref[y, x] = 0
            done[y & 3, x & 3] = True


@numba.njit(cache=True)
def _parse_inter(d, pos, t, by, bx, mby, mbx, num, slice_of, done, ref, mv, W):
    if t == 0:
        p0, p1 = _pred_mv(by, bx, 4, 0, 0, mby, mbx, num, slice_of, done, ref, mv, W)
        dx, pos = _se(d, pos)
        dy, pos = _se(d, pos)
        _set_part(by, bx, 4, 4, p0 + dx, p1 + dy, mv, ref, done)
    elif t == 1:
        for p in range(2):
            p0, p1 = _pred_mv(by + 2 * p, bx, 4, 1, p, mby, mbx, num, slice_of, done, ref, mv, W)
            dx, pos = _se(d, pos)
            dy, pos = _se(d, pos)
            _set_part(by + 2 * p, bx, 4, 2, p0 + dx, p1 + dy, mv, ref, done)
    elif t == 2:
        for p in range(2):
            p0, p1 = _pred_mv(by, bx + 2 * p, 2, 2, p, mby, mbx, num, slice_of, done, ref, mv, W)
            dx, pos = _se(d, pos)
            dy, pos = _se(d, pos)
            _set_part(by, bx + 2 * p, 2, 4, p0 + dx, p1 + dy, mv, ref, done)
    else:
        sub = np.zeros(4, np.int64)
        for i in range(4):
            sub[i], pos = _ue(d, pos)
            if sub[i] > 3:
                return -1
        for i in range(4):
            y0, x0 = by + 2 * (i // 2), bx + 2 * (i % 2)
            st = sub[i]
            n = 1 if st == 0 else (2 if st < 3 else 4)
            for j in range(n):
                if st == 0:
                    dy_, dx_, w, h = 0, 0, 2, 2
                elif st == 1:
                    dy_, dx_, w, h = j, 0, 2, 1
                elif st == 2:
                    dy_, dx_, w, h = 0, j, 1, 2
                else:
                    dy_, dx_, w, h = j >> 1, j & 1, 1, 1
                p0, p1 = _pred_mv(y0 + dy_, x0 + dx_, w, 0, 0, mby, mbx, num, slice_of, done, ref, mv, W)
                dx, pos = _se(d, pos)
                dy, pos = _se(d, pos)
                _set_part(y0 + dy_, x0 + dx_, w, h, p0 + dx, p1 + dy, mv, ref, done)
    return pos


class FrameArrays:
    """The parse stage's output for one frame (see the module docstring)."""

    def __init__(self, mb_h, mb_w):
        H, W = mb_h, mb_w
        self.mbtype = np.zeros((H, W), np.int8)
        self.qp = np.zeros((H, W), np.int8)
        self.cbp = np.zeros((H, W), np.int8)
        self.modes = np.zeros((H, W, 2), np.int8)
        self.i4 = np.full((4 * H, 4 * W), -1, np.int8)
        self.mv = np.zeros((4 * H, 4 * W, 2), np.int16)
        self.coef = np.zeros((H, W, 27, 16), np.int16)
        self.slice_of = np.full((H, W), -1, np.int32)
        self._ref = np.full((4 * H, 4 * W), -1, np.int8)
        self._nnz = np.zeros((4 * H, 4 * W), np.int8)
        self._nnzc = np.zeros((2, 2 * H, 2 * W), np.int8)


@numba.njit(cache=True)
def _parse_frame(
    data,
    starts,
    info,
    W,
    slice_of,
    mbtype,
    mbqp,
    cbps,
    modes,
    i4,
    mv,
    ref,
    nnz,
    nnzc,
    coef,
    ct_lut,
    cdc_lut,
    tz_lut,
    cdtz_lut,
    run_lut,
    zigzag,
    intra_cbp,
    inter_cbp,
    blk_x,
    blk_y,
):
    """Every slice of a frame: data (the slices' bytes, each followed by PAD zeros) from starts[k]; info[k]: data bit
    offset, slice type, quantizer, first and last macroblock. Returns 0, or -1 outside the subset."""
    for k in range(len(starts)):
        d = data[starts[k] :]
        pos = _parse_slice(d, info[k, 0], info[k, 1], info[k, 2], info[k, 3], info[k, 4], k, W, slice_of, mbtype,
                           mbqp, cbps, modes, i4, mv, ref, nnz, nnzc, coef, ct_lut, cdc_lut, tz_lut, cdtz_lut,
                           run_lut, zigzag, intra_cbp, inter_cbp, blk_x, blk_y)  # fmt: skip
        if pos < 0:
            return -1
    return 0


def slice_table(nals, sps, pps):
    """A frame's slices (unescaped NAL units) -> (data, starts, info) for _parse_frame, in macroblock order."""
    headers = [h264.parse_slice_header(n, sps, pps) for n in nals]
    order = sorted(range(len(nals)), key=lambda i: headers[i]["first_mb"])
    total = sps["mb_h"] * sps["mb_w"]
    info = np.zeros((len(nals), 5), np.int64)
    parts, starts, at = [], [], 0
    for k, i in enumerate(order):
        h = headers[i]
        last = (headers[order[k + 1]]["first_mb"] if k + 1 < len(order) else total) - 1
        info[k] = h["data_bit"], h["type"], h["qp"], h["first_mb"], last
        parts.append(nals[i] + bytes(PAD))
        starts.append(at)
        at += len(parts[-1])
    return np.frombuffer(b"".join(parts), np.uint8), np.array(starts, np.int64), info


def parse_frame(nals, sps, pps, out=None):
    """A frame's slice NAL units (unescaped) -> FrameArrays (out, reused if given)."""
    H, W = sps["mb_h"], sps["mb_w"]
    f = out or FrameArrays(H, W)
    f.slice_of[:] = -1
    data, starts, info = slice_table(nals, sps, pps)
    r = _parse_frame(
        data, starts, info, W, f.slice_of, f.mbtype, f.qp, f.cbp, f.modes, f.i4, f.mv, f._ref, f._nnz, f._nnzc,
        f.coef, COEFF_TOKEN_LUT, CHROMA_DC_COEFF_TOKEN_LUT, TOTAL_ZEROS_LUT, CHROMA_DC_TOTAL_ZEROS_LUT, RUN_LUT,
        ZIGZAG, INTRA_CBP, INTER_CBP, BLK_X, BLK_Y,
    )  # fmt: skip
    if r < 0:
        raise h264.UnsupportedStream("I_PCM or a P sub-macroblock type past 3")
    return f
