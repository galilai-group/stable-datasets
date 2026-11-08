"""The MP4 stores' H.264 subset decoded on the GPU's CUDA cores, in Triton: frames bit-exact with stable_datasets.framestore.h264
(the reference) and so with ffmpeg, Y, U and V.

    frames = decode(slices, n_frames, mb_h, mb_w, ref_of, device)   # (Y, U, V) uint8 CUDA tensors, coded size

Stages (one launch each):
    parse        one program per slice, scalar code (one slice is a sequential bitstream): CAVLC macroblock layer,
                 motion vector prediction; writes the per-macroblock arrays of stable_datasets.framestore.h264_parse
    inter        per frame step: one program per macroblock, the inter ones (P_Skip and P): luma six-tap / chroma
                 bilinear motion compensation from the frame before plus the dequantized, inverse-transformed residual
    intra        per frame step: one program per slice row, its intra macroblocks left to right (each predicts from its
                 left neighbour; the stores' slices are macroblock rows, so nothing above is available)

Needs Triton 3.6 or newer (torch 2.11): Triton 3.4 (torch 2.8) crashes while compiling these kernels on H100s,
intermittently.

The frames of a batch are clips of whole GOPs: frame f's reference is ref_of[f] (-1 for an I frame), and the frames
are decoded in steps by their position in their GOP, so that every reference is complete before it is read.
"""

import numpy as np

from . import h264, h264_parse


try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # the CPU parts (tables, host preparation) still import
    triton = None

MIN_TRITON = (3, 6)  # Triton 3.4 (torch 2.8) crashes while compiling these kernels on H100s


def _require_triton():
    """Raise unless a Triton new enough for these kernels is installed (the GPU stages call this first)."""
    if triton is None:
        raise ImportError(
            "stable_datasets.framestore.gpu_h264 decodes on the GPU with Triton: pip install 'triton>=3.6'"
        )
    if tuple(int(p) for p in triton.__version__.split(".")[:2]) < MIN_TRITON:
        raise RuntimeError(
            f"stable_datasets.framestore.gpu_h264 needs Triton 3.6 or newer (torch 2.11), found {triton.__version__}: older Tritons "
            "crash while compiling its kernels; upgrade torch, or decode MP4 stores on the CPU (decoders='cpu')"
        )


N_COEF = 27 * 16  # coefficient levels per macroblock (h264_parse.FrameArrays.coef)

if triton is not None:
    # --- parse: LANES slices per program, one per lane; every branch a mask (lanes step one macroblock at a time) --

    @triton.jit
    def _window(data, pos, m):
        """64 bits from bit pos of each lane's slice, left-aligned (at least 33 of them valid): data, the slice's
        first 32-bit word (words in bitstream order, padded); m: lanes that read."""
        i = pos >> 5
        w0 = tl.load(data + i, mask=m, other=0).to(tl.uint32).to(tl.uint64)
        w1 = tl.load(data + i + 1, mask=m, other=0).to(tl.uint32).to(tl.uint64)
        return ((w0 << 32) | w1) << (pos & 31).to(tl.uint64)

    @triton.jit
    def _peek(data, pos, k, m):
        """The k (0..32) bits at bit pos."""
        k = k + tl.zeros_like(pos).to(tl.int32)
        return ((_window(data, pos, m) >> 1) >> (63 - k).to(tl.uint64)).to(tl.int32)

    @triton.jit
    def _zeros(data, pos, m):
        """Leading zero bits at pos (32 if the next 32 are zero)."""
        return libdevice.clz((_window(data, pos, m) >> 32).to(tl.int32))

    @triton.jit
    def _ue(data, pos, m):
        """(value, position after) of an Exp-Golomb code (up to 16 leading zeros, all the stores' codes have); one
        load of the window and a count of leading zeros; position unchanged where not m."""
        v = _window(data, pos, m)
        z = libdevice.clz((v >> 32).to(tl.int32))
        code = (v >> (63 - 2 * z).to(tl.uint64)).to(tl.int32)
        return tl.where(m, code - 1, 0), tl.where(m, pos + 2 * z + 1, pos)

    @triton.jit
    def _se(data, pos, m):
        k, pos = _ue(data, pos, m)
        return tl.where((k & 1) == 1, (k + 1) >> 1, -(k >> 1)), pos

    @triton.jit
    def _median(a, b, c):
        return tl.maximum(tl.minimum(a, b), tl.minimum(tl.maximum(a, b), c))

    @triton.jit
    def _avail(yy, xx, mby, mbx, done, scale, W, use_done, m):
        """Whether block (yy, xx) of a `scale` (2 or 4) blocks per macroblock grid is available to macroblock
        (mby, mbx), in a stream whose slices are macroblock rows (prepare checks): the current macroblock's blocks
        (with use_done, once decoded: done, 16 flags per lane, raster) and the left macroblock's; nothing above."""
        ud = tl.full(yy.shape, use_done, tl.int32) != 0
        scale = scale + tl.zeros_like(yy)
        sh = tl.where(scale == 4, 2, 1)
        inside = m & (yy >= 0) & (xx >= 0) & (xx < W * scale)
        my = tl.where(inside, yy >> sh, -1)
        mx = tl.where(inside, xx >> sh, -1)
        same = (my == mby) & (mx == mbx)
        d = tl.load(done + (yy & 3) * 4 + (xx & 3), mask=inside & same & ud, other=1)
        return inside & tl.where(same, d != 0, (my == mby) & (mx == mbx - 1))

    @triton.jit
    def _nc(nnz, by, bx, mby, mbx, done, scale, W, GW, m):
        """Predicted non-zero count from the left and top blocks; nnz: a grid GW blocks wide (per lane)."""
        ha = _avail(by, bx - 1, mby, mbx, done, scale, W, False, m)
        hb = _avail(by - 1, bx, mby, mbx, done, scale, W, False, m)
        a = tl.load(nnz + by * GW + bx - 1, mask=ha, other=0).to(tl.int32)
        b = tl.load(nnz + (by - 1) * GW + bx, mask=hb, other=0).to(tl.int32)
        return tl.where(ha & hb, (a + b + 1) >> 1, tl.where(ha, a, tl.where(hb, b, 0)))

    @triton.jit
    def _residual(data, pos, nc, maxc, cdc, out, scan, levels, m, ct_lut, cdc_lut, tz_lut, cdtz_lut, run_lut, zigzag,
                  lev_lut):  # fmt: skip
        """One CAVLC block per lane in m, written at out[raster position] (zigzag where scan); returns
        (total_coeff, pos)."""
        t = tl.where(nc < 2, 0, tl.where(nc < 4, 1, tl.where(nc < 8, 2, 3)))
        e = tl.where(
            cdc,
            tl.load(cdc_lut + _peek(data, pos, 8, m), mask=m & cdc, other=0),
            tl.load(ct_lut + t * 65536 + _peek(data, pos, 16, m), mask=m & (~cdc), other=0),
        )
        e = tl.where(m, e, 0)
        pos += (e >> 8).to(tl.int64)
        total = (e >> 2) & 31
        t1 = e & 3
        suffix = tl.where((total > 10) & (t1 < 3), 1, 0)
        n = tl.max(total, 0)
        sb = _peek(data, pos, t1, m)  # the trailing ones' signs
        for j in tl.static_range(3):
            tl.store(levels + j, 1 - 2 * ((sb >> tl.maximum(t1 - 1 - j, 0)) & 1), mask=m & (j < t1))
        pos += t1.to(tl.int64)
        for i in range(n):
            nt = m & (i < total) & (i >= t1)
            # common case from a table over the next 8 bits (level and code length per suffix length); long codes
            # (length 0 in the table) take the general path
            e = tl.load(lev_lut + suffix * 256 + _peek(data, pos, 8, nt), mask=nt, other=0)
            adv = e & 255
            lv = e >> 8
            lv = tl.where((i == t1) & (t1 < 3), lv + tl.where(lv > 0, 1, -1), lv)
            esc = nt & (adv == 0)
            if tl.max(esc.to(tl.int32), 0) > 0:
                prefix = _zeros(data, pos, esc)
                size = tl.where((prefix == 14) & (suffix == 0), 4, tl.where(prefix >= 15, prefix - 3, suffix))
                code = (tl.minimum(prefix, 15) << suffix) + _peek(data, pos + prefix + 1, size, esc)
                code += tl.where((prefix >= 15) & (suffix == 0), 15, 0)
                code += tl.where(prefix >= 16, (1 << tl.maximum(prefix - 3, 0)) - 4096, 0)
                code += tl.where((i == t1) & (t1 < 3), 2, 0)
                lvs = tl.where((code & 1) == 0, (code + 2) >> 1, (-code - 1) >> 1)
                lv = tl.where(esc, lvs, lv)
                adv = tl.where(esc, prefix + 1 + size, adv)
            pos += tl.where(nt, adv, 0).to(tl.int64)
            tl.store(levels + i, lv, mask=nt)
            s1 = tl.where(suffix == 0, 1, suffix)
            s2 = tl.where((tl.abs(lv) > (3 << tl.maximum(s1 - 1, 0))) & (s1 < 6), s1 + 1, s1)
            suffix = tl.where(nt, s2, suffix)
        mz = m & (total > 0) & (total < maxc)
        tz = tl.where(
            cdc,
            tl.load(cdtz_lut + (total - 1) * 8 + _peek(data, pos, 3, mz & cdc), mask=mz & cdc, other=0),
            tl.load(tz_lut + (total - 1) * 512 + _peek(data, pos, 9, mz & (~cdc)), mask=mz & (~cdc), other=0),
        )
        pos += tl.where(mz, tz >> 8, 0).to(tl.int64)
        zeros = tl.where(mz, tz & 255, 0)
        p = zeros + total - 1 + tl.where(maxc == 15, 1, 0)
        for i in range(n):
            mi = m & (i < total)
            k = tl.where(scan, tl.load(zigzag + tl.minimum(tl.maximum(p, 0), 15), mask=mi, other=0), p)
            tl.store(out + k, tl.load(levels + i, mask=mi, other=0).to(tl.int16), mask=mi)
            mr = mi & (zeros > 0) & (i < total - 1)
            rl = tl.load(run_lut + (tl.minimum(zeros, 7) - 1) * 2048 + _peek(data, pos, 11, mr), mask=mr, other=0)
            pos += tl.where(mr, rl >> 8, 0).to(tl.int64)
            r = tl.where(mr, rl & 255, 0)
            zeros -= r
            p -= tl.where(mi, 1 + r, 0)
        return tl.where(m, total, 0), pos

    @triton.jit
    def _nb_mv(yy, xx, mby, mbx, done, W, ref, mv, m):
        """(reference (-2 unavailable, -1 intra), mv x, mv y) of a neighbouring 4x4 block."""
        ok = _avail(yy, xx, mby, mbx, done, 4, W, True, m)
        o = yy * (4 * W) + xx
        r = tl.load(ref + o, mask=ok, other=-2).to(tl.int32)
        m0 = tl.load(mv + o * 2, mask=ok, other=0).to(tl.int32)
        m1 = tl.load(mv + o * 2 + 1, mask=ok, other=0).to(tl.int32)
        return tl.where(ok, r, -2), m0, m1

    @triton.jit
    def _pred_mv(by, bx, w, shape, part, mby, mbx, done, W, ref, mv, m):
        """Predicted mv (8.4.1.3) of the partition at block (by, bx), w blocks wide; shape 1 16x8, 2 8x16."""
        shape = shape + tl.zeros_like(by)
        part = part + tl.zeros_like(by)
        ra, a0, a1 = _nb_mv(by, bx - 1, mby, mbx, done, W, ref, mv, m)
        rb, b0, b1 = _nb_mv(by - 1, bx, mby, mbx, done, W, ref, mv, m)
        rc, c0, c1 = _nb_mv(by - 1, bx + w, mby, mbx, done, W, ref, mv, m)
        rd, d0, d1 = _nb_mv(by - 1, bx - 1, mby, mbx, done, W, ref, mv, m)
        noc = rc == -2
        rc = tl.where(noc, rd, rc)
        c0 = tl.where(noc, d0, c0)
        c1 = tl.where(noc, d1, c1)
        sub = (rb == -2) & (rc == -2) & (ra != -2)
        rb2 = tl.where(sub, ra, rb)
        b02 = tl.where(sub, a0, b0)
        b12 = tl.where(sub, a1, b1)
        rc2 = tl.where(sub, ra, rc)
        c02 = tl.where(sub, a0, c0)
        c12 = tl.where(sub, a1, c1)
        n = (ra == 0).to(tl.int32) + (rb2 == 0).to(tl.int32) + (rc2 == 0).to(tl.int32)
        p0 = tl.where(n == 1, tl.where(ra == 0, a0, tl.where(rb2 == 0, b02, c02)), _median(a0, b02, c02))
        p1 = tl.where(n == 1, tl.where(ra == 0, a1, tl.where(rb2 == 0, b12, c12)), _median(a1, b12, c12))
        d16x8_0 = (shape == 1) & (part == 0) & (rb == 0)
        d16x8_1 = (shape == 1) & (part == 1) & (ra == 0)
        d8x16_0 = (shape == 2) & (part == 0) & (ra == 0)
        d8x16_1 = (shape == 2) & (part == 1) & (rc == 0)
        p0 = tl.where(d16x8_0, b0, tl.where(d16x8_1 | d8x16_0, a0, tl.where(d8x16_1, c0, p0)))
        p1 = tl.where(d16x8_0, b1, tl.where(d16x8_1 | d8x16_0, a1, tl.where(d8x16_1, c1, p1)))
        return p0, p1

    @triton.jit
    def _set_part(by, bx, w, h, m0, m1, W, ref, mv, done, m):
        w = w + tl.zeros_like(by)
        h = h + tl.zeros_like(by)
        for k in tl.static_range(16):
            yy = k // 4
            xx = k % 4
            mk = m & (yy < h) & (xx < w)
            o = (by + yy) * (4 * W) + bx + xx
            tl.store(mv + o * 2, m0.to(tl.int16), mask=mk)
            tl.store(mv + o * 2 + 1, m1.to(tl.int16), mask=mk)
            tl.store(ref + o, tl.zeros_like(m0).to(tl.int8), mask=mk)
            tl.store(done + ((by + yy) & 3) * 4 + ((bx + xx) & 3), tl.full(m0.shape, 1, tl.int8), mask=mk)

    @triton.jit
    def _parse_kernel(data, info, n_slices, mbtype, mbqp, cbps, modes, i4, mv, ref, nnz, coef, scratch, err, ct_lut,
                      cdc_lut, tz_lut, cdtz_lut, run_lut, zigzag, intra_cbp, inter_cbp, blk_x, blk_y, lev_lut,
                      H: tl.constexpr, W: tl.constexpr, LANES: tl.constexpr):  # fmt: skip
        """LANES slices per program, one per lane. info[s]: word offset, data bit, slice type, qp, first mb, last mb,
        frame, slice number. Per-frame arrays are indexed by frame; scratch: 32 int32 per slice."""
        lane = tl.program_id(0) * LANES + tl.arange(0, LANES)
        act0 = lane < n_slices
        ip = info + lane * 8
        d = data + tl.load(ip, mask=act0, other=0).to(tl.int64)
        pos = tl.load(ip + 1, mask=act0, other=0).to(tl.int64)
        stype = tl.load(ip + 2, mask=act0, other=2)
        qp = tl.load(ip + 3, mask=act0, other=0)
        mb = tl.load(ip + 4, mask=act0, other=0)
        last = tl.load(ip + 5, mask=act0, other=-1)
        f = tl.load(ip + 6, mask=act0, other=0).to(tl.int64)
        NMB: tl.constexpr = H * W
        G: tl.constexpr = 16 * H * W
        mbtype = mbtype + f * NMB
        mbqp = mbqp + f * NMB
        cbps = cbps + f * NMB
        modes = modes + f * NMB * 2
        i4 = i4 + f * G
        mv = mv + f * G * 2
        ref = ref + f * G
        nnz = nnz + f * (G + G // 2)  # luma grid, then the two chroma grids
        coef = coef + f * NMB * 27 * 16
        done = (scratch + lane.to(tl.int64) * 32).to(tl.pointer_type(tl.int8))
        levels = scratch + lane.to(tl.int64) * 32 + 8
        bad = tl.zeros([LANES], tl.int32)
        skip = tl.zeros([LANES], tl.int32)
        need_skip = (stype == 0).to(tl.int32)
        active = act0 & (mb <= last)
        zero8 = tl.zeros([LANES], tl.int8)
        while tl.max(active.to(tl.int32), 0) > 0:
            m = active
            rs = m & (need_skip != 0)
            v, pos = _ue(d, pos, rs)
            skip = tl.where(rs, v, skip)
            need_skip = tl.where(rs, 0, need_skip)
            mby = mb // W
            mbx = mb % W
            by = mby * 4
            bx = mbx * 4
            for k in tl.static_range(16):
                o = (by + k // 4) * (4 * W) + bx + k % 4
                tl.store(done + k, zero8, mask=m)
                tl.store(nnz + o, zero8, mask=m)
                tl.store(i4 + o, zero8 - 1, mask=m)
            for k in tl.static_range(8):
                tl.store(nnz + G + (k // 4) * (G // 4) + (mby * 2 + (k // 2) % 2) * (2 * W) + mbx * 2 + k % 2, zero8,
                         mask=m)  # fmt: skip
            tl.store(mbqp + mb, qp.to(tl.int8), mask=m)
            tl.store(cbps + mb, zero8, mask=m)
            # P_Skip
            sk = m & (skip > 0)
            if tl.max(sk.to(tl.int32), 0) > 0:  # (whole sections run only when some lane needs them)
                ra, a0, a1 = _nb_mv(by, bx - 1, mby, mbx, done, W, ref, mv, sk)
                rb, b0, b1 = _nb_mv(by - 1, bx, mby, mbx, done, W, ref, mv, sk)
                zero = (
                    (ra == -2) | (rb == -2) | ((ra == 0) & (a0 == 0) & (a1 == 0)) | ((rb == 0) & (b0 == 0) & (b1 == 0))
                )
                p0, p1 = _pred_mv(by, bx, 4, 0, 0, mby, mbx, done, W, ref, mv, sk)
                tl.store(mbtype + mb, zero8, mask=sk)
                _set_part(by, bx, 4, 4, tl.where(zero, 0, p0), tl.where(zero, 0, p1), W, ref, mv, done, sk)
            skip = tl.where(sk, skip - 1, skip)
            # a coded macroblock
            cd = m & (~sk)
            t, pos = _ue(d, pos, cd)
            inter = cd & (stype == 0) & (t < 5)
            intr = cd & (~inter)
            t = tl.where(intr & (stype == 0), t - 5, t)
            tl.store(mbtype + mb, zero8 + 1, mask=inter)
            big = inter & (t >= 3)  # 8x8 partitions: four sub-macroblock types first
            if tl.max(big.to(tl.int32), 0) > 0:
                for i in range(4):
                    st, pos = _ue(d, pos, big)
                    bad = tl.where(big & (st > 3), 1, bad)
                    tl.store(levels + 16 + i, st, mask=big)
            for i in range(4):
                mi = inter & ((t >= 3) | (i == 0))
                st = tl.load(levels + 16 + i, mask=big, other=0)
                nj = tl.where(t < 3, tl.where(t == 0, 1, 2), tl.where(st == 0, 1, tl.where(st == 3, 4, 2)))
                for j in range(4):
                    mj = mi & (j < nj)
                    if tl.max(mj.to(tl.int32), 0) > 0:
                        w = tl.where(t < 3, tl.where(t == 2, 2, 4), tl.where(st <= 1, 2, 1))
                        h = tl.where(t < 3, tl.where(t == 1, 2, 4), tl.where((st == 0) | (st == 2), 2, 1))
                        dy = tl.where(
                            t < 3, tl.where(t == 1, 2 * j, 0), tl.where(st == 1, j, tl.where(st == 3, j // 2, 0))
                        )
                        dx = tl.where(
                            t < 3, tl.where(t == 2, 2 * j, 0), tl.where(st == 2, j, tl.where(st == 3, j % 2, 0))
                        )
                        y0 = by + 2 * (i // 2) + dy
                        x0 = bx + 2 * (i % 2) + dx
                        q0, q1 = _pred_mv(y0, x0, w, tl.where(t < 3, t, 0), tl.where(t < 3, j, 0), mby, mbx, done, W, ref,
                                          mv, mj)  # fmt: skip
                        ex, pos = _se(d, pos, mj)
                        ey, pos = _se(d, pos, mj)
                        _set_part(y0, x0, w, h, q0 + ex, q1 + ey, W, ref, mv, done, mj)
            c, pos = _ue(d, pos, inter)
            cbp = tl.where(inter, tl.load(inter_cbp + c, mask=inter, other=0), 0)
            # intra
            bad = tl.where(intr & (t == 25), 1, bad)
            i4m = intr & (t == 0)
            i16m = intr & (t != 0)
            if tl.max(intr.to(tl.int32), 0) > 0:
                for k in tl.static_range(16):
                    o = (by + k // 4) * (4 * W) + bx + k % 4
                    tl.store(ref + o, zero8 - 1, mask=intr)
                    tl.store(mv + o * 2, zero8.to(tl.int16), mask=intr)
                    tl.store(mv + o * 2 + 1, zero8.to(tl.int16), mask=intr)
                tl.store(mbtype + mb, zero8 + 2, mask=i4m)
                tl.store(mbtype + mb, zero8 + 3, mask=i16m)
                tl.store(modes + mb * 2, ((t - 1) % 4).to(tl.int8), mask=i16m)
            if tl.max(i4m.to(tl.int32), 0) > 0:
                for n in range(16):
                    x = tl.load(blk_x + n)
                    y = tl.load(blk_y + n)
                    hl = _avail(by + y, bx + x - 1, mby, mbx, done, 4, W, False, i4m)
                    ht = _avail(by + y - 1, bx + x, mby, mbx, done, 4, W, False, i4m)
                    left = tl.load(i4 + (by + y) * (4 * W) + bx + x - 1, mask=hl, other=2).to(tl.int32)
                    top = tl.load(i4 + (by + y - 1) * (4 * W) + bx + x, mask=ht, other=2).to(tl.int32)
                    pred = tl.where(hl & ht, tl.minimum(tl.where(left < 0, 2, left), tl.where(top < 0, 2, top)), 2)
                    flag = _peek(d, pos, 1, i4m)
                    r = _peek(d, pos + 1, 3, i4m)
                    mode = tl.where(flag == 1, pred, r + (r >= pred).to(tl.int32))
                    pos += tl.where(i4m, tl.where(flag == 1, 1, 4), 0).to(tl.int64)
                    tl.store(i4 + (by + y) * (4 * W) + bx + x, mode.to(tl.int8), mask=i4m)
            cm, pos = _ue(d, pos, intr)
            tl.store(modes + mb * 2 + 1, cm.to(tl.int8), mask=intr)
            c2, pos = _ue(d, pos, i4m)
            cbp = tl.where(i4m, tl.load(intra_cbp + c2, mask=i4m, other=0), cbp)
            cbp = tl.where(i16m, ((t - 1) // 4) % 3 * 16 + tl.where(t >= 13, 15, 0), cbp)
            tl.store(cbps + mb, cbp.to(tl.int8), mask=cd)
            hr = cd & ((cbp != 0) | i16m)
            dqv, pos = _se(d, pos, hr)
            qp = tl.where(hr, (qp + dqv + 52) % 52, qp)
            tl.store(mbqp + mb, qp.to(tl.int8), mask=hr)
            out = coef + mb.to(tl.int64) * (27 * 16)
            # residual blocks in bitstream order: 0 intra 16x16 DC, 1-16 luma (decoding order), 17-18 chroma DC,
            # 19-26 chroma AC
            for jb in range(27):
                nn = tl.minimum(tl.maximum(jb - 1, 0), 15)
                x = tl.load(blk_x + nn)
                y = tl.load(blk_y + nn)
                luma = (jb >= 1) & (jb <= 16)
                cdc = (jb == 17) | (jb == 18)
                cac = jb >= 19
                p = tl.where(jb >= 23, 1, tl.where(jb >= 19, 0, jb - 17))
                cn = (jb - 19) % 4
                if jb == 0:
                    present = hr & i16m
                elif jb <= 16:
                    present = hr & (((cbp >> (nn // 4)) & 1) == 1)
                elif jb <= 18:
                    present = hr & ((cbp >> 4) != 0)
                else:
                    present = hr & ((cbp >> 5) != 0)
                if tl.max(present.to(tl.int32), 0) > 0:
                    yy = tl.where(cac, mby * 2 + cn // 2, by + tl.where(jb == 0, 0, y))
                    xx = tl.where(cac, mbx * 2 + cn % 2, bx + tl.where(jb == 0, 0, x))
                    plane = nnz + tl.where(cac, G + p * (G // 4), 0)
                    scale = tl.where(cac, 2, 4)
                    gw = tl.where(cac, 2 * W, 4 * W)
                    nc = _nc(plane, yy, xx, mby, mbx, done, scale, W, gw, present & (~cdc))
                    nc = tl.where(cdc, -1, nc)
                    maxc = tl.where(cdc, 4, tl.where((jb == 0) | (luma & (~i16m)), 16, 15))
                    off = tl.where(
                        jb == 0,
                        256,
                        tl.where(luma, (y * 4 + x) * 16, tl.where(cdc, (17 + p) * 16, (19 + 4 * p + cn) * 16)),
                    )
                    cdcv = present & cdc
                    zk = tl.arange(0, 16)  # the block's levels start at zero (the buffer is not cleared)
                    tl.store(
                        (out + off)[:, None] + zk[None, :], tl.zeros([LANES, 16], tl.int16), mask=present[:, None]
                    )
                    total, pos = _residual(d, pos, nc, maxc, cdcv, out + off, ~cdcv, levels, present, ct_lut, cdc_lut,
                                           tz_lut, cdtz_lut, run_lut, zigzag, lev_lut)  # fmt: skip
                    tl.store(plane + yy * gw + xx, total.to(tl.int8), mask=present & (luma | cac))
            need_skip = tl.where(cd, (stype == 0).to(tl.int32), need_skip)
            mb = tl.where(m, mb + 1, mb)
            active = active & (mb <= last) & (bad == 0)
        tl.store(err + lane, bad, mask=act0)

    # --- reconstruction ----------------------------------------------------------------------------------------

    @triton.jit
    def _clip(x):
        return tl.minimum(tl.maximum(x, 0), 255)

    @triton.jit
    def _level(cb, r: tl.constexpr, c: tl.constexpr, qp, dq, dc_mode: tl.constexpr, dcval, ac):
        """Dequantized coefficient (r, c) of the block at cb (int16 levels, raster): level * v << qp / 6, or the
        already scaled DC dcval at (0, 0) with dc_mode."""
        if (r % 2 == 0) and (c % 2 == 0):
            cls: tl.constexpr = 0
        elif (r % 2 == 1) and (c % 2 == 1):
            cls: tl.constexpr = 2
        else:
            cls: tl.constexpr = 1
        v = tl.load(dq + (qp % 6) * 3 + cls)
        d = (tl.load(cb + r * 4 + c, mask=ac, other=0).to(tl.int32) * v) << (qp // 6)
        if dc_mode and r == 0 and c == 0:
            d = dcval + d * 0
        return d

    @triton.jit
    def _idct_px(cb, y, x, qp, dq, dc_mode: tl.constexpr, dcval, ac):
        """The inverse 4x4 transform's output at (y, x) (vectors) of the block at cb (vector of pointers)."""
        f0 = tl.zeros(y.shape, tl.int32)
        f1 = tl.zeros(y.shape, tl.int32)
        f2 = tl.zeros(y.shape, tl.int32)
        f3 = tl.zeros(y.shape, tl.int32)
        for r in tl.static_range(4):
            d0 = _level(cb, r, 0, qp, dq, dc_mode, dcval, ac)
            d1 = _level(cb, r, 1, qp, dq, dc_mode, dcval, ac)
            d2 = _level(cb, r, 2, qp, dq, dc_mode, dcval, ac)
            d3 = _level(cb, r, 3, qp, dq, dc_mode, dcval, ac)
            e0 = d0 + d2
            e1 = d0 - d2
            e2 = (d1 >> 1) - d3
            e3 = d1 + (d3 >> 1)
            fr = tl.where(x == 0, e0 + e3, tl.where(x == 1, e1 + e2, tl.where(x == 2, e1 - e2, e0 - e3)))
            if r == 0:
                f0 = fr
            elif r == 1:
                f1 = fr
            elif r == 2:
                f2 = fr
            else:
                f3 = fr
        g0 = f0 + f2
        g1 = f0 - f2
        g2 = (f1 >> 1) - f3
        g3 = f1 + (f3 >> 1)
        h = tl.where(y == 0, g0 + g3, tl.where(y == 1, g1 + g2, tl.where(y == 2, g1 - g2, g0 - g3)))
        return (h + 32) >> 6

    @triton.jit
    def _ld(plane, yy, xx, HP, WP):
        """Reference samples at (yy, xx), clamped to the plane (HP x WP)."""
        return tl.load(plane + tl.minimum(tl.maximum(yy, 0), HP - 1) * WP + tl.minimum(tl.maximum(xx, 0), WP - 1)).to(
            tl.int32
        )

    @triton.jit
    def _tap_h(plane, yy, xx, HP, WP):
        """Six-tap filter across columns xx - 2 .. xx + 3, unrounded."""
        return (
            _ld(plane, yy, xx - 2, HP, WP)
            - 5 * _ld(plane, yy, xx - 1, HP, WP)
            + 20 * _ld(plane, yy, xx, HP, WP)
            + 20 * _ld(plane, yy, xx + 1, HP, WP)
            - 5 * _ld(plane, yy, xx + 2, HP, WP)
            + _ld(plane, yy, xx + 3, HP, WP)
        )

    @triton.jit
    def _tap_v(plane, yy, xx, HP, WP):
        return (
            _ld(plane, yy - 2, xx, HP, WP)
            - 5 * _ld(plane, yy - 1, xx, HP, WP)
            + 20 * _ld(plane, yy, xx, HP, WP)
            + 20 * _ld(plane, yy + 1, xx, HP, WP)
            - 5 * _ld(plane, yy + 2, xx, HP, WP)
            + _ld(plane, yy + 3, xx, HP, WP)
        )

    @triton.jit
    def _luma_mc(plane, Y0, X0, xf, yf, HP, WP):
        """Luma inter prediction (8.4.2.2.1) at integer (Y0, X0) plus quarter-sample fraction (xf, yf); each kind of
        sample (half samples b, h, s, m and the centre j, 36 taps) computed only where some pixel needs it (a
        macroblock's pixels usually share one motion vector)."""
        G = _ld(plane, Y0, X0, HP, WP)
        zero = G * 0
        b = zero
        s = zero
        hh = zero
        m = zero
        j = zero
        if tl.max(((xf != 0) & (yf <= 1)).to(tl.int32), 0) > 0:  # b: a b c e f g
            b = _clip((_tap_h(plane, Y0, X0, HP, WP) + 16) >> 5)
        if tl.max(((yf == 3) & (xf != 0)).to(tl.int32), 0) > 0:  # s: p q r
            s = _clip((_tap_h(plane, Y0 + 1, X0, HP, WP) + 16) >> 5)
        if tl.max(((xf <= 1) & (yf != 0)).to(tl.int32), 0) > 0:  # h: d h n e i p
            hh = _clip((_tap_v(plane, Y0, X0, HP, WP) + 16) >> 5)
        if tl.max(((xf == 3) & (yf != 0)).to(tl.int32), 0) > 0:  # m: g k r
            m = _clip((_tap_v(plane, Y0, X0 + 1, HP, WP) + 16) >> 5)
        if tl.max((((xf == 2) & (yf != 0)) | ((yf == 2) & (xf != 0))).to(tl.int32), 0) > 0:  # j: f i j k q
            j1 = (
                _tap_h(plane, Y0 - 2, X0, HP, WP)
                - 5 * _tap_h(plane, Y0 - 1, X0, HP, WP)
                + 20 * _tap_h(plane, Y0, X0, HP, WP)
                + 20 * _tap_h(plane, Y0 + 1, X0, HP, WP)
                - 5 * _tap_h(plane, Y0 + 2, X0, HP, WP)
                + _tap_h(plane, Y0 + 3, X0, HP, WP)
            )
            j = _clip((j1 + 512) >> 10)
        H1 = _ld(plane, Y0, X0 + 1, HP, WP)
        M = _ld(plane, Y0 + 1, X0, HP, WP)
        # (xf, yf) -> sample: rows of the standard's table 8-12
        r0 = tl.where(xf == 0, G, tl.where(xf == 1, (G + b + 1) >> 1, tl.where(xf == 2, b, (H1 + b + 1) >> 1)))
        r1 = tl.where(
            xf == 0,
            (G + hh + 1) >> 1,
            tl.where(xf == 1, (b + hh + 1) >> 1, tl.where(xf == 2, (b + j + 1) >> 1, (b + m + 1) >> 1)),
        )
        r2 = tl.where(xf == 0, hh, tl.where(xf == 1, (hh + j + 1) >> 1, tl.where(xf == 2, j, (j + m + 1) >> 1)))
        r3 = tl.where(
            xf == 0,
            (M + hh + 1) >> 1,
            tl.where(xf == 1, (hh + s + 1) >> 1, tl.where(xf == 2, (j + s + 1) >> 1, (m + s + 1) >> 1)),
        )
        return tl.where(yf == 0, r0, tl.where(yf == 1, r1, tl.where(yf == 2, r2, r3)))

    @triton.jit
    def _chroma_res(cbase, cy, cx, p, cbp, qpc, dq):
        """Chroma residual at (cy, cx) of plane p (vectors) of a macroblock's coefficients cbase."""
        n = (cy // 4) * 2 + cx // 4
        dcb = cbase + (17 + p) * 16
        c00 = tl.load(dcb).to(tl.int32)
        c01 = tl.load(dcb + 1).to(tl.int32)
        c10 = tl.load(dcb + 2).to(tl.int32)
        c11 = tl.load(dcb + 3).to(tl.int32)
        bi = n // 2
        bj = n % 2
        f = tl.where(
            bi == 0,
            tl.where(bj == 0, c00 + c01 + c10 + c11, c00 - c01 + c10 - c11),
            tl.where(bj == 0, c00 + c01 - c10 - c11, c00 - c01 - c10 + c11),
        )
        dcv = ((f * tl.load(dq + (qpc % 6) * 3) * 16) << (qpc // 6)) >> 5
        ab = cbase + (19 + 4 * p + n) * 16
        r = _idct_px(ab, cy % 4, cx % 4, qpc, dq, True, dcv, ((cbp >> 5) & 1) == 1)  # AC only when coded
        return tl.where(((cbp >> 4) & 3) != 0, r, 0)

    @triton.jit
    def _inter_kernel(frames, ref_of, mbtype, mbqp, cbps, mv, coef, Yp, Cp, dq, cqp, chroma_off,
                      H: tl.constexpr, W: tl.constexpr):  # fmt: skip
        """One macroblock per program; frames: the step's frame indices. Writes inter macroblocks only."""
        pid = tl.program_id(0)
        NMB: tl.constexpr = H * W
        f = tl.load(frames + pid // NMB).to(tl.int64)
        mb = pid % NMB
        mby = mb // W
        mbx = mb % W
        t = tl.load(mbtype + f * NMB + mb)
        if t <= 1:
            HP: tl.constexpr = 16 * H
            WP: tl.constexpr = 16 * W
            rf = tl.load(ref_of + f).to(tl.int64)
            qp = tl.load(mbqp + f * NMB + mb).to(tl.int32)
            cbp = tl.load(cbps + f * NMB + mb).to(tl.int32)
            cbase = coef + (f * NMB + mb) * (27 * 16)
            l = tl.arange(0, 256)
            y = l // 16
            x = l % 16
            bo = ((mby * 4 + y // 4) * (4 * W) + mbx * 4 + x // 4).to(tl.int64)
            mvx = tl.load(mv + (f * (16 * NMB) + bo) * 2).to(tl.int32)
            mvy = tl.load(mv + (f * (16 * NMB) + bo) * 2 + 1).to(tl.int32)
            ref_y = Yp + rf * (HP * WP)
            pred = _luma_mc(ref_y, mby * 16 + y + (mvy >> 2), mbx * 16 + x + (mvx >> 2), mvx & 3, mvy & 3, HP, WP)
            blk = (y // 4) * 4 + x // 4
            b8 = (y // 8) * 2 + x // 8
            res = _idct_px(cbase + blk * 16, y % 4, x % 4, qp, dq, False, 0, ((cbp >> b8) & 1) == 1)
            tl.store(Yp + f * (HP * WP) + (mby * 16 + y) * WP + mbx * 16 + x, _clip(pred + res).to(tl.uint8))
            # chroma: both planes, 8x8 each
            cl = tl.arange(0, 128)
            p = cl // 64
            cy = (cl % 64) // 8
            cx = cl % 8
            cbo = ((mby * 4 + cy // 2) * (4 * W) + mbx * 4 + cx // 2).to(tl.int64)
            cmx = tl.load(mv + (f * (16 * NMB) + cbo) * 2).to(tl.int32)
            cmy = tl.load(mv + (f * (16 * NMB) + cbo) * 2 + 1).to(tl.int32)
            HC: tl.constexpr = 8 * H
            WC: tl.constexpr = 8 * W
            plane_ref = Cp + rf * (2 * HC * WC) + p * (HC * WC)
            yi = mby * 8 + cy + (cmy >> 3)
            xi = mbx * 8 + cx + (cmx >> 3)
            fy = cmy & 7
            fx = cmx & 7
            A = _ld(plane_ref, yi, xi, HC, WC)
            B = _ld(plane_ref, yi, xi + 1, HC, WC)
            C = _ld(plane_ref, yi + 1, xi, HC, WC)
            D = _ld(plane_ref, yi + 1, xi + 1, HC, WC)
            cpred = ((8 - fx) * (8 - fy) * A + fx * (8 - fy) * B + (8 - fx) * fy * C + fx * fy * D + 32) >> 6
            qpc = tl.load(cqp + tl.minimum(tl.maximum(qp + chroma_off, 0), 51))
            cres = _chroma_res(cbase, cy, cx, p, cbp, qpc, dq)
            dst = Cp + f * (2 * HC * WC) + p * (HC * WC) + (mby * 8 + cy) * WC + mbx * 8 + cx
            tl.store(dst, _clip(cpred + cres).to(tl.uint8))

    @triton.jit
    def _T(plane, sy, sx, k, ht, htr, htl, WP):
        """p[k, -1] of the 4x4 block at (sy, sx): above row (k 0..7, 4..7 replaced by k = 3 without the above-right
        block; k = -1 the top-left sample)."""
        kk = tl.minimum(tl.maximum(k, -1), 7)  # lanes a mode does not use can ask for anything: stay near the block
        kk = tl.where((kk >= 4) & (~htr), 3, kk)
        ok = tl.where(kk < 0, htl, ht)
        return tl.load(plane + (sy - 1) * WP + sx + kk, mask=ok, other=0).to(tl.int32)

    @triton.jit
    def _L(plane, sy, sx, k, hl, htl, WP):
        k = tl.minimum(tl.maximum(k, -1), 3)
        ok = tl.where(k < 0, htl, hl)
        return tl.load(plane + (sy + k) * WP + sx - 1, mask=ok, other=0).to(tl.int32)

    @triton.jit
    def _intra4(plane, sy, sx, mode, y, x, ht, hl, htl, htr, WP):
        T0 = _T(plane, sy, sx, x, ht, htr, htl, WP)
        L0 = _L(plane, sy, sx, y, hl, htl, WP)
        TL = _T(plane, sy, sx, x * 0 - 1, ht, htr, htl, WP)
        st = (
            _T(plane, sy, sx, x * 0, ht, htr, htl, WP)
            + _T(plane, sy, sx, x * 0 + 1, ht, htr, htl, WP)
            + _T(plane, sy, sx, x * 0 + 2, ht, htr, htl, WP)
            + _T(plane, sy, sx, x * 0 + 3, ht, htr, htl, WP)
        )
        sl = (
            _L(plane, sy, sx, y * 0, hl, htl, WP)
            + _L(plane, sy, sx, y * 0 + 1, hl, htl, WP)
            + _L(plane, sy, sx, y * 0 + 2, hl, htl, WP)
            + _L(plane, sy, sx, y * 0 + 3, hl, htl, WP)
        )
        dc = tl.where(ht & hl, (st + sl + 4) >> 3, tl.where(hl, (sl + 2) >> 2, tl.where(ht, (st + 2) >> 2, 128)))
        p = dc  # mode 2; the others where any lane uses them (mode: a vector)
        if tl.max((mode == 0).to(tl.int32), 0) > 0:
            pk = T0
            p = tl.where(mode == 0, pk, p)
        if tl.max((mode == 1).to(tl.int32), 0) > 0:
            pk = L0
            p = tl.where(mode == 1, pk, p)
        if tl.max((mode == 3).to(tl.int32), 0) > 0:
            k = x + y
            pk = tl.where(
                (x == 3) & (y == 3),
                (_T(plane, sy, sx, x * 0 + 6, ht, htr, htl, WP) + 3 * _T(plane, sy, sx, x * 0 + 7, ht, htr, htl, WP) + 2) >> 2,
                (_T(plane, sy, sx, k, ht, htr, htl, WP) + 2 * _T(plane, sy, sx, k + 1, ht, htr, htl, WP)
                 + _T(plane, sy, sx, k + 2, ht, htr, htl, WP) + 2) >> 2,
            )  # fmt: skip
            p = tl.where(mode == 3, pk, p)
        if tl.max((mode == 4).to(tl.int32), 0) > 0:
            a = (_T(plane, sy, sx, x - y - 2, ht, htr, htl, WP) + 2 * _T(plane, sy, sx, x - y - 1, ht, htr, htl, WP)
                 + _T(plane, sy, sx, x - y, ht, htr, htl, WP) + 2) >> 2  # fmt: skip
            b = (_L(plane, sy, sx, y - x - 2, hl, htl, WP) + 2 * _L(plane, sy, sx, y - x - 1, hl, htl, WP)
                 + _L(plane, sy, sx, y - x, hl, htl, WP) + 2) >> 2  # fmt: skip
            c = (_T(plane, sy, sx, x * 0, ht, htr, htl, WP) + 2 * TL + _L(plane, sy, sx, y * 0, hl, htl, WP) + 2) >> 2
            pk = tl.where(x > y, a, tl.where(x < y, b, c))
            p = tl.where(mode == 4, pk, p)
        if tl.max((mode == 5).to(tl.int32), 0) > 0:
            z = 2 * x - y
            k = x - (y >> 1)
            e = (_T(plane, sy, sx, k - 1, ht, htr, htl, WP) + _T(plane, sy, sx, k, ht, htr, htl, WP) + 1) >> 1
            o = (_T(plane, sy, sx, k - 2, ht, htr, htl, WP) + 2 * _T(plane, sy, sx, k - 1, ht, htr, htl, WP)
                 + _T(plane, sy, sx, k, ht, htr, htl, WP) + 2) >> 2  # fmt: skip
            m1 = (_L(plane, sy, sx, y * 0, hl, htl, WP) + 2 * TL + _T(plane, sy, sx, x * 0, ht, htr, htl, WP) + 2) >> 2
            m2 = (_L(plane, sy, sx, y - 1, hl, htl, WP) + 2 * _L(plane, sy, sx, y - 2, hl, htl, WP)
                  + _L(plane, sy, sx, y - 3, hl, htl, WP) + 2) >> 2  # fmt: skip
            pk = tl.where(z >= 0, tl.where((z % 2) == 0, e, o), tl.where(z == -1, m1, m2))
            p = tl.where(mode == 5, pk, p)
        if tl.max((mode == 6).to(tl.int32), 0) > 0:
            z = 2 * y - x
            k = y - (x >> 1)
            e = (_L(plane, sy, sx, k - 1, hl, htl, WP) + _L(plane, sy, sx, k, hl, htl, WP) + 1) >> 1
            o = (_L(plane, sy, sx, k - 2, hl, htl, WP) + 2 * _L(plane, sy, sx, k - 1, hl, htl, WP)
                 + _L(plane, sy, sx, k, hl, htl, WP) + 2) >> 2  # fmt: skip
            m1 = (_L(plane, sy, sx, y * 0, hl, htl, WP) + 2 * TL + _T(plane, sy, sx, x * 0, ht, htr, htl, WP) + 2) >> 2
            m2 = (_T(plane, sy, sx, x - 1, ht, htr, htl, WP) + 2 * _T(plane, sy, sx, x - 2, ht, htr, htl, WP)
                  + _T(plane, sy, sx, x - 3, ht, htr, htl, WP) + 2) >> 2  # fmt: skip
            pk = tl.where(z >= 0, tl.where((z % 2) == 0, e, o), tl.where(z == -1, m1, m2))
            p = tl.where(mode == 6, pk, p)
        if tl.max((mode == 7).to(tl.int32), 0) > 0:
            k = x + (y >> 1)
            e = (_T(plane, sy, sx, k, ht, htr, htl, WP) + _T(plane, sy, sx, k + 1, ht, htr, htl, WP) + 1) >> 1
            o = (_T(plane, sy, sx, k, ht, htr, htl, WP) + 2 * _T(plane, sy, sx, k + 1, ht, htr, htl, WP)
                 + _T(plane, sy, sx, k + 2, ht, htr, htl, WP) + 2) >> 2  # fmt: skip
            pk = tl.where((y % 2) == 0, e, o)
            p = tl.where(mode == 7, pk, p)
        if tl.max((mode == 8).to(tl.int32), 0) > 0:
            z = x + 2 * y
            k = y + (x >> 1)
            e = (
                _L(plane, sy, sx, tl.minimum(k, 3), hl, htl, WP)
                + _L(plane, sy, sx, tl.minimum(k + 1, 3), hl, htl, WP)
                + 1
            ) >> 1
            o = (_L(plane, sy, sx, tl.minimum(k, 3), hl, htl, WP) + 2 * _L(plane, sy, sx, tl.minimum(k + 1, 3), hl, htl, WP)
                 + _L(plane, sy, sx, tl.minimum(k + 2, 3), hl, htl, WP) + 2) >> 2  # fmt: skip
            l2 = _L(plane, sy, sx, y * 0 + 2, hl, htl, WP)
            l3 = _L(plane, sy, sx, y * 0 + 3, hl, htl, WP)
            pk = tl.where(z > 5, l3, tl.where(z == 5, (l2 + 3 * l3 + 2) >> 2, tl.where((z % 2) == 0, e, o)))

            p = tl.where(mode == 8, pk, p)
        return p

    @triton.jit
    def _hsign(i, k: tl.constexpr):
        """Entry (i, k) of the 4x4 Hadamard matrix [[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]] (i a vector)."""
        if k == 0:
            out = i * 0 + 1
        elif k == 1:
            out = tl.where(i < 2, 1, -1)
        elif k == 2:
            out = tl.where((i == 0) | (i == 3), 1, -1)
        else:
            out = tl.where((i % 2) == 0, 1, -1)
        return out

    @triton.jit
    def _intra_kernel(frames, slice_of, mbtype, mbqp, cbps, modes, i4, coef, Yp, Cp, dq, cqp, chroma_off, blk_x,
                      blk_y, H: tl.constexpr, W: tl.constexpr):  # fmt: skip
        """One macroblock row per program (a slice: nothing above is available), its intra macroblocks left to
        right; frames: the step's frame indices."""
        pid = tl.program_id(0)
        f = tl.load(frames + pid // H).to(tl.int64)
        mby = pid % H
        NMB: tl.constexpr = H * W
        HP: tl.constexpr = 16 * H
        WP: tl.constexpr = 16 * W
        HC: tl.constexpr = 8 * H
        WC: tl.constexpr = 8 * W
        Yf = Yp + f * (HP * WP)
        for mbx in range(W):
            mb = mby * W + mbx
            t = tl.load(mbtype + f * NMB + mb).to(tl.int32)
            if t >= 2:
                have_left = mbx > 0
                qp = tl.load(mbqp + f * NMB + mb).to(tl.int32)
                cbp = tl.load(cbps + f * NMB + mb).to(tl.int32)
                cbase = coef + (f * NMB + mb) * (27 * 16)
                if t == 2:
                    # 4x4 blocks in waves of x + 2y (10 waves, up to two blocks each, every neighbour in an
                    # earlier wave): 32 lanes, a block per 16
                    l32 = tl.arange(0, 32)
                    yy = (l32 % 16) // 4
                    xx = l32 % 4
                    j = l32 // 16
                    for wv in range(10):
                        byn = tl.maximum(wv - 3 + 1, 0) // 2 + j
                        bxn = wv - 2 * byn
                        ok = (byn <= 3) & (bxn >= 0) & (bxn <= 3)
                        byn = tl.minimum(byn, 3)
                        bxn = tl.minimum(tl.maximum(bxn, 0), 3)
                        n = (bxn >> 1) * 4 + (byn >> 1) * 8 + (bxn & 1) + (byn & 1) * 2  # decoding order
                        sy = mby * 16 + byn * 4
                        sx = mbx * 16 + bxn * 4
                        ht = byn > 0
                        hl = (bxn > 0) | have_left
                        htl = ((bxn > 0) & (byn > 0)) | ((bxn == 0) & (byn > 0) & have_left)
                        # above-right: inside this macroblock and decoded before (the row above: unavailable)
                        tx = tl.minimum(bxn + 1, 3)
                        ty = tl.maximum(byn - 1, 0)
                        htr = ht & (bxn < 3) & (((tx >> 1) * 4 + (ty >> 1) * 8 + (tx & 1) + (ty & 1) * 2) < n)
                        mode4 = tl.load(
                            i4 + f * (16 * NMB) + (mby * 4 + byn) * (4 * W) + mbx * 4 + bxn, mask=ok, other=2
                        )
                        mode4 = tl.where(ok, mode4.to(tl.int32), 2)
                        pred4 = _intra4(Yf, sy, sx, mode4, yy, xx, ht, hl, htl, htr, WP)
                        res4 = _idct_px(
                            cbase + (byn * 4 + bxn) * 16, yy, xx, qp, dq, False, 0, ok & (((cbp >> (n // 4)) & 1) == 1)
                        )
                        tl.store(Yf + (sy + yy) * WP + sx + xx, _clip(pred4 + res4).to(tl.uint8), mask=ok)
                        tl.debug_barrier()
                else:  # intra 16x16: vertical / plane need the row above, which a row slice never has
                    lane16 = tl.arange(0, 256)
                    y16 = lane16 // 16
                    x16 = lane16 % 16
                    mode16 = tl.load(modes + (f * NMB + mb) * 2).to(tl.int32)
                    left16 = tl.load(Yf + (mby * 16 + y16) * WP + mbx * 16 - 1, mask=have_left, other=0).to(tl.int32)
                    lcol16 = tl.load(
                        Yf + (mby * 16 + tl.arange(0, 16)) * WP + mbx * 16 - 1, mask=have_left, other=0
                    ).to(tl.int32)
                    dc16 = tl.where(have_left, (tl.sum(lcol16, 0) + 8) >> 4, 128)
                    pred16 = tl.where(mode16 == 1, left16, dc16)
                    # DC: inverse Hadamard of the 4x4 DC levels, then scaled (8.5.10)
                    bi16 = y16 // 4
                    bj16 = x16 // 4
                    acc16 = tl.zeros([256], tl.int32)
                    for k in tl.static_range(4):
                        sk16 = _hsign(bi16, k)
                        for m in tl.static_range(4):
                            acc16 += sk16 * _hsign(bj16, m) * tl.load(cbase + 256 + k * 4 + m).to(tl.int32)
                    v16 = tl.load(dq + (qp % 6) * 3) * 16
                    big16 = qp >= 36
                    sh_up16 = tl.maximum(qp // 6 - 6, 0)
                    sh_dn16 = tl.maximum(6 - qp // 6, 0)
                    rnd16 = tl.where(big16, 0, 1 << tl.maximum(5 - qp // 6, 0))
                    dcv16 = tl.where(big16, (acc16 * v16) << sh_up16, (acc16 * v16 + rnd16) >> sh_dn16)
                    res16 = _idct_px(
                        cbase + (bi16 * 4 + bj16) * 16, y16 % 4, x16 % 4, qp, dq, True, dcv16, (cbp & 15) != 0
                    )
                    tl.store(Yf + (mby * 16 + y16) * WP + mbx * 16 + x16, _clip(pred16 + res16).to(tl.uint8))
                # chroma: DC (left only or none) or horizontal
                cl = tl.arange(0, 128)
                p = cl // 64
                cy = (cl % 64) // 8
                cx = cl % 8
                Cf = Cp + f * (2 * HC * WC) + p * (HC * WC)
                cmode = tl.load(modes + (f * NMB + mb) * 2 + 1).to(tl.int32)
                lft = tl.load(Cf + (mby * 8 + cy) * WC + mbx * 8 - 1, mask=have_left, other=0).to(tl.int32)
                lo = tl.where(cy < 4, 0, 4)
                s4 = (
                    tl.load(Cf + (mby * 8 + lo) * WC + mbx * 8 - 1, mask=have_left, other=0).to(tl.int32)
                    + tl.load(Cf + (mby * 8 + lo + 1) * WC + mbx * 8 - 1, mask=have_left, other=0).to(tl.int32)
                    + tl.load(Cf + (mby * 8 + lo + 2) * WC + mbx * 8 - 1, mask=have_left, other=0).to(tl.int32)
                    + tl.load(Cf + (mby * 8 + lo + 3) * WC + mbx * 8 - 1, mask=have_left, other=0).to(tl.int32)
                )
                cdc = tl.where(have_left, (s4 + 2) >> 2, 128)
                cpred = tl.where(cmode == 1, lft, cdc)
                qpc = tl.load(cqp + tl.minimum(tl.maximum(qp + chroma_off, 0), 51))
                cres = _chroma_res(cbase, cy, cx, p, cbp, qpc, dq)
                tl.store(Cf + (mby * 8 + cy) * WC + mbx * 8 + cx, _clip(cpred + cres).to(tl.uint8))
                tl.debug_barrier()


# --- host side -------------------------------------------------------------------------------------------------

_TABLES = {}
_BUFFERS = {}


def _tables(device):
    import torch

    if device not in _TABLES:
        P = h264_parse
        t = lambda a, dt=torch.int32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        _TABLES[device] = {
            "ct": t(P.COEFF_TOKEN_LUT),
            "cdc": t(P.CHROMA_DC_COEFF_TOKEN_LUT),
            "tz": t(P.TOTAL_ZEROS_LUT),
            "cdtz": t(P.CHROMA_DC_TOTAL_ZEROS_LUT),
            "run": t(P.RUN_LUT),
            "zigzag": t(P.ZIGZAG),
            "intra_cbp": t(P.INTRA_CBP),
            "inter_cbp": t(P.INTER_CBP),
            "blk_x": t(P.BLK_X),
            "blk_y": t(P.BLK_Y),
            "dq": t(np.array(h264.DEQUANT).reshape(-1)),
            "cqp": t(np.array(h264.CHROMA_QP)),
            "lev": t(_level_lut()),
        }
    return _TABLES[device]


def _level_lut():
    """(level << 8) | code length of the CAVLC level codes that fit in 8 bits, per suffix length 0..6 and 8-bit
    window (0: longer, decoded the general way). Levels before the first-coefficient adjustment (9.2.2.1)."""
    t = np.zeros((7, 256), np.int32)
    for s in range(7):
        for w in range(256):
            prefix = 8 - w.bit_length() if w else 8
            if prefix + 1 + s > 8:
                continue
            code = (prefix << s) + ((w >> (8 - prefix - 1 - s)) & ((1 << s) - 1))
            level = (code + 2) >> 1 if code % 2 == 0 else (-code - 1) >> 1
            t[s, w] = level * 256 + prefix + 1 + s
    return t


class Batch:
    """Frames prepared for decode: the slices' bytes (unescaped, each followed by h264_parse.PAD or more zeros to a
    whole word) as 32-bit words in bitstream order, a row per slice of info (word offset, data bit, slice type, qp,
    first mb, last mb, frame, slice number), each frame's
    reference (-1: an I frame) and decode step (its position in its GOP)."""

    def __init__(self, data, info, ref_of, step, mb_h, mb_w, chroma_off, crop):
        self.data, self.info, self.ref_of, self.step = data, info, ref_of, step
        self.mb_h, self.mb_w, self.chroma_off, self.crop = mb_h, mb_w, chroma_off, crop


def prepare(clips):
    """clips: [(mp4 bytes, first frame, frames)], each starting on a keyframe -> Batch. Checks the subset and that
    every slice is one macroblock row (what the GPU stages assume)."""
    parts, info, ref_of, step = [], [], [], []
    at, geom = 0, None
    for data, first, count in clips:
        sps_nal, pps_nal, nal_len, samples = h264.mp4_samples(data)
        sps, pps = h264.parse_sps(h264.unescape(sps_nal)), h264.parse_pps(h264.unescape(pps_nal))
        g = (sps["mb_h"], sps["mb_w"], pps["chroma_qp_offset"], tuple(sps["crop"]))
        if geom is None:
            geom = g
        elif g != geom:
            raise ValueError(f"a batch's videos must share one coded size and chroma offset: {g} != {geom}")
        H, W = sps["mb_h"], sps["mb_w"]
        for k in range(first, first + count):
            f = len(ref_of)
            nals = [h264.unescape(n) for n in h264.sample_nals(data, *samples[k], nal_len) if n[0] & 31 in (1, 5)]
            idr = nals[0][0] & 31 == 5
            if k == first and not idr:
                raise ValueError(f"clip starts at frame {first}, which is not a keyframe")
            ref_of.append(-1 if idr else f - 1)
            step.append(0 if idr else step[-1] + 1)
            headers = sorted(((h264.parse_slice_header(n, sps, pps), n) for n in nals), key=lambda x: x[0]["first_mb"])
            if len(headers) != H or any(h["first_mb"] != r * W for r, (h, _) in enumerate(headers)):
                raise h264.UnsupportedStream("the GPU decoder needs one slice per macroblock row")
            for r, (h, n) in enumerate(headers):
                info.append((at // 4, h["data_bit"], h["type"], h["qp"], h["first_mb"], h["first_mb"] + W - 1, f, r))
                parts.append(n + bytes(h264_parse.PAD + (-(len(n) + h264_parse.PAD)) % 4))  # whole words
                at += len(parts[-1])
    return Batch(
        np.frombuffer(b"".join(parts), ">u4").astype(np.int32),  # big-endian words: bit order = bitstream order
        np.array(info, np.int32),
        np.array(ref_of, np.int32),
        np.array(step, np.int32),
        geom[0],
        geom[1],
        geom[2],
        geom[3],
    )


def decode(batch, device="cuda", stream=None, clear=False):
    """A Batch -> (Y, U, V) uint8 CUDA tensors (frames, coded height, coded width) and the parse arrays. The
    coefficient array holds only what reconstruction reads (the levels of decoded blocks) unless clear."""
    _require_triton()
    import torch

    T = _tables(device)
    H, W = batch.mb_h, batch.mb_w
    F = len(batch.ref_of)
    dev = torch.device(device)
    data = torch.as_tensor(batch.data).to(dev, non_blocking=True)
    info = torch.as_tensor(batch.info).to(dev, non_blocking=True)
    S = info.shape[0]
    # per-batch arrays, reused across calls of one shape: parse writes every macroblock's entries, so nothing needs
    # clearing (coefficients: only decoded blocks are written, and only those are read)
    key = (dev, F, H, W, S)
    if key not in _BUFFERS:
        _BUFFERS.clear()
        e = lambda shape, dt: torch.empty(shape, dtype=dt, device=dev)  # noqa: E731
        _BUFFERS[key] = (
            {
                "mbtype": e((F, H, W), torch.int8),
                "qp": e((F, H, W), torch.int8),
                "cbp": e((F, H, W), torch.int8),
                "modes": e((F, H, W, 2), torch.int8),
                "i4": e((F, 4 * H, 4 * W), torch.int8),
                "mv": e((F, 4 * H, 4 * W, 2), torch.int16),
                "coef": e((F, H, W, 27, 16), torch.int16),
            },
            e((F, 4 * H, 4 * W), torch.int8),
            e((F, 24 * H * W), torch.int8),
            e((S, 32), torch.int32),
            e(S, torch.int32),
            e((F, 16 * H, 16 * W), torch.uint8),
            e((F, 2, 8 * H, 8 * W), torch.uint8),
        )
    a, ref, nnz, scratch, err, Y, C = _BUFFERS[key]
    if clear:
        a["coef"].zero_()
    # intra slices first: the longest (dense residuals), so the predicted ones fill the GPU around them
    info = info[torch.argsort((info[:, 2] != 2).to(torch.int8), stable=True)]
    lanes = 8
    _parse_kernel[(triton.cdiv(S, lanes),)](
        data, info, S, a["mbtype"], a["qp"], a["cbp"], a["modes"], a["i4"], a["mv"], ref, nnz, a["coef"], scratch, err,
        T["ct"], T["cdc"], T["tz"], T["cdtz"], T["run"], T["zigzag"], T["intra_cbp"], T["inter_cbp"], T["blk_x"],
        T["blk_y"], T["lev"], H=H, W=W, LANES=lanes, num_warps=1,
    )  # fmt: skip
    ref_of = torch.from_numpy(batch.ref_of).to(dev)
    for s in range(int(batch.step.max()) + 1):
        frames = torch.from_numpy(np.flatnonzero(batch.step == s).astype(np.int32)).to(dev)
        n = frames.numel()
        if s > 0:
            _inter_kernel[(n * H * W,)](
                frames, ref_of, a["mbtype"], a["qp"], a["cbp"], a["mv"], a["coef"], Y, C, T["dq"], T["cqp"],
                batch.chroma_off, H=H, W=W, num_warps=4,
            )  # fmt: skip
        _intra_kernel[(n * H,)](
            frames, a["mbtype"], a["mbtype"], a["qp"], a["cbp"], a["modes"], a["i4"], a["coef"], Y, C, T["dq"],
            T["cqp"], batch.chroma_off, T["blk_x"], T["blk_y"], H=H, W=W, num_warps=1,
        )  # fmt: skip
    if int(err.max()):
        raise h264.UnsupportedStream("I_PCM or a P sub-macroblock type past 3")
    return (Y, C[:, 0], C[:, 1]), a


def gather(store, clips, out=None):
    """Clips of an MP4 store built for the GPU decoder ([(video, start, frames)], starts on keyframes) -> (bytes:
    each clip's stretch of its MP4, back to back; slice table (slices, 4) int32: payload offset in bytes, payload
    size, data bit, qp | intra << 7; geometry (mb_h, mb_w, chroma qp offset)). The CPU's share of a batch: one copy
    per clip, into out (a uint8 array; the bytes are then out[:n]) if given."""
    parts, tables, at, geom = [], [], 0, None
    for v, start, frames in clips:
        n, H, W, coff, _, _ = store.slices(v)
        if geom is None:
            geom = (H, W, coff)
        elif geom != (H, W, coff):
            raise ValueError("a batch's videos must share one coded size (build the store with a canvas)")
        b, t = store.clip_bytes(v, start, frames)
        t = t.copy()
        t[:, 0] += at
        if out is not None:
            if at + len(b) > len(out):
                raise ValueError(f"a batch's bytes do not fit in {len(out):,}")
            out[at : at + len(b)] = b
        else:
            parts.append(b)
        tables.append(t)
        at += len(b)
    return (out[:at] if out is not None else np.concatenate(parts)), np.concatenate(tables), geom


if triton is not None:

    @triton.jit
    def _epb_count_kernel(raw, tab, kept, CH: tl.constexpr):
        """Per slice: its payload bytes less its emulation prevention bytes (03 after 00 00)."""
        s = tl.program_id(0)
        start = tl.load(tab + s * 4).to(tl.int64)
        size = tl.load(tab + s * 4 + 1)
        k = tl.arange(0, CH)
        n = tl.zeros([CH], tl.int32)
        for c in range(0, size, CH):
            p = c + k
            ok = (p >= 2) & (p < size)
            b0 = tl.load(raw + start + p - 2, mask=ok, other=1)
            b1 = tl.load(raw + start + p - 1, mask=ok, other=1)
            b2 = tl.load(raw + start + p, mask=ok, other=0)
            n += ((b0 == 0) & (b1 == 0) & (b2 == 3) & ok).to(tl.int32)
        tl.store(kept + s, size - tl.sum(n, 0))

    @triton.jit
    def _epb_copy_kernel(raw, tab, out_start, out, CH: tl.constexpr):
        """Per slice: its kept bytes from out_start[s] on, each written at its place in a big-endian 32-bit word."""
        s = tl.program_id(0)
        start = tl.load(tab + s * 4).to(tl.int64)
        size = tl.load(tab + s * 4 + 1)
        o = tl.load(out_start + s)
        k = tl.arange(0, CH)
        at = tl.zeros([], tl.int64)
        for c in range(0, size, CH):
            p = c + k
            inside = p < size
            b0 = tl.load(raw + start + p - 2, mask=inside & (p >= 2), other=1)
            b1 = tl.load(raw + start + p - 1, mask=inside & (p >= 1), other=1)
            b2 = tl.load(raw + start + p, mask=inside, other=0)
            keep = inside & ~((b0 == 0) & (b1 == 0) & (b2 == 3) & (p >= 2))
            rank = tl.cumsum(keep.to(tl.int32), 0) - 1
            dest = o + at + rank
            tl.store(out + ((dest >> 2) << 2) + 3 - (dest & 3), b2, mask=keep)  # bytes swapped within words
            at += tl.sum(keep.to(tl.int32), 0)


def prepare_device(raw, table, geom, device="cuda"):
    """gather's output -> a Batch on the device: emulation prevention bytes removed and each slice padded to whole
    32-bit words (bitstream order), by two kernels on the GPU (no host round trip)."""
    _require_triton()
    import torch

    H, W, coff = geom
    dev = torch.device(device)
    S = table.shape[0]
    raw = torch.as_tensor(raw).to(dev, non_blocking=True)
    tab = torch.as_tensor(table).to(dev, non_blocking=True)
    kept = torch.empty(S, dtype=torch.int32, device=dev)
    _epb_count_kernel[(S,)](raw, tab, kept, CH=256, num_warps=4)
    padded = (kept.long() + h264_parse.PAD + 3) // 4 * 4
    out_start = torch.cumsum(padded, 0) - padded
    bound = int(table[:, 1].astype(np.int64).sum()) + S * (h264_parse.PAD + 3) + 4  # no sync for the size
    out = torch.zeros(bound // 4 * 4, dtype=torch.uint8, device=dev)
    _epb_copy_kernel[(S,)](raw, tab, out_start, out, CH=256, num_warps=4)
    words = out.view(torch.int32)
    tab = tab.long()
    r = torch.arange(S, device=dev) % H
    f = torch.arange(S, device=dev) // H
    intra = (tab[:, 3] >> 7) & 1
    info = torch.stack(
        [out_start // 4, tab[:, 2], torch.where(intra == 1, 2, 0), tab[:, 3] & 127, r * W, r * W + W - 1, f, r], 1
    ).to(torch.int32)
    fi = (table[::H, 3] >> 7) & 1  # each frame: intra (a keyframe)
    F = len(fi)
    k = np.arange(F)
    last_key = np.maximum.accumulate(np.where(fi == 1, k, 0))
    ref_of = np.where(fi == 1, -1, k - 1).astype(np.int32)
    step = (k - last_key).astype(np.int32)
    return Batch(words, info, ref_of, step, H, W, coff, None)


if triton is not None:

    @triton.jit
    def _hpass_kernel(src, sf, ss, dst, df, rows_n, clip_of, tab_of, hs, cols, xs, NO, K: tl.constexpr,
                      WB: tl.constexpr):  # fmt: skip
        """Horizontal resize of one source row of one frame into dst (float32, rows_n rows per frame, NO wide):
        cols/xs[t]: weights and first source column per output column of region table t; hs[t]: the region's rows
        in this plane (rows past it are skipped). sf, ss: the source's frame and row strides."""
        pid = tl.program_id(0)
        f = pid // rows_n
        y = pid % rows_n
        t = tl.load(tab_of + tl.load(clip_of + f) * 2)
        if y < tl.load(hs + t):
            b = tl.arange(0, WB)
            ok = b < NO
            x0 = tl.load(xs + t * NO + b, mask=ok, other=0)
            row = src + f.to(tl.int64) * sf + y * ss
            acc = tl.zeros([WB], tl.float32)
            for k in tl.static_range(K):
                w = tl.load(cols + (t * NO + b) * K + k, mask=ok, other=0.0)
                acc += w * tl.load(row + x0 + k, mask=ok & (w != 0), other=0).to(tl.float32)
            tl.store(dst + (f.to(tl.int64) * rows_n + y) * NO + b, acc, mask=ok)

    @triton.jit
    def _vpass_rgb_kernel(iy, iu, iv, out, clip_of, tab_of, rows, ys, cy, cys, HY, HCH, NO, H: tl.constexpr,
                          W: tl.constexpr, WB: tl.constexpr, K: tl.constexpr):  # fmt: skip
        """One output row of one frame: vertical resize of the horizontally resized planes (iy: HY rows per frame,
        iu/iv: HCH), BT.601 limited range conversion, transposed back where the store transposed."""
        pid = tl.program_id(0)
        f = pid // H
        r = pid % H
        c = tl.arange(0, WB)
        ok = c < W
        clip = tl.load(clip_of + f)
        t = tl.load(tab_of + clip * 2)
        tr = tl.load(tab_of + clip * 2 + 1) == 1
        a = tl.where(tr, c, r)  # output row and column in coded orientation
        b = tl.where(tr, r, c)
        y0 = tl.load(ys + t * NO + a, mask=ok, other=0)
        c0 = tl.load(cys + t * NO + a, mask=ok, other=0)
        fy = iy + f.to(tl.int64) * HY * NO
        fc = f.to(tl.int64) * HCH * NO
        yv = tl.zeros([WB], tl.float32)
        u = tl.zeros([WB], tl.float32)
        v = tl.zeros([WB], tl.float32)
        for k in tl.static_range(K):
            w = tl.load(rows + (t * NO + a) * K + k, mask=ok, other=0.0)
            yv += w * tl.load(fy + (y0 + k) * NO + b, mask=ok & (w != 0), other=0.0)
            wc = tl.load(cy + (t * NO + a) * K + k, mask=ok, other=0.0)
            o = fc + (c0 + k) * NO + b
            m = ok & (wc != 0)
            u += wc * tl.load(iu + o, mask=m, other=0.0)
            v += wc * tl.load(iv + o, mask=m, other=0.0)
        yy = (yv - 16.0) * 1.164
        u = u - 128.0
        v = v - 128.0
        dst = out + (f.to(tl.int64) * H + r) * (W * 3) + c * 3
        tl.store(dst, (tl.minimum(tl.maximum(yy + 1.596 * v, 0.0), 255.0) + 0.5).to(tl.uint8), mask=ok)
        tl.store(dst + 1, (tl.minimum(tl.maximum(yy - 0.392 * u - 0.813 * v, 0.0), 255.0) + 0.5).to(tl.uint8),
                 mask=ok)  # fmt: skip
        tl.store(dst + 2, (tl.minimum(tl.maximum(yy + 2.017 * u, 0.0), 255.0) + 0.5).to(tl.uint8), mask=ok)


def _aa_weights(n_in, n_out, K):
    """torch's antialiased bilinear resize as (first source index, K weights) per output index (n_in -> n_out)."""
    scale = n_in / n_out
    support = scale if scale > 1 else 1.0
    inv = 1 / scale if scale > 1 else 1.0
    first = np.zeros(n_out, np.int32)
    w = np.zeros((n_out, K), np.float32)
    for i in range(n_out):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        k = np.arange(lo, hi)
        ww = np.maximum(0, 1 - np.abs((k - center + 0.5) * inv))
        if hi - lo > K:
            raise ValueError(f"resize {n_in} -> {n_out} needs {hi - lo} taps, more than {K}")
        first[i] = lo
        w[i, : hi - lo] = ww / ww.sum()
    return first, w


_RGB_TABLES = {}


def to_rgb(Y, U, V, regions, frames, out, K=8):
    """Decoded frames -> RGB clips resized into out (clips, frames, H, W, 3) uint8: clip c's frames are
    c * frames ... (c + 1) * frames - 1 of Y, U, V; regions[c] = (h, w, transposed), its valid region of the coded
    frame (stable_datasets.framestore.mp4's canvas). One kernel: antialiased bilinear resize (torch's weights) of the Y and chroma
    planes and BT.601 limited range conversion, transposed back where the store transposed."""
    _require_triton()
    import torch

    H, W = out.shape[2:4]
    dev = Y.device
    assert U.stride() == V.stride() and U.stride(1) == U.shape[2] and Y.is_contiguous()
    keys = [tuple(int(x) for x in r) for r in regions]
    uniq = sorted(set(keys))
    NO = max(H, W)
    tabs = []
    for h, w, tr in uniq:
        oh, ow = (W, H) if tr else (H, W)  # output size in coded orientation
        key = (h, w, oh, ow, K, NO)
        if key not in _RGB_TABLES:
            parts = []
            for n_in, n_out in ((h, oh), (w, ow), ((h + 1) // 2, oh), ((w + 1) // 2, ow)):
                first, wt = _aa_weights(n_in, n_out, K)
                fp = np.zeros(NO, np.int32)
                wp = np.zeros((NO, K), np.float32)
                fp[:n_out], wp[:n_out] = first, wt
                parts.append((fp, wp))
            _RGB_TABLES[key] = parts
        tabs.append(_RGB_TABLES[key])
    t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731
    rows = t(np.stack([p[0][1] for p in tabs]), torch.float32)
    ys = t(np.stack([p[0][0] for p in tabs]), torch.int32)
    cols = t(np.stack([p[1][1] for p in tabs]), torch.float32)
    xs = t(np.stack([p[1][0] for p in tabs]), torch.int32)
    cy = t(np.stack([p[2][1] for p in tabs]), torch.float32)
    cys = t(np.stack([p[2][0] for p in tabs]), torch.int32)
    cx = t(np.stack([p[3][1] for p in tabs]), torch.float32)
    cxs = t(np.stack([p[3][0] for p in tabs]), torch.int32)
    tab_of = t([[uniq.index(k), k[2]] for k in keys], torch.int32)
    hs_y = t([k[0] for k in uniq], torch.int32)
    hs_c = t([(k[0] + 1) // 2 for k in uniq], torch.int32)
    nf = len(keys) * frames
    clip_of = t(np.arange(nf) // frames, torch.int32)
    HY, HC = Y.shape[1], U.shape[1]
    WB = triton.next_power_of_2(NO)
    iy = torch.empty((nf, HY, NO), dtype=torch.float32, device=dev)
    iu = torch.empty((nf, HC, NO), dtype=torch.float32, device=dev)
    iv = torch.empty((nf, HC, NO), dtype=torch.float32, device=dev)
    _hpass_kernel[(nf * HY,)](Y, Y.stride(0), Y.stride(1), iy, 0, HY, clip_of, tab_of, hs_y, cols, xs, NO, K=K, WB=WB)
    for src, dst in ((U, iu), (V, iv)):
        _hpass_kernel[(nf * HC,)](
            src, src.stride(0), src.stride(1), dst, 0, HC, clip_of, tab_of, hs_c, cx, cxs, NO, K=K, WB=WB
        )  # fmt: skip
    _vpass_rgb_kernel[(nf * H,)](
        iy, iu, iv, out, clip_of, tab_of, rows, ys, cy, cys, HY, HC, NO, H=H, W=W, WB=triton.next_power_of_2(W), K=K,
    )  # fmt: skip
