"""Decoding restart-coded JPEGs on the GPU, one lane per MCU for the entropy decode (prototype).

A converted frame store (stable_datasets.framestore.convert, or stable_datasets.framestore.build) has a restart marker after every MCU: each MCU's
entropy-coded bits start at a known byte, with every DC predictor reset. So MCUs decode independently, with no
search for where a code starts (what general-purpose parallel Huffman decoders have to solve), and one lane per MCU
decodes its blocks' quantized DCT coefficients -- on the CUDA cores, where a GPU without JPEG engines leaves this to a
CPU core (nvJPEG's Huffman stage) and a GPU with them caps out at their rate.

    hdr = header(jpeg)                                  # tables and block layout of a frame (or a packed clip)
    coefs = decode_mcus(data, mcu_off, hdr)             # (n, blocks per MCU, 64) int16, natural order (Triton)
    coefs = decode_mcus_np(data, mcu_off, hdr)          # the same lane-parallel algorithm in numpy (the reference)

data: the bytes (a uint8 tensor / array, padded by at least 8 bytes); mcu_off: the byte where each MCU's entropy data
starts. Baseline, 8-bit, Huffman-coded, up to 3 components and 10 blocks per MCU; every table a frame's scan uses
(header() raises otherwise).

    pix = color_np(idct_np(coefs, hdr), hdr)            # (n, mcu_h, mcu_w, 3) uint8: libjpeg-turbo's pixels
    frame = frames_from_mcus(pix, rows, cols, h, w)     # a whole frame's MCUs, in raster order -> (h, w, 3)
"""

from contextlib import nullcontext

import numpy as np


ZIGZAG = np.array(  # zigzag index -> natural index
    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
     28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
     54, 47, 55, 62, 63],
    np.int32,
)  # fmt: skip
MAX_BLOCKS = 10


def _lut(counts, symbols):
    """A Huffman table as a 16-bit lookup: entry (length << 8) | symbol for every 16-bit window starting with a code."""
    lut = np.zeros(1 << 16, np.uint16)
    code, k = 0, 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            lo = code << (16 - length)
            lut[lo : lo + (1 << (16 - length))] = (length << 8) | symbols[k]
            code, k = code + 1, k + 1
        code <<= 1
    return lut


def header(jpeg):
    """A baseline JPEG's entropy-decoding facts: {"lut": (4, 65536) uint16 -- DC 0, DC 1, AC 0, AC 1 --, "blocks":
    (MAX_BLOCKS, 3) int32 per block of an MCU (DC table, AC table, component), "nblocks", "quant": (4, 64) natural
    order, "comps": [(h, v, quant table)], "mcus": (rows, cols), "size": (h, w), "sos_end": where the scan's data starts}."""
    b = np.frombuffer(bytes(jpeg), np.uint8)
    assert b[0] == 0xFF and b[1] == 0xD8, "not a JPEG"
    i, tabs, quant, comps, scan, size = 2, {}, np.zeros((4, 64), np.int32), [], None, None
    while scan is None:
        m, L = b[i + 1], (int(b[i + 2]) << 8) | int(b[i + 3])
        seg = b[i + 4 : i + 2 + L]
        if m == 0xC4:  # DHT
            j = 0
            while j < len(seg):
                tc, th = seg[j] >> 4, seg[j] & 15
                counts = seg[j + 1 : j + 17].astype(int)
                n = int(counts.sum())
                tabs[(int(tc), int(th))] = _lut(counts, seg[j + 17 : j + 17 + n].astype(int))
                j += 17 + n
        elif m == 0xDB:  # DQT
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                assert pq == 0, "16-bit quantization tables"
                quant[tq, ZIGZAG] = seg[j + 1 : j + 65]
                j += 65
        elif m == 0xC0:  # SOF0
            size = ((int(seg[1]) << 8) | int(seg[2]), (int(seg[3]) << 8) | int(seg[4]))
            for c in range(seg[5]):
                cid, hv, tq = seg[6 + 3 * c : 9 + 3 * c]
                comps.append((int(cid), int(hv >> 4), int(hv & 15), int(tq)))
        elif 0xC1 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            raise ValueError(f"not baseline Huffman (SOF {m:#x})")
        elif m == 0xDA:  # SOS
            scan = [(int(seg[1 + 2 * c]), int(seg[2 + 2 * c] >> 4), int(seg[2 + 2 * c] & 15)) for c in range(seg[0])]
        i += 2 + L
    assert size is not None and len(scan) == len(comps), "one interleaved scan of every component"
    lut = np.zeros((4, 1 << 16), np.uint16)
    for (tc, th), t in tabs.items():
        assert th < 2, "table ids 0 and 1 only"
        lut[2 * tc + th] = t
    blocks = []
    ids = [c[0] for c in comps]
    for cid, td, ta in scan:
        _, h, v, _ = comps[ids.index(cid)]
        for _ in range(h * v):
            blocks.append((td, 2 + ta, ids.index(cid)))
    assert len(blocks) <= MAX_BLOCKS
    hmax, vmax = max(c[1] for c in comps), max(c[2] for c in comps)
    rows, cols = -(-size[0] // (8 * vmax)), -(-size[1] // (8 * hmax))
    bl = np.zeros((MAX_BLOCKS, 3), np.int32)
    bl[: len(blocks)] = blocks
    return {
        "lut": lut,
        "blocks": bl,
        "nblocks": len(blocks),
        "quant": quant,
        "comps": [(h, v, tq) for _, h, v, tq in comps],
        "mcus": (rows, cols),
        "size": size,
        "sos_end": i,
    }


def mcu_offsets(jpeg, seg_len, sos_end):
    """Where each MCU's entropy data starts in a restart-coded JPEG: after the scan header, each segment followed by
    its 2-byte restart marker."""
    return sos_end + np.concatenate(([0], np.cumsum(np.asarray(seg_len, np.int64) + 2)[:-1]))


def decode_mcus_np(data, mcu_off, hdr):
    """The lane-parallel algorithm of the Triton kernel, in numpy: every MCU's blocks' coefficients, (n, nblocks, 64)
    int16 in natural order. Each step, every lane refills its bit buffer and decodes one Huffman symbol (and its
    value bits)."""
    data = np.concatenate([np.asarray(data, np.uint8), np.zeros(16, np.uint8)])
    n, nb_mcu = len(mcu_off), hdr["nblocks"]
    lut, blocks = hdr["lut"].astype(np.int64), hdr["blocks"]
    out = np.zeros((n, nb_mcu, 64), np.int16)
    bp = np.asarray(mcu_off, np.int64).copy()
    bb = np.zeros(n, np.uint64)
    nbits = np.zeros(n, np.int64)
    blk = np.zeros(n, np.int64)
    k = np.zeros(n, np.int64)
    pred = np.zeros((n, 3), np.int64)
    marker = np.zeros(n, bool)
    lanes = np.arange(n)
    active = np.ones(n, bool)
    while active.any():
        for _ in range(4):  # refill to at least 49 bits (a symbol and its value take at most 32)
            need = active & (nbits <= 48)
            byte = data[bp].astype(np.int64)
            nxt = data[bp + 1].astype(np.int64)
            ff = byte == 0xFF
            marker |= need & ff & (nxt != 0)  # a restart marker: the segment ended, feed zeros
            byte = np.where(marker, 0, byte)
            step = np.where(marker, 0, np.where(ff, 2, 1))
            bb = np.where(need, bb | (byte.astype(np.uint64) << (56 - nbits).astype(np.uint64)), bb)
            bp = np.where(need, bp + step, bp)
            nbits = np.where(need, nbits + 8, nbits)
        cur = np.minimum(blk, nb_mcu - 1)
        dc = k == 0
        tab = np.where(dc, blocks[cur, 0], blocks[cur, 1])
        e = lut[tab, (bb >> np.uint64(48)).astype(np.int64)]
        length, sym = e >> 8, e & 255
        bb = bb << length.astype(np.uint64)
        nbits -= length
        r, s = np.where(dc, 0, sym >> 4), np.where(dc, sym, sym & 15)
        val = (bb >> (64 - s).astype(np.uint64)).astype(np.int64)
        val = np.where(s == 0, 0, val)
        bb = np.where(s > 0, bb << s.astype(np.uint64), bb)
        nbits -= s
        val = np.where(val < (1 << np.maximum(s - 1, 0)), val - (1 << s) + 1, val)
        val = np.where(s == 0, 0, val)
        comp = blocks[cur, 2]
        p = pred[lanes, comp] + val
        pred[lanes, comp] = np.where(active & dc, p, pred[lanes, comp])
        kk = np.where(dc, 0, k + r)
        put = active & (dc | (s > 0)) & (kk < 64)
        out[lanes[put], cur[put], ZIGZAG[kk[put]]] = np.where(dc, p, val)[put]
        k = np.where(dc, 1, np.where(s > 0, kk + 1, np.where(r == 15, k + 16, 64)))  # ZRL or EOB
        nxt_blk = active & (k >= 64)
        blk = np.where(nxt_blk, blk + 1, blk)
        k = np.where(nxt_blk, 0, k)
        active &= blk < nb_mcu
    return out


# ---------------------------------------------------------------------------------------------------------------
# dequantization + IDCT (libjpeg's accurate integer IDCT, jidctint.c) and upsampling + colour (jdcolor.c / jdmerge.c):
# bit-exact with libjpeg-turbo's islow IDCT and its plain (not "fancy") upsampling, the mode nvJPEG also matches
# ---------------------------------------------------------------------------------------------------------------
CONST_BITS, PASS1_BITS = 13, 2
F = {  # FIX(x) = round(x * 2^13)
    "0_298631336": 2446, "0_390180644": 3196, "0_541196100": 4433, "0_765366865": 6270, "0_899976223": 7373,
    "1_175875602": 9633, "1_501321110": 12299, "1_847759065": 15137, "1_961570560": 16069, "2_053119869": 16819,
    "2_562915447": 20995, "3_072711026": 25172,
}  # fmt: skip


def _descale(x, n):
    return x if n == 0 else (x + (1 << (n - 1))) >> n


def _idct_1d(x, n_out, xp=np):
    """One 1-D pass of jpeg_idct_islow on x[0..7] (arrays), outputs descaled by n_out bits."""
    z1 = (x[2] + x[6]) * F["0_541196100"]
    tmp2 = z1 + x[6] * -F["1_847759065"]
    tmp3 = z1 + x[2] * F["0_765366865"]
    tmp0 = (x[0] + x[4]) << CONST_BITS
    tmp1 = (x[0] - x[4]) << CONST_BITS
    t10, t13, t11, t12 = tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2
    o0, o1, o2, o3 = x[7], x[5], x[3], x[1]
    z1, z2, z3, z4 = o0 + o3, o1 + o2, o0 + o2, o1 + o3
    z5 = (z3 + z4) * F["1_175875602"]
    o0, o1, o2, o3 = o0 * F["0_298631336"], o1 * F["2_053119869"], o2 * F["3_072711026"], o3 * F["1_501321110"]
    z1, z2 = z1 * -F["0_899976223"], z2 * -F["2_562915447"]
    z3, z4 = z3 * -F["1_961570560"] + z5, z4 * -F["0_390180644"] + z5
    o0, o1, o2, o3 = o0 + z1 + z3, o1 + z2 + z4, o2 + z2 + z3, o3 + z1 + z4
    return [
        _descale(t10 + o3, n_out), _descale(t11 + o2, n_out), _descale(t12 + o1, n_out), _descale(t13 + o0, n_out),
        _descale(t13 - o0, n_out), _descale(t12 - o1, n_out), _descale(t11 - o2, n_out), _descale(t10 - o3, n_out),
    ]  # fmt: skip


def _idct_matrix():
    """jpeg_idct_islow's 1-D pass as an exact integer matrix: within a pass it only adds and multiplies by constants,
    rounding once at the end, so pass(x) = descale(IDCT_M @ x, n). (8, 8) int64, out x in."""
    eye = [np.eye(8, dtype=np.int64)[k] for k in range(8)]
    return np.stack(_idct_1d(eye, 0), 0)


IDCT_M = None  # set below, once _idct_1d is defined


def idct_matrix_np(coefs, hdr):
    """idct_np through IDCT_M (the fused kernel's formulation): equal to it bit for bit."""
    q = hdr["quant"][block_quant(hdr)].astype(np.int64)
    x = (coefs.astype(np.int64) * q).reshape(*coefs.shape[:2], 8, 8)  # [..., k (row), c]
    ws = _descale(np.einsum("jk,...kc->...jc", IDCT_M, x), CONST_BITS - PASS1_BITS)  # columns
    out = _descale(np.einsum("...rc,jc->...rj", ws, IDCT_M), CONST_BITS + PASS1_BITS + 3)  # rows
    return _range_limit(out).astype(np.uint8).reshape(coefs.shape)


def fusable(hdr):
    """Whether the fused pixel kernel takes this layout: gray, or luma at the MCU's full size with each chroma
    component one block per MCU (4:2:0, 4:2:2, 4:4:0, 4:4:4)."""
    comps = hdr["comps"]
    return len(comps) == 1 or (
        len(comps) == 3
        and (comps[0][0], comps[0][1]) == (max(c[0] for c in comps), max(c[1] for c in comps))
        and all((h, v) == (1, 1) for h, v, _ in comps[1:])
    )


def pixels_np(coefs, hdr):
    """color_np(idct_np(coefs)) as the fused kernel computes it: per luma block, its upsampled chroma tiles come from
    the chroma IDCT with row- and column-selected copies of IDCT_M (plain upsampling picks rows and columns, and picking
    rows of a matrix product is the product with picked rows), so no gather is needed. Equal bit for bit."""
    comps = hdr["comps"]
    H0, V0 = comps[0][0], comps[0][1]
    q = hdr["quant"][block_quant(hdr)].astype(np.int64)
    x = (coefs.astype(np.int64) * q).reshape(*coefs.shape[:2], 8, 8)
    n = coefs.shape[0]

    def idct8(xb, mr, mc):
        ws = _descale(np.einsum("rk,nkc->nrc", mr, xb), CONST_BITS - PASS1_BITS)
        return _range_limit(_descale(np.einsum("nrc,jc->nrj", ws, mc), CONST_BITS + PASS1_BITS + 3))

    out = np.zeros((n, 8 * V0, 8 * H0, 1 if len(comps) == 1 else 3), np.uint8)
    rr = np.arange(8)
    for by in range(V0):
        for bx in range(H0):
            y = idct8(x[:, by * H0 + bx], IDCT_M, IDCT_M)
            if len(comps) == 1:
                out[:, by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8, 0] = y
                continue
            mr, mc = IDCT_M[(by * 8 + rr) // V0], IDCT_M[(bx * 8 + rr) // H0]
            cb = idct8(x[:, H0 * V0], mr, mc) - 128
            cr = idct8(x[:, H0 * V0 + 1], mr, mc) - 128
            half = 1 << 15
            r = y + ((_fix16(1.40200) * cr + half) >> 16)
            b = y + ((_fix16(1.77200) * cb + half) >> 16)
            g = y + ((-_fix16(0.34414) * cb + half - _fix16(0.71414) * cr) >> 16)
            out[:, by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8] = np.clip(np.stack([r, g, b], -1), 0, 255)
    return out


def _range_limit(x):
    """libjpeg's post-IDCT range limit: x + 128 clamped to 0..255 (wrapping as its 1024-entry table does)."""
    m = x & 1023
    return np.where(m < 128, m + 128, np.where(m < 512, 255, np.where(m < 896, 0, m - 896)))


IDCT_M = _idct_matrix()


def block_quant(hdr):
    """Per block of an MCU, its quantization table's index."""
    return np.array([hdr["comps"][c][2] for c in hdr["blocks"][: hdr["nblocks"], 2]], np.int32)


def idct_np(coefs, hdr):
    """(n, nblocks, 64) quantized coefficients -> (n, nblocks, 64) uint8 samples (row-major 8 x 8 per block)."""
    q = hdr["quant"][block_quant(hdr)].astype(np.int64)  # (nblocks, 64)
    x = (coefs.astype(np.int64) * q).reshape(*coefs.shape[:2], 8, 8)  # [..., row, col]
    ws = _idct_1d([x[..., k, :] for k in range(8)], CONST_BITS - PASS1_BITS)  # columns: ws[k][..., col]
    rows = [np.stack([ws[k][..., c] for c in range(8)], -1) for k in range(8)]  # rows[r][..., col]
    out = np.zeros(x.shape, np.int64)
    for r in range(8):
        res = _idct_1d([rows[r][..., c] for c in range(8)], CONST_BITS + PASS1_BITS + 3)
        for c in range(8):
            out[..., r, c] = res[c]
    return _range_limit(out).astype(np.uint8).reshape(coefs.shape)


def _fix16(x):
    return int(x * 65536 + 0.5)


def color_np(samples, hdr):
    """(n, nblocks, 64) samples -> (n, mcu_h, mcu_w, channels) uint8: chroma replicated (libjpeg-turbo's plain
    upsampling), YCbCr -> RGB with libjpeg's fixed-point tables; gray frames give one channel."""
    comps = hdr["comps"]
    hmax, vmax = max(c[0] for c in comps), max(c[1] for c in comps)
    MH, MW = 8 * vmax, 8 * hmax
    py, px = np.mgrid[:MH, :MW]
    first = np.cumsum([0] + [h * v for h, v, _ in comps])
    planes = []
    for c, (h, v, _) in enumerate(comps):
        sy, sx = py * v // vmax, px * h // hmax
        planes.append(samples[:, first[c] + (sy // 8) * h + sx // 8, (sy % 8) * 8 + sx % 8].astype(np.int64))
    if len(comps) == 1:
        return planes[0].astype(np.uint8)[..., None]
    y, cb, cr = planes[0], planes[1] - 128, planes[2] - 128
    half = 1 << 15
    r = y + ((_fix16(1.40200) * cr + half) >> 16)
    b = y + ((_fix16(1.77200) * cb + half) >> 16)
    g = y + ((-_fix16(0.34414) * cb + half - _fix16(0.71414) * cr) >> 16)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def frames_from_mcus(tiles, rows, cols, h, w):
    """MCU tiles (rows * cols, mcu_h, mcu_w, ch) of one frame, in raster order -> the (h, w, ch) frame."""
    _, mh, mw, ch = tiles.shape
    return tiles.reshape(rows, cols, mh, mw, ch).transpose(0, 2, 1, 3, 4).reshape(rows * mh, cols * mw, ch)[:h, :w]


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _decode_kernel(data, mcu_off, n, lut, blocks, zigzag, out, NB: tl.constexpr, LANES: tl.constexpr):
        lane = tl.program_id(0) * LANES + tl.arange(0, LANES)
        active = lane < n
        bp = tl.load(mcu_off + lane, mask=active, other=0)
        bb = tl.zeros([LANES], tl.uint64)
        nbits = tl.zeros([LANES], tl.int32)
        blk = tl.zeros([LANES], tl.int32)
        k = tl.zeros([LANES], tl.int32)
        p0 = tl.zeros([LANES], tl.int32)
        p1 = tl.zeros([LANES], tl.int32)
        p2 = tl.zeros([LANES], tl.int32)
        marker = tl.zeros([LANES], tl.int1)
        base = out + lane.to(tl.int64) * (NB * 64)
        while tl.max(active.to(tl.int32), 0) > 0:
            for _ in tl.static_range(4):
                need = active & (nbits <= 48)
                byte = tl.load(data + bp, mask=need, other=0).to(tl.int32)
                nxt = tl.load(data + bp + 1, mask=need, other=0).to(tl.int32)
                ff = byte == 0xFF
                marker = marker | (need & ff & (nxt != 0))
                byte = tl.where(marker, 0, byte)
                step = tl.where(marker, 0, tl.where(ff, 2, 1))
                bb = tl.where(need, bb | (byte.to(tl.uint64) << (56 - nbits).to(tl.uint64)), bb)
                bp = tl.where(need, bp + step, bp)
                nbits = tl.where(need, nbits + 8, nbits)
            cur = tl.minimum(blk, NB - 1)
            dc = k == 0
            tdc = tl.load(blocks + cur * 3, mask=active, other=0)
            tac = tl.load(blocks + cur * 3 + 1, mask=active, other=0)
            comp = tl.load(blocks + cur * 3 + 2, mask=active, other=0)
            tab = tl.where(dc, tdc, tac)
            e = tl.load(lut + tab * 65536 + (bb >> 48).to(tl.int32), mask=active, other=0).to(tl.int32)
            length = e >> 8
            sym = e & 255
            bb = bb << length.to(tl.uint64)
            nbits -= length
            r = tl.where(dc, 0, sym >> 4)
            s = tl.where(dc, sym, sym & 15)
            val = tl.where(s > 0, (bb >> (64 - s).to(tl.uint64)).to(tl.int32), 0)
            bb = tl.where(s > 0, bb << s.to(tl.uint64), bb)
            nbits -= s
            half = tl.where(s > 0, 1 << tl.maximum(s - 1, 0), 0)
            val = tl.where((s > 0) & (val < half), val - (1 << s) + 1, val)
            pred = tl.where(comp == 0, p0, tl.where(comp == 1, p1, p2)) + val
            upd = active & dc
            p0 = tl.where(upd & (comp == 0), pred, p0)
            p1 = tl.where(upd & (comp == 1), pred, p1)
            p2 = tl.where(upd & (comp == 2), pred, p2)
            kk = tl.where(dc, 0, k + r)
            put = active & (dc | (s > 0)) & (kk < 64)
            zz = tl.load(zigzag + tl.minimum(kk, 63), mask=put, other=0)
            tl.store(base + cur * 64 + zz, tl.where(dc, pred, val).to(tl.int16), mask=put)
            k = tl.where(dc, 1, tl.where(s > 0, kk + 1, tl.where(r == 15, k + 16, 64)))
            nb = active & (k >= 64)
            blk = tl.where(nb, blk + 1, blk)
            k = tl.where(nb, 0, k)
            active = active & (blk < NB)

    @triton.jit
    def _idct_1d_tl(x0, x1, x2, x3, x4, x5, x6, x7, N_OUT: tl.constexpr):
        z1 = (x2 + x6) * 4433
        tmp2 = z1 + x6 * -15137
        tmp3 = z1 + x2 * 6270
        tmp0 = (x0 + x4) << 13
        tmp1 = (x0 - x4) << 13
        t10 = tmp0 + tmp3
        t13 = tmp0 - tmp3
        t11 = tmp1 + tmp2
        t12 = tmp1 - tmp2
        z1 = x7 + x1
        z2 = x5 + x3
        z3 = x7 + x3
        z4 = x5 + x1
        z5 = (z3 + z4) * 9633
        o0 = x7 * 2446
        o1 = x5 * 16819
        o2 = x3 * 25172
        o3 = x1 * 12299
        z1 = z1 * -7373
        z2 = z2 * -20995
        z3 = z3 * -16069 + z5
        z4 = z4 * -3196 + z5
        o0 = o0 + z1 + z3
        o1 = o1 + z2 + z4
        o2 = o2 + z2 + z3
        o3 = o3 + z1 + z4
        r: tl.constexpr = 1 << (N_OUT - 1)
        return (
            (t10 + o3 + r) >> N_OUT,
            (t11 + o2 + r) >> N_OUT,
            (t12 + o1 + r) >> N_OUT,
            (t13 + o0 + r) >> N_OUT,
            (t13 - o0 + r) >> N_OUT,
            (t12 - o1 + r) >> N_OUT,
            (t11 - o2 + r) >> N_OUT,
            (t10 - o3 + r) >> N_OUT,
        )

    @triton.jit
    def _range_limit_tl(x):
        m = x & 1023
        return tl.where(m < 128, m + 128, tl.where(m < 512, 255, tl.where(m < 896, 0, m - 896)))

    @triton.jit
    def _idct_kernel(coefs, bq, quant, scratch, out, total, NB: tl.constexpr, P: tl.constexpr):
        g = tl.program_id(0) * P + tl.arange(0, P)[:, None]  # blocks
        c = tl.arange(0, 8)[None, :]
        ok = (g < total) & (c < 8)
        qb = quant + tl.load(bq + g % NB, mask=g < total, other=0) * 64
        src = coefs + g.to(tl.int64) * 64
        ws = scratch + g.to(tl.int64) * 64
        # pass 1: columns; x_k = row k of every column
        w0, w1, w2, w3, w4, w5, w6, w7 = _idct_1d_tl(
            tl.load(src + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + c, mask=ok, other=0),
            tl.load(src + 8 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 8 + c, mask=ok, other=0),
            tl.load(src + 16 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 16 + c, mask=ok, other=0),
            tl.load(src + 24 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 24 + c, mask=ok, other=0),
            tl.load(src + 32 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 32 + c, mask=ok, other=0),
            tl.load(src + 40 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 40 + c, mask=ok, other=0),
            tl.load(src + 48 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 48 + c, mask=ok, other=0),
            tl.load(src + 56 + c, mask=ok, other=0).to(tl.int32) * tl.load(qb + 56 + c, mask=ok, other=0),
            11,
        )
        tl.store(ws + c, w0, mask=ok)
        tl.store(ws + 8 + c, w1, mask=ok)
        tl.store(ws + 16 + c, w2, mask=ok)
        tl.store(ws + 24 + c, w3, mask=ok)
        tl.store(ws + 32 + c, w4, mask=ok)
        tl.store(ws + 40 + c, w5, mask=ok)
        tl.store(ws + 48 + c, w6, mask=ok)
        tl.store(ws + 56 + c, w7, mask=ok)
        tl.debug_barrier()
        # pass 2: rows (now indexed by c); y_j = column j of every row
        r = c * 8
        y0, y1, y2, y3, y4, y5, y6, y7 = _idct_1d_tl(
            tl.load(ws + r, mask=ok, other=0),
            tl.load(ws + r + 1, mask=ok, other=0),
            tl.load(ws + r + 2, mask=ok, other=0),
            tl.load(ws + r + 3, mask=ok, other=0),
            tl.load(ws + r + 4, mask=ok, other=0),
            tl.load(ws + r + 5, mask=ok, other=0),
            tl.load(ws + r + 6, mask=ok, other=0),
            tl.load(ws + r + 7, mask=ok, other=0),
            18,
        )
        dst = out + g.to(tl.int64) * 64 + r
        tl.store(dst, _range_limit_tl(y0).to(tl.uint8), mask=ok)
        tl.store(dst + 1, _range_limit_tl(y1).to(tl.uint8), mask=ok)
        tl.store(dst + 2, _range_limit_tl(y2).to(tl.uint8), mask=ok)
        tl.store(dst + 3, _range_limit_tl(y3).to(tl.uint8), mask=ok)
        tl.store(dst + 4, _range_limit_tl(y4).to(tl.uint8), mask=ok)
        tl.store(dst + 5, _range_limit_tl(y5).to(tl.uint8), mask=ok)
        tl.store(dst + 6, _range_limit_tl(y6).to(tl.uint8), mask=ok)
        tl.store(dst + 7, _range_limit_tl(y7).to(tl.uint8), mask=ok)

    @triton.jit
    def _plane(samples, m, py, px, ok, FIRST: tl.constexpr, H: tl.constexpr, V: tl.constexpr, HMAX: tl.constexpr,
               VMAX: tl.constexpr, NB: tl.constexpr):  # fmt: skip
        sy = py * V // VMAX
        sx = px * H // HMAX
        blk = FIRST + (sy // 8) * H + sx // 8
        return tl.load(samples + m * (NB * 64) + blk * 64 + (sy % 8) * 8 + sx % 8, mask=ok, other=0).to(tl.int32)

    @triton.jit
    def _color_kernel(samples, out, out_off, sy, sx, sc, n, NB: tl.constexpr, NCOMP: tl.constexpr, H0: tl.constexpr,
                      V0: tl.constexpr, H1: tl.constexpr, V1: tl.constexpr, H2: tl.constexpr, V2: tl.constexpr,
                      HMAX: tl.constexpr, VMAX: tl.constexpr, P: tl.constexpr, PIX: tl.constexpr):  # fmt: skip
        """Upsampling + colour of P MCUs, written at out + out_off[m] + y * sy + x * sx + channel * sc (MCU tiles, or
        the frame canvas)."""
        m = tl.program_id(0) * P + tl.arange(0, P)[:, None]
        pix = tl.arange(0, PIX)[None, :]
        MW: tl.constexpr = 8 * HMAX
        py = pix // MW
        px = pix % MW
        ok = (m < n) & (pix < PIX)
        m64 = m.to(tl.int64)
        dst = out + tl.load(out_off + m, mask=m < n, other=0) + py * sy + px * sx
        y = _plane(samples, m64, py, px, ok, 0, H0, V0, HMAX, VMAX, NB)
        if NCOMP == 1:
            tl.store(dst, y.to(tl.uint8), mask=ok)
        else:
            cb = _plane(samples, m64, py, px, ok, H0 * V0, H1, V1, HMAX, VMAX, NB) - 128
            cr = _plane(samples, m64, py, px, ok, H0 * V0 + H1 * V1, H2, V2, HMAX, VMAX, NB) - 128
            r = tl.minimum(tl.maximum(y + ((91881 * cr + 32768) >> 16), 0), 255)
            b = tl.minimum(tl.maximum(y + ((116130 * cb + 32768) >> 16), 0), 255)
            g = tl.minimum(tl.maximum(y + ((-22554 * cb + 32768 - 46802 * cr) >> 16), 0), 255)
            tl.store(dst, r.to(tl.uint8), mask=ok)
            tl.store(dst + sc, g.to(tl.uint8), mask=ok)
            tl.store(dst + 2 * sc, b.to(tl.uint8), mask=ok)

    @triton.jit
    def _idct8_tl(x, mr, mc):
        """x (P, 8, 8) dequantized coefficients -> samples: descale(descale(mr @ x) @ mc^T), range-limited."""
        ws = (tl.sum(mr[None, :, :, None] * x[:, None, :, :], axis=2) + 1024) >> 11
        o = (tl.sum(ws[:, :, None, :] * mc[None, None, :, :], axis=3) + 131072) >> 18
        return _range_limit_tl(o)

    @triton.jit
    def _pixels_kernel(coefs, bq, quant, mat, out, out_off, n, sy, sx, sc, NB: tl.constexpr, NCOMP: tl.constexpr,
                       H0: tl.constexpr, V0: tl.constexpr, P: tl.constexpr):  # fmt: skip
        """Dequantization, the IDCT, plain upsampling and colour of P MCUs, written at out + out_off[m] + y * sy +
        x * sx + channel * sc (MCU tiles, or the frame canvas). Chroma tiles per luma block come from the chroma IDCT
        with row/column-selected IDCT matrices (pixels_np)."""
        m = tl.program_id(0) * P + tl.arange(0, P)[:, None, None]
        ok = m < n
        k = tl.arange(0, 8)[None, :, None]
        c = tl.arange(0, 8)[None, None, :]
        rr = tl.arange(0, 8)
        full = tl.load(mat + rr[:, None] * 8 + rr[None, :])
        src = coefs + m.to(tl.int64) * (NB * 64)
        dst = out + tl.load(out_off + m, mask=ok, other=0)
        for by in tl.static_range(V0):
            for bx in tl.static_range(H0):
                b = by * H0 + bx
                qy = quant + tl.load(bq + b) * 64
                x = tl.load(src + b * 64 + k * 8 + c, mask=ok, other=0).to(tl.int32) * tl.load(qy + k * 8 + c)
                y = _idct8_tl(x, full, full)
                at = dst + (by * 8 + k) * sy + (bx * 8 + c) * sx
                if NCOMP == 1:
                    tl.store(at, y.to(tl.uint8), mask=ok)
                else:
                    mr = tl.load(mat + ((by * 8 + rr) // V0)[:, None] * 8 + rr[None, :])
                    mc = tl.load(mat + ((bx * 8 + rr) // H0)[:, None] * 8 + rr[None, :])
                    qb = quant + tl.load(bq + H0 * V0) * 64
                    qr = quant + tl.load(bq + H0 * V0 + 1) * 64
                    xb = tl.load(src + (H0 * V0) * 64 + k * 8 + c, mask=ok, other=0).to(tl.int32)
                    xr = tl.load(src + (H0 * V0 + 1) * 64 + k * 8 + c, mask=ok, other=0).to(tl.int32)
                    cb = _idct8_tl(xb * tl.load(qb + k * 8 + c), mr, mc) - 128
                    cr = _idct8_tl(xr * tl.load(qr + k * 8 + c), mr, mc) - 128
                    red = tl.minimum(tl.maximum(y + ((91881 * cr + 32768) >> 16), 0), 255)
                    blue = tl.minimum(tl.maximum(y + ((116130 * cb + 32768) >> 16), 0), 255)
                    green = tl.minimum(tl.maximum(y + ((-22554 * cb + 32768 - 46802 * cr) >> 16), 0), 255)
                    tl.store(at, red.to(tl.uint8), mask=ok)
                    tl.store(at + sc, green.to(tl.uint8), mask=ok)
                    tl.store(at + 2 * sc, blue.to(tl.uint8), mask=ok)


def decode_mcus(data, mcu_off, hdr, lanes=64):
    """decode_mcus_np on the GPU (Triton): data a uint8 CUDA tensor (padded by 8 bytes), mcu_off int64. The header's
    tables are copied to the device once and kept in hdr."""
    import torch

    lut, blocks, zz, *_ = _device_tables(hdr, data.device)
    dev = data.device
    n, nb = len(mcu_off), hdr["nblocks"]
    out = torch.zeros(n, nb, 64, dtype=torch.int16, device=dev)
    _decode_kernel[(triton.cdiv(n, lanes),)](data, mcu_off, n, lut, blocks, zz, out, NB=nb, LANES=lanes)
    return out


def _device_tables(hdr, dev):
    import torch

    cache = hdr.setdefault("_device", {})
    if dev not in cache:
        cache[dev] = (
            torch.from_numpy(hdr["lut"].astype(np.int32)).to(dev),
            torch.from_numpy(hdr["blocks"]).to(dev),
            torch.from_numpy(ZIGZAG).to(dev),
            torch.from_numpy(block_quant(hdr)).to(dev),
            torch.from_numpy(hdr["quant"].astype(np.int32)).to(dev),
        )
    return cache[dev]


def idct(coefs, hdr, blocks_per_program=32):
    """idct_np on the GPU: (n, nblocks, 64) int16 -> (n, nblocks, 64) uint8 samples."""
    import torch

    *_, bq, quant = _device_tables(hdr, coefs.device)
    total = coefs.shape[0] * coefs.shape[1]
    scratch = torch.empty(total, 64, dtype=torch.int32, device=coefs.device)
    out = torch.empty(coefs.shape, dtype=torch.uint8, device=coefs.device)
    P = blocks_per_program
    _idct_kernel[(triton.cdiv(total, P),)](coefs, bq, quant, scratch, out, total, NB=hdr["nblocks"], P=P)
    return out


def color_into(samples, hdr, out, out_off, sy, sx, sc, mcus_per_program=4):
    """color_np on the GPU, written at out's element out_off[m] + y * sy + x * sx + channel * sc for MCU m (MCU
    tiles, or straight into a frame canvas)."""
    comps = hdr["comps"]
    hmax, vmax = max(c[0] for c in comps), max(c[1] for c in comps)
    hv = [(h, v) for h, v, _ in comps] + [(1, 1)] * (3 - len(comps))
    n, P = samples.shape[0], mcus_per_program
    _color_kernel[(triton.cdiv(n, P),)](
        samples, out, out_off, sy, sx, sc, n, NB=hdr["nblocks"], NCOMP=len(comps), H0=hv[0][0], V0=hv[0][1],
        H1=hv[1][0], V1=hv[1][1], H2=hv[2][0], V2=hv[2][1], HMAX=hmax, VMAX=vmax, P=P, PIX=64 * hmax * vmax,
    )  # fmt: skip


def color(samples, hdr, mcus_per_program=4):
    """color_np on the GPU: (n, nblocks, 64) uint8 samples -> (n, mcu_h, mcu_w, channels) uint8."""
    import torch

    comps = hdr["comps"]
    hmax, vmax = max(c[0] for c in comps), max(c[1] for c in comps)
    n, ch = samples.shape[0], 1 if len(comps) == 1 else 3
    mh, mw = 8 * vmax, 8 * hmax
    out = torch.empty(n, mh, mw, ch, dtype=torch.uint8, device=samples.device)
    off = torch.arange(n, device=samples.device, dtype=torch.int64) * (mh * mw * ch)
    color_into(samples, hdr, out, off, mw * ch, ch, 1, mcus_per_program)
    return out


def pixels_into(coefs, hdr, out, out_off, sy, sx, sc, mcus_per_program=1, num_warps=1):
    """pixels_np on the GPU, one kernel (fusable(hdr) layouts): MCU m's pixels written at out's element out_off[m] +
    y * sy + x * sx + channel * sc -- MCU tiles, or straight into a frame canvas. One MCU and one warp per program is
    fastest (H100, a 64-clip token batch of 238k MCUs: 0.98 ms, against 4.4 ms at 4 warps and 1.9 ms for the IDCT and
    colour kernels apart)."""
    import torch

    *_, bq, quant = _device_tables(hdr, coefs.device)
    cache = hdr.setdefault("_device", {})
    if ("mat", coefs.device) not in cache:
        cache[("mat", coefs.device)] = torch.from_numpy(IDCT_M.astype(np.int32)).to(coefs.device)
    comps = hdr["comps"]
    n, P = coefs.shape[0], mcus_per_program
    _pixels_kernel[(triton.cdiv(n, P),)](
        coefs, bq, quant, cache[("mat", coefs.device)], out, out_off, n, sy, sx, sc, NB=hdr["nblocks"],
        NCOMP=len(comps), H0=comps[0][0], V0=comps[0][1], P=P, num_warps=num_warps,
    )  # fmt: skip


def decode_rgb(data, mcu_off, hdr):
    """Every MCU's pixels on the GPU: (n, mcu_h, mcu_w, channels) uint8, libjpeg-turbo's (islow IDCT, plain
    upsampling)."""
    import torch

    coefs = decode_mcus(data, mcu_off, hdr)
    if not fusable(hdr):
        return color(idct(coefs, hdr), hdr)
    comps = hdr["comps"]
    mh, mw, ch = 8 * comps[0][1], 8 * comps[0][0], 1 if len(comps) == 1 else 3
    tiles = torch.empty(coefs.shape[0], mh, mw, ch, dtype=torch.uint8, device=coefs.device)
    off = torch.arange(coefs.shape[0], device=coefs.device, dtype=torch.int64) * (mh * mw * ch)
    pixels_into(coefs, hdr, tiles, off, mw * ch, ch, 1)
    return tiles


# ---------------------------------------------------------------------------------------------------------------
# whole frames: the loaders' entry point
# ---------------------------------------------------------------------------------------------------------------
def _table_keys(buf, off, sos_end):
    """Per frame, a key of its header with the frame size left out (equal keys: equal tables and sampling)."""
    import hashlib

    keys = []
    for f in range(len(off) - 1):
        h = bytes(buf[off[f] : sos_end[f]])
        i, parts = 2, []
        while i + 4 <= len(h):
            m, L = h[i + 1], (h[i + 2] << 8) | h[i + 3]
            if m in (0xC4, 0xDB, 0xDA):
                parts.append(h[i : i + 2 + L])
            elif m == 0xC0:
                parts.append(h[i + 9 : i + 2 + L])  # components and sampling, not the size
            i += 2 + L
        keys.append(hashlib.blake2b(b"".join(parts), digest_size=8).digest())
    return keys


def decode_buffer(buf, off, device="cuda", stream=None, prof=None):
    """Whole restart-coded frames on the GPU: buf a uint8 CPU tensor holding the frames back to back (page-locked
    for an asynchronous copy), off their offsets (len + 1) -> list of (3, h, w) uint8 tensors on the device,
    libjpeg-turbo's pixels (islow IDCT, plain upsampling).

    The headers are read on the CPU; every MCU's start is found on the GPU, from the restart markers (0xFF 0xD0-D7
    occur in entropy-coded data only as markers: a data 0xFF is followed by a stuffed 0x00), so no segment lengths
    are needed. Frames whose tables differ decode in separate launches. ValueError for a frame without a restart
    marker after every MCU (convert the store first). prof: a dict that collects each step's synchronized seconds
    (for profiling; it slows the decode)."""
    import time

    import torch

    from . import codec

    t = [time.perf_counter()]

    def mark(step):
        if prof is not None:
            torch.cuda.synchronize()
            now = time.perf_counter()
            prof[step] = prof.get(step, 0.0) + now - t[0]
            t[0] = now

    off = np.asarray(off, np.int64)
    host = buf.numpy()
    info = codec.probe(host, off)
    if not (info["ok"] == 1).all():
        raise ValueError("a frame is not a baseline JPEG this decoder reads")
    sos, nm = info["sos_end"], info["rows"] * info["cols"]
    n = int(off[-1])
    mark("headers (CPU)")
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        d = torch.zeros(n + 16, dtype=torch.uint8, device=device)
        d[:n].copy_(buf[:n], non_blocking=True)
        mark("bytes to the GPU")
        rst = torch.nonzero((d[: n - 1] == 0xFF) & (d[1:n] >= 0xD0) & (d[1:n] <= 0xD7)).squeeze(1) + 2
        f = torch.searchsorted(torch.from_numpy(off[1:]).to(device), rst, right=True)
        rst = rst[rst >= torch.from_numpy(sos).to(device)[f]]  # markers in the scans, not in a header's bytes
        starts = torch.sort(torch.cat([torch.from_numpy(sos).to(device), rst])).values
        if len(starts) != int(nm.sum()):
            raise ValueError(
                "frames without a restart marker after every MCU: convert the store (stable_datasets.framestore.convert)"
            )
        mark("MCU starts (restart markers)")
        first = np.concatenate(([0], np.cumsum(nm)))
        keys = _table_keys(host, off, sos)
        mark("table keys (CPU)")
        out = [None] * (len(off) - 1)
        for key in dict.fromkeys(keys):
            fr = [i for i, k in enumerate(keys) if k == key]
            hdr = _HEADERS.get(key)
            if hdr is None:
                hdr = _HEADERS[key] = header(host[off[fr[0]] : sos[fr[0]]].tobytes() + b"\xff\xd9")
            if len(fr) == len(keys):
                sel = starts
            else:
                sel = starts[
                    torch.from_numpy(np.concatenate([np.arange(first[i], first[i + 1]) for i in fr])).to(device)
                ]
            if prof is None:
                tiles = decode_rgb(d, sel, hdr)
            else:
                coefs = decode_mcus(d, sel, hdr)
                mark("entropy decode")
                samples = idct(coefs, hdr)
                mark("IDCT")
                tiles = color(samples, hdr)
                mark("colour")
            k = 0
            for i in fr:
                rows, cols = int(info["rows"][i]), int(info["cols"][i])
                t = tiles[k : k + rows * cols]
                k += rows * cols
                _, mh, mw, ch = t.shape
                img = t.reshape(rows, cols, mh, mw, ch).permute(0, 2, 1, 3, 4).reshape(rows * mh, cols * mw, ch)
                out[i] = img[: int(info["h"][i]), : int(info["w"][i])].permute(2, 0, 1)
        mark("tiles -> images")
    return out


_HEADERS = {}


def table_headers(host, off):
    """The tables of each JPEG host[off[i]:off[i + 1]] (only its header is read): (keys, {key: header}), equal keys
    for equal tables and sampling; headers parsed once per process and kept."""
    from . import codec

    off = np.asarray(off, np.int64)
    sos = codec.probe(host, off)["sos_end"]
    keys = _table_keys(host, off, sos)
    for i, k in enumerate(keys):
        if k not in _HEADERS:
            _HEADERS[k] = header(host[off[i] : sos[i]].tobytes() + b"\xff\xd9")
    return keys, {k: _HEADERS[k] for k in dict.fromkeys(keys)}
