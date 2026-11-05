"""The restart re-coder's fast path in numba: what csrc/jpeg_restart.c's fast_restart_many does, for when that C
cannot be built (no compiler, or no libjpeg headers). Same output bytes and statuses, bit for bit (tests compare the
two); frames it does not handle (status 2: progressive, non-standard Huffman tables, Adobe/RGB, multi-scan,
corrupt) need libjpeg, which codec.to_restart then reaches through jpegtran.

    out, out_off, status = fast_restart_many(buf, off)    # interval 1 only

How it works (see the C for the details): with a restart marker after every MCU, the only symbols that change are
each component's first DC difference in an MCU, so those are re-coded and every other bit is copied.
"""

import numpy as np


try:
    import numba
except ImportError:
    numba = None

# libjpeg's default Huffman tables (ITU T.81 Annex K.3): counts of codes of length 1..16, then the symbols.
# DC luminance, AC luminance, DC chrominance, AC chrominance, as jpeg_set_defaults installs them.
_STD_BITS = (
    (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D),
    (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
    (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77),
)
_AC_LUM = bytes.fromhex(
    "01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435"
    "363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798"
    "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4"
    "f5f6f7f8f9fa"
)
_AC_CHR = bytes.fromhex(
    "000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a"
    "35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a9293949596"
    "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4"
    "f5f6f7f8f9fa"
)
_STD_VALS = (bytes(range(12)), _AC_LUM, bytes(range(12)), _AC_CHR)

FALLBACK = 2


def _tables():
    """Per standard table: its DHT contents, decoding tables (9-bit lookup, AC skip table, the canonical-code
    bounds) and encoding tables, as the C's build()."""
    bits = np.zeros((4, 17), np.int64)
    vals = np.zeros((4, 256), np.int64)
    count = np.zeros(4, np.int64)
    lut = np.zeros((4, 512), np.int64)
    skip = np.zeros((4, 2048), np.int64)
    maxcode = np.zeros((4, 18), np.int64)
    valptr = np.zeros((4, 17), np.int64)
    mincode = np.zeros((4, 17), np.int64)
    code = np.zeros((4, 256), np.int64)
    size = np.zeros((4, 256), np.int64)
    for t in range(4):
        b, v = _STD_BITS[t], _STD_VALS[t]
        assert sum(b) == len(v)
        bits[t, 1:] = b
        count[t] = len(v)
        vals[t, : len(v)] = list(v)
        c = k = 0
        for ln in range(1, 17):
            valptr[t, ln], mincode[t, ln] = k, c
            for _ in range(b[ln - 1]):
                sym = v[k]
                code[t, sym], size[t, sym] = c, ln
                if ln <= 9:
                    lut[t, c << (9 - ln) : (c + 1) << (9 - ln)] = (ln << 8) | sym
                k, c = k + 1, c + 1
            maxcode[t, ln] = c - 1 if b[ln - 1] else -1
            c <<= 1
        maxcode[t, 17] = 0x7FFFFFFF
        c = k = 0
        for ln in range(1, 17):
            for _ in range(b[ln - 1]):
                rs = v[k]
                r, sz = rs >> 4, rs & 15
                if ln <= 11:
                    adv = r + 1 if sz else (16 if r == 15 else 0)
                    eob, zrl = int(not sz and r != 15), int(not sz and r == 15)
                    skip[t, c << (11 - ln) : (c + 1) << (11 - ln)] = (ln + sz) | (adv << 5) | (eob << 10) | (zrl << 11)
                k, c = k + 1, c + 1
            c <<= 1
    return bits, vals, count, lut, skip, maxcode, valptr, mincode, code, size


STD_BITS, STD_VALS, STD_COUNT, LUT, SKIP, MAXCODE, VALPTR, MINCODE, ENC_CODE, ENC_SIZE = _tables()
MARGIN = 1024  # zero bytes after a frame's scan data: a corrupt block may decode past its end before the check

if numba is not None:
    U = np.uint64

    @numba.njit(inline="always")
    def _be64(c, p):
        v = U(0)
        for k in range(8):
            v = (v << U(8)) | U(c[p + k])
        return v

    @numba.njit(inline="always")
    def _peek(c, pos):
        """The 64 bits from bit pos."""
        p, r = pos >> 3, pos & 7
        v = _be64(c, p)
        if r:
            v = (v << U(r)) | U(c[p + 8] >> (8 - r))
        return v

    @numba.njit(inline="always")
    def _fill(c, bp, bb, nb):
        """Top the bit register up to 56..63 bits."""
        bb |= _be64(c, bp) >> U(nb)
        return bp + ((63 - nb) >> 3), bb, nb | 56

    @numba.njit(inline="always")
    def _huff(t, c, bp, bb, nb):
        """One Huffman symbol of standard table t (-1 if none)."""
        if nb < 16:
            bp, bb, nb = _fill(c, bp, bb, nb)
        e = LUT[t, np.int64(bb >> U(55))]
        if e:
            n = e >> 8
            return e & 255, bp, bb << U(n), nb - n
        w = np.int64(bb >> U(48))
        for ln in range(10, 17):
            code = w >> (16 - ln)
            if code <= MAXCODE[t, ln]:
                return _vals_at(t, ln, code), bp, bb << U(ln), nb - ln
        return -1, bp, bb, nb

    @numba.njit(inline="always")
    def _vals_at(t, ln, code):
        return STD_VALS[t, VALPTR[t, ln] + code - MINCODE[t, ln]]

    @numba.njit(inline="always")
    def _put(out, n, cap, acc, wn, bits, k):
        """Append the k (<= 32) low bits of bits, stuffing a 0 after each 0xFF byte."""
        acc = (acc << U(k)) | (U(bits) & ((U(1) << U(k)) - U(1)))
        wn += k
        while wn >= 8:
            b = np.int64((acc >> U(wn - 8)) & U(255))
            wn -= 8
            if n + 2 <= cap:
                out[n] = b
                n += 1
                if b == 0xFF:
                    out[n] = 0
                    n += 1
            else:
                n = cap + 1
        return n, acc, wn

    @numba.njit(inline="always")
    def _raw(out, n, cap, b):
        if n < cap:
            out[n] = b
            return n + 1
        return cap + 1

    @numba.njit(inline="always")
    def _raw16(out, n, cap, v):
        return _raw(out, _raw(out, n, cap, v >> 8), cap, v & 255)

    @numba.njit(inline="always")
    def _dht(out, n, cap, t, cls, idx):
        n = _raw(out, _raw(out, n, cap, 0xFF), cap, 0xC4)
        n = _raw(out, _raw16(out, n, cap, 2 + 1 + 16 + STD_COUNT[t]), cap, (cls << 4) | idx)
        for ln in range(1, 17):
            n = _raw(out, n, cap, STD_BITS[t, ln])
        for i in range(STD_COUNT[t]):
            n = _raw(out, n, cap, STD_VALS[t, i])
        return n

    @numba.njit(inline="always")
    def _be16(s, p):
        return (np.int64(s[p]) << 8) | np.int64(s[p + 1])

    @numba.njit(cache=True, nogil=True)
    def _fast_one(s, out, o0, cap, clean, seg):
        """One frame s re-coded into out[o0:] (out[:cap] usable): (0, end) ok, (2, 0) not handled here, (-3, 0) out
        of space. clean: scratch of len(s) + MARGIN bytes; seg: of len(s) // 2 + 2 entries."""
        n = len(s)
        jfif, jmaj, jmin, junit, jx, jy = 0, 1, 1, 0, 1, 1
        h = w = nc = dri = got_sof = 0
        cid = np.zeros(3, np.int64)
        chv = np.zeros(3, np.int64)
        ctq = np.zeros(3, np.int64)
        ctd = np.zeros(3, np.int64)
        cta = np.zeros(3, np.int64)
        qprec = np.zeros(4, np.int64)
        qhave = np.zeros(4, np.int64)
        q = np.zeros((4, 64), np.int64)
        tbits = np.zeros((2, 4, 17), np.int64)
        tvals = np.zeros((2, 4, 256), np.int64)
        tcount = np.zeros((2, 4), np.int64)
        thave = np.zeros((2, 4), np.int64)
        com_at = np.zeros(32, np.int64)
        com_len = np.zeros(32, np.int64)
        ncom = 0
        i, sos_end = 2, 0
        if n < 4 or s[0] != 0xFF or s[1] != 0xD8:
            return FALLBACK, 0
        while i + 4 <= n:
            if s[i] != 0xFF:
                return FALLBACK, 0
            m = np.int64(s[i + 1])
            if m == 0xFF:
                i += 1
                continue
            if m == 0x01 or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            if not (
                0xE0 <= m <= 0xEF
                or m == 0xFE
                or m == 0xDB
                or m == 0xC4
                or m == 0xDD
                or m == 0xDA
                or (0xC0 <= m <= 0xCF and m != 0xC8)
            ):
                return FALLBACK, 0
            L = _be16(s, i + 2)
            if L < 2 or i + 2 + L > n:
                return FALLBACK, 0
            p, end = i + 4, i + 2 + L
            if (
                m == 0xE0
                and L >= 16
                and s[p] == 0x4A
                and s[p + 1] == 0x46
                and s[p + 2] == 0x49
                and s[p + 3] == 0x46
                and s[p + 4] == 0
            ):
                jfif = 1
                if s[p + 5] != 1 and s[p + 5] != 2:
                    return FALLBACK, 0
                jmaj, jmin, junit = np.int64(s[p + 5]), np.int64(s[p + 6]), np.int64(s[p + 7])
                jx, jy = _be16(s, p + 8), _be16(s, p + 10)
            elif (
                m == 0xEE
                and L >= 14
                and s[p] == 0x41
                and s[p + 1] == 0x64
                and s[p + 2] == 0x6F
                and s[p + 3] == 0x62
                and s[p + 4] == 0x65
            ):
                return FALLBACK, 0
            elif m == 0xFE:
                if ncom == 32:
                    return FALLBACK, 0
                com_at[ncom], com_len[ncom] = p, L - 2
                ncom += 1
            elif m == 0xDB:
                while p < end:
                    pq, t = np.int64(s[p]) >> 4, np.int64(s[p]) & 15
                    if t > 3 or pq > 1 or p + 1 + 64 * (pq + 1) > end:
                        return FALLBACK, 0
                    qprec[t] = 0
                    for k in range(64):
                        q[t, k] = _be16(s, p + 1 + 2 * k) if pq else np.int64(s[p + 1 + k])
                        if q[t, k] > 255:
                            qprec[t] = 1
                    qhave[t] = 1
                    p += 1 + 64 * (pq + 1)
            elif m == 0xC4:
                while p < end:
                    tc, th = np.int64(s[p]) >> 4, np.int64(s[p]) & 15
                    if tc > 1 or th > 3 or p + 17 > end:
                        return FALLBACK, 0
                    cnt = 0
                    for ln in range(1, 17):
                        tbits[tc, th, ln] = s[p + ln]
                        cnt += np.int64(s[p + ln])
                    if cnt > 256 or p + 17 + cnt > end:
                        return FALLBACK, 0
                    for k in range(cnt):
                        tvals[tc, th, k] = s[p + 17 + k]
                    tcount[tc, th] = cnt
                    thave[tc, th] = 1
                    p += 17 + cnt
            elif m == 0xC0 or m == 0xC1:
                if L < 8 or s[p] != 8:
                    return FALLBACK, 0
                h, w, nc = _be16(s, p + 1), _be16(s, p + 3), np.int64(s[p + 5])
                if (nc != 1 and nc != 3) or L != 8 + 3 * nc or h == 0 or w == 0:
                    return FALLBACK, 0
                for c in range(nc):
                    cid[c], chv[c], ctq[c] = s[p + 6 + 3 * c], s[p + 7 + 3 * c], s[p + 8 + 3 * c]
                got_sof = 1
            elif 0xC2 <= m <= 0xCF and m != 0xC4 and m != 0xC8 and m != 0xCC:
                return FALLBACK, 0  # progressive, lossless, hierarchical, arithmetic
            elif m == 0xDD:
                if L != 4:
                    return FALLBACK, 0
                dri = _be16(s, p)
            elif m == 0xDA:
                if not got_sof or s[p] != nc or L != 6 + 2 * nc:
                    return FALLBACK, 0
                for c in range(nc):
                    if s[p + 1 + 2 * c] != cid[c]:
                        return FALLBACK, 0
                    ctd[c], cta[c] = np.int64(s[p + 2 + 2 * c]) >> 4, np.int64(s[p + 2 + 2 * c]) & 15
                if s[p + 1 + 2 * nc] != 0 or s[p + 2 + 2 * nc] != 63 or s[p + 3 + 2 * nc] != 0:
                    return FALLBACK, 0
                sos_end = i + 2 + L
                break
            i += 2 + L
        if sos_end == 0:
            return FALLBACK, 0
        if nc == 3 and not jfif:
            return FALLBACK, 0  # libjpeg may take it for RGB
        # the scan's tables must be the standard ones libjpeg writes: luminance for component 0, chrominance else
        for c in range(nc):
            r = 0 if c == 0 else 2
            if ctd[c] > 3 or cta[c] > 3 or not thave[0, ctd[c]] or not thave[1, cta[c]] or ctq[c] > 3:
                return FALLBACK, 0
            if not qhave[ctq[c]]:
                return FALLBACK, 0
            for cls in range(2):
                tt, std = (ctd[c], r) if cls == 0 else (cta[c], r + 1)
                if tcount[cls, tt] != STD_COUNT[std]:
                    return FALLBACK, 0
                for ln in range(1, 17):
                    if tbits[cls, tt, ln] != STD_BITS[std, ln]:
                        return FALLBACK, 0
                for k in range(STD_COUNT[std]):
                    if tvals[cls, tt, k] != STD_VALS[std, k]:
                        return FALLBACK, 0
        # entropy-coded data, unstuffed; seg[k]: where the k-th restart interval after the first starts
        cn = ns = 0
        j, eoi = sos_end, 0
        while j < n:
            x = s[j]
            if x != 0xFF:
                clean[cn] = x
                cn += 1
                j += 1
                continue
            if j + 1 >= n:
                break
            x = np.int64(s[j + 1])
            if x == 0x00:
                clean[cn] = 0xFF
                cn += 1
                j += 2
            elif 0xD0 <= x <= 0xD7:  # libjpeg resyncs (and warns) on a marker out of order
                if x - 0xD0 != (ns & 7):
                    return FALLBACK, 0
                seg[ns] = cn
                ns += 1
                j += 2
            elif x == 0xFF:
                j += 1
            elif x == 0xD9:
                eoi = 1
                break
            else:
                return FALLBACK, 0  # DNL, another scan, ...
        if not eoi:
            return FALLBACK, 0
        clean[cn : cn + MARGIN] = 0
        # MCU geometry
        hmax = vmax = 1
        nblk = np.zeros(3, np.int64)
        for c in range(nc):
            hc, vc = chv[c] >> 4, chv[c] & 15
            if hc < 1 or hc > 4 or vc < 1 or vc > 4:
                return FALLBACK, 0
            hmax, vmax = max(hmax, hc), max(vmax, vc)
            nblk[c] = hc * vc
        if nc == 1:
            hmax = vmax = 1
            nblk[0] = 1
        mx, my = (w + 8 * hmax - 1) // (8 * hmax), (h + 8 * vmax - 1) // (8 * vmax)
        M = mx * my
        if ns != ((M - 1) // dri if dri else 0):
            return FALLBACK, 0  # a marker where libjpeg ends the data, or one missing
        # header, as libjpeg writes it
        o = o0
        for b in (0xFF, 0xD8, 0xFF, 0xE0, 0, 16, 0x4A, 0x46, 0x49, 0x46, 0):
            o = _raw(out, o, cap, b)
        o = _raw(out, _raw(out, _raw(out, o, cap, jmaj), cap, jmin), cap, junit)
        o = _raw(out, _raw(out, _raw16(out, _raw16(out, o, cap, jx), cap, jy), cap, 0), cap, 0)
        for k in range(ncom):
            o = _raw16(out, _raw(out, _raw(out, o, cap, 0xFF), cap, 0xFE), cap, com_len[k] + 2)
            for b in range(com_len[k]):
                o = _raw(out, o, cap, s[com_at[k] + b])
        sent = np.zeros(4, np.int64)
        prec = 0
        for c in range(nc):
            t = ctq[c]
            if sent[t]:
                continue
            sent[t] = 1
            prec |= qprec[t]
            o = _raw16(out, _raw(out, _raw(out, o, cap, 0xFF), cap, 0xDB), cap, 2 + 1 + 64 * (qprec[t] + 1))
            o = _raw(out, o, cap, (qprec[t] << 4) | t)
            for k in range(64):
                if qprec[t]:
                    o = _raw(out, o, cap, q[t, k] >> 8)
                o = _raw(out, o, cap, q[t, k] & 255)
        o = _raw(out, _raw(out, o, cap, 0xFF), cap, 0xC1 if prec else 0xC0)
        o = _raw(out, _raw16(out, o, cap, 8 + 3 * nc), cap, 8)
        o = _raw(out, _raw16(out, _raw16(out, o, cap, h), cap, w), cap, nc)
        for c in range(nc):
            o = _raw(out, _raw(out, _raw(out, o, cap, cid[c]), cap, chv[c]), cap, ctq[c])
        o = _dht(out, _dht(out, o, cap, 0, 0, 0), cap, 1, 1, 0)
        if nc == 3:
            o = _dht(out, _dht(out, o, cap, 2, 0, 1), cap, 3, 1, 1)
        o = _raw16(out, _raw16(out, _raw(out, _raw(out, o, cap, 0xFF), cap, 0xDD), cap, 4), cap, 1)
        o = _raw(out, _raw16(out, _raw(out, _raw(out, o, cap, 0xFF), cap, 0xDA), cap, 6 + 2 * nc), cap, nc)
        for c in range(nc):
            o = _raw(out, _raw(out, o, cap, cid[c]), cap, 0x00 if c == 0 else 0x11)
        o = _raw(out, _raw(out, _raw(out, o, cap, 0), cap, 63), cap, 0)
        # MCUs
        limit, next_seg = cn * 8 + 64, 0
        pred = np.zeros(3, np.int64)
        bp, bb, nb = 0, U(0), 0
        acc, wn = U(0), 0
        for mcu in range(M):
            if dri and mcu > 0 and mcu % dri == 0:
                if next_seg >= ns:
                    return FALLBACK, 0
                left = seg[next_seg] * 8 - (bp * 8 - nb)
                if left < 0 or left >= 8:
                    return FALLBACK, 0
                bp, bb, nb = seg[next_seg], U(0), 0
                next_seg += 1
                pred[:] = 0
            for c in range(nc):
                dc, ac = (0, 1) if c == 0 else (2, 3)
                copy_from = 0
                for b in range(nblk[c]):
                    t, bp, bb, nb = _huff(dc, clean, bp, bb, nb)
                    if t < 0 or t > 11:
                        return FALLBACK, 0
                    diff = 0
                    if t:
                        if nb < 16:
                            bp, bb, nb = _fill(clean, bp, bb, nb)
                        diff = np.int64(bb >> U(64 - t))
                        bb, nb = bb << U(t), nb - t
                        if diff < (1 << (t - 1)):
                            diff -= (1 << t) - 1
                    pred[c] += diff
                    if b == 0:  # the predictor restarts at 0: code the DC value itself
                        v = pred[c]
                        a, cat = abs(v), 0
                        while a:
                            cat += 1
                            a >>= 1
                        if cat > 11:
                            return FALLBACK, 0
                        o, acc, wn = _put(out, o, cap, acc, wn, ENC_CODE[dc, cat], ENC_SIZE[dc, cat])
                        if cat:
                            o, acc, wn = _put(out, o, cap, acc, wn, (v - 1 if v < 0 else v) & ((1 << cat) - 1), cat)
                        copy_from = bp * 8 - nb
                    # only the symbols' lengths matter: the bits are copied
                    zrl, k = 0, 1
                    while k < 64:
                        if nb < 32:
                            bp, bb, nb = _fill(clean, bp, bb, nb)
                        e = SKIP[ac, np.int64(bb >> U(53))]
                        if e:
                            bb, nb = bb << U(e & 31), nb - (e & 31)
                            if e & (1 << 10):
                                break
                            zrl = e >> 11
                            k += (e >> 5) & 31
                            continue
                        rs, bp, bb, nb = _huff(ac, clean, bp, bb, nb)
                        if rs < 0:
                            return FALLBACK, 0
                        r, sz = rs >> 4, rs & 15
                        if sz == 0:
                            if r != 15:
                                break
                            zrl = 1
                            k += 16
                            continue
                        zrl = 0
                        k += r + 1
                        if nb < 16:
                            bp, bb, nb = _fill(clean, bp, bb, nb)
                        bb, nb = bb << U(sz), nb - sz
                    if zrl or k > 64:
                        return FALLBACK, 0
                    if bp > cn + 8:
                        return FALLBACK, 0
                pos = bp * 8 - nb
                if pos > limit:
                    return FALLBACK, 0
                a = copy_from
                while a < pos:  # the rest of this component's blocks, bit for bit
                    k = min(pos - a, 32)
                    o, acc, wn = _put(out, o, cap, acc, wn, _peek(clean, a) >> U(64 - k), k)
                    a += k
            if wn:
                o, acc, wn = _put(out, o, cap, acc, wn, (1 << (8 - wn)) - 1, 8 - wn)  # pad with 1 bits
            o = _raw(out, _raw(out, o, cap, 0xFF), cap, 0xD9 if mcu == M - 1 else 0xD0 + (mcu & 7))
        left = cn * 8 - (bp * 8 - nb)
        if left < 0 or left >= 8:
            return FALLBACK, 0
        if o > cap:
            return -3, 0
        return 0, o

    @numba.njit(cache=True, nogil=True)
    def _fast_many(buf, off, out, out_off, status):
        big = 0
        for i in range(len(off) - 1):
            big = max(big, off[i + 1] - off[i])
        clean = np.zeros(big + MARGIN + 16, np.uint8)
        seg = np.zeros(big // 2 + 2, np.int64)
        o = 0
        out_off[0] = 0
        for i in range(len(off) - 1):
            r, end = _fast_one(buf[off[i] : off[i + 1]], out, o, len(out), clean, seg)
            if r == -3:
                return -1 - i
            status[i] = r
            if r == 0:
                o = end
            out_off[i + 1] = o
        return o


def fast_restart_many(buf, off):
    """As codec.to_restart's fast path (fast_restart_many in C): (out, out_off, status), status 2 where not
    handled here."""
    n = len(off) - 1
    out_off, status = np.zeros(n + 1, np.int64), np.zeros(n, np.int32)
    cap = int(1.5 * (off[-1] - off[0])) + 4096 * n
    while True:
        out = np.empty(cap, np.uint8)
        r = _fast_many(buf, off, out, out_off, status)
        if r >= 0:
            return out[:r], out_off, status
        cap *= 2
