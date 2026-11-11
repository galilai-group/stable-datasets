"""A bit-exact reference decoder for the subset of H.264 that MP4 stores are written in (stable_datasets.framestore.mp4 with
gpu=True): the reference the GPU decoder (stable_datasets.framestore.gpu_h264) is checked against, and its tests' oracle against
ffmpeg. Pure Python and numpy: slow (~1 s per 384 x 682 frame); for tests and for working out the GPU stages.

The subset, which the builder's encoder settings guarantee (and this decoder checks, raising on anything else):
Baseline-style streams (CAVLC, frame pictures, 4:2:0, 8-bit, no B slices, no 8x8 transform, flat scaling lists),
one reference frame, no deblocking (disable_deblocking_filter_idc 1), one slice group. Slices may start anywhere
(the stores use one slice per macroblock row, so that every row decodes independently: neighbours in another slice
are unavailable, by the standard).

    for yuv in decode_mp4(open("v.mp4", "rb").read()):   # each frame: (Y (H, W), U (H/2, W/2), V (H/2, W/2)) uint8
        ...

Stages, as the GPU decoder splits them: parse (slice headers, macroblock layer and CAVLC residuals, motion vector
prediction: sequential within a slice), then per macroblock in a frame's order: prediction (intra from reconstructed
neighbours, inter from the previous frame) plus the inverse transform of its residual.
"""

import numpy as np


# --- CAVLC tables (ITU-T H.264 9.2; values as ffmpeg's h264_cavlc.c lists them) ---------------------------------

COEFF_TOKEN_LEN = [
    [1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13, 13, 11, 9, 13, 13,
     13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16,
     16, 16, 16, 16, 16, 16],
    [2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9, 6, 11, 11, 11, 7,
     12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14,
     14, 14, 14],
    [4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4, 8, 7, 7, 5, 8, 8,
     7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0] + [6] * 56,
]  # fmt: skip
COEFF_TOKEN_BITS = [
    [1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5, 4, 8, 10, 13, 4,
     15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5,
     8],
    [3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5, 4, 11, 14, 13, 4,
     15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4],
    [15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14, 13, 9, 8, 10, 9, 8,
     15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7,
     6, 1, 4, 3, 2],
    [3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0] + list(range(8, 64)),
]  # fmt: skip
CHROMA_DC_COEFF_TOKEN_LEN = [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7]
CHROMA_DC_COEFF_TOKEN_BITS = [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0]
TOTAL_ZEROS_LEN = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9], [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6], [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5], [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6], [6, 5, 3, 3, 3, 2, 3, 4, 3, 6], [6, 4, 5, 3, 2, 2, 3, 3, 6], [6, 6, 4, 2, 2, 3, 2, 5],
    [5, 5, 3, 2, 2, 2, 4], [4, 4, 3, 3, 1, 3], [4, 4, 2, 1, 3], [3, 3, 1, 2], [2, 2, 1], [1, 1],
]  # fmt: skip
TOTAL_ZEROS_BITS = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1], [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0], [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0], [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0], [1, 1, 5, 4, 3, 3, 2, 1, 1, 0], [1, 1, 1, 3, 3, 2, 2, 1, 0], [1, 0, 1, 3, 2, 1, 1, 1],
    [1, 0, 1, 3, 2, 1, 1], [0, 1, 1, 2, 1, 3], [0, 1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1], [0, 1],
]  # fmt: skip
CHROMA_DC_TOTAL_ZEROS_LEN = [[1, 2, 3, 3], [1, 2, 2], [1, 1]]
CHROMA_DC_TOTAL_ZEROS_BITS = [[1, 1, 1, 0], [1, 1, 0], [1, 0]]
RUN_LEN = [[1, 1], [1, 2, 2], [2, 2, 2, 2], [2, 2, 2, 3, 3], [2, 2, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3, 3],
           [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # fmt: skip
RUN_BITS = [[1, 0], [1, 1, 0], [3, 2, 1, 0], [3, 2, 1, 1, 0], [3, 2, 3, 2, 1, 0], [3, 0, 1, 3, 2, 5, 4],
            [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]]  # fmt: skip

GOLOMB_TO_INTRA4X4_CBP = [47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26,
                          28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41]  # fmt: skip
GOLOMB_TO_INTER_CBP = [0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36,
                       40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41]  # fmt: skip
# zig-zag scan: coefficient index -> raster position x + 4 y in the 4x4 block
ZIGZAG = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
DEQUANT = [[10, 13, 16], [11, 14, 18], [13, 16, 20], [14, 18, 23], [16, 20, 25], [18, 23, 29]]
CHROMA_QP = list(range(30)) + [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39]


def _vlc(lens, bits, values):
    """{(length, code): value} of a VLC table (length 0: no such code)."""
    return {(n, b): v for n, b, v in zip(lens, bits, values) if n}


COEFF_TOKEN = [_vlc(COEFF_TOKEN_LEN[t], COEFF_TOKEN_BITS[t], [(i // 4, i % 4) for i in range(68)]) for t in range(4)]
CHROMA_DC_COEFF_TOKEN = _vlc(
    CHROMA_DC_COEFF_TOKEN_LEN, CHROMA_DC_COEFF_TOKEN_BITS, [(i // 4, i % 4) for i in range(20)]
)
TOTAL_ZEROS = [_vlc(n, b, range(len(n))) for n, b in zip(TOTAL_ZEROS_LEN, TOTAL_ZEROS_BITS)]
CHROMA_DC_TOTAL_ZEROS = [
    _vlc(n, b, range(len(n))) for n, b in zip(CHROMA_DC_TOTAL_ZEROS_LEN, CHROMA_DC_TOTAL_ZEROS_BITS)
]
RUN = [_vlc(n, b, range(len(n))) for n, b in zip(RUN_LEN, RUN_BITS)]


class UnsupportedStream(ValueError):
    """The stream uses a feature outside the subset (see the module docstring)."""


# --- bitstream -------------------------------------------------------------------------------------------------


def unescape(nal):
    """A NAL unit's payload with its emulation prevention bytes (00 00 03 -> 00 00) removed."""
    b = bytes(nal)
    if b.find(b"\x00\x00\x03") < 0:
        return b
    out, i, zeros = bytearray(), 0, 0
    for c in b:
        if zeros >= 2 and c == 3:
            zeros = 0
            continue
        out.append(c)
        zeros = zeros + 1 if c == 0 else 0
        i += 1
    return bytes(out)


class Bits:
    def __init__(self, data, pos=0):
        self.data, self.pos = data, pos
        self.n = len(data) * 8

    def u(self, k):
        v = 0
        for _ in range(k):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def peek(self, k):
        p, v = self.pos, self.u(min(k, self.n - self.pos))
        v <<= max(0, k - (self.n - p))
        self.pos = p
        return v

    def ue(self):
        z = 0
        while not self.u(1):
            z += 1
        return (1 << z) - 1 + self.u(z)

    def se(self):
        k = self.ue()
        return (k + 1) >> 1 if k & 1 else -(k >> 1)

    def vlc(self, table, maxlen=16):
        code = 0
        for n in range(1, maxlen + 1):
            code = (code << 1) | self.u(1)
            if (n, code) in table:
                return table[(n, code)]
        raise ValueError(f"no VLC code at bit {self.pos}")

    def more_rbsp_data(self):
        """Whether data is left before the rbsp trailing bits (the last 1 bit of the payload)."""
        last = self.n - 1
        while last >= 0 and not (self.data[last >> 3] >> (7 - (last & 7))) & 1:
            last -= 1
        return self.pos < last


# --- parameter sets and slice headers --------------------------------------------------------------------------


def parse_sps(rbsp):
    b = Bits(rbsp, 8)  # after the NAL header byte
    s = {"profile_idc": b.u(8)}
    b.u(16)  # constraint flags, level
    s["id"] = b.ue()
    if s["profile_idc"] in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        if b.ue() != 1:
            raise UnsupportedStream("chroma format other than 4:2:0")
        if b.ue() or b.ue():
            raise UnsupportedStream("bit depth other than 8")
        b.u(1)  # qpprime_y_zero_transform_bypass
        if b.u(1):
            raise UnsupportedStream("scaling matrices")
    s["log2_max_frame_num"] = b.ue() + 4
    s["poc_type"] = b.ue()
    if s["poc_type"] == 0:
        s["log2_max_poc_lsb"] = b.ue() + 4
    elif s["poc_type"] == 1:
        b.u(1)
        b.se()
        b.se()
        for _ in range(b.ue()):
            b.se()
    s["max_ref_frames"] = b.ue()
    b.u(1)
    s["mb_w"], s["mb_h"] = b.ue() + 1, b.ue() + 1
    if not b.u(1):
        raise UnsupportedStream("field pictures")
    b.u(1)  # direct_8x8_inference
    crop = [0, 0, 0, 0]
    if b.u(1):
        crop = [b.ue() for _ in range(4)]  # left, right, top, bottom, in units of 2 (4:2:0 frames)
    s["crop"] = crop
    return s


def parse_pps(rbsp):
    b = Bits(rbsp, 8)
    p = {"id": b.ue(), "sps_id": b.ue()}
    if b.u(1):
        raise UnsupportedStream("CABAC")
    p["bottom_field_poc_present"] = b.u(1)
    if b.ue():
        raise UnsupportedStream("slice groups")
    p["num_ref_idx_default"] = b.ue() + 1
    b.ue()
    if b.u(1) or b.u(2):
        raise UnsupportedStream("weighted prediction")
    p["init_qp"] = 26 + b.se()
    b.se()
    p["chroma_qp_offset"] = b.se()
    p["deblocking_control"] = b.u(1)
    p["constrained_intra"] = b.u(1)
    p["redundant_pic_cnt"] = b.u(1)
    if b.more_rbsp_data():
        if b.u(1):
            raise UnsupportedStream("8x8 transform")
        if b.u(1):
            raise UnsupportedStream("scaling matrices")
        b.se()
    return p


def parse_slice_header(rbsp, sps, pps):
    """The slice header's fields and the bit position where its slice data starts."""
    nal_ref_idc, nal_type = (rbsp[0] >> 5) & 3, rbsp[0] & 31
    b = Bits(rbsp, 8)
    h = {"first_mb": b.ue(), "type": b.ue() % 5, "pps_id": b.ue(), "idr": nal_type == 5}
    if h["type"] not in (0, 2):  # P, I
        raise UnsupportedStream("B, SP or SI slices")
    h["frame_num"] = b.u(sps["log2_max_frame_num"])
    if h["idr"]:
        b.ue()
    if sps["poc_type"] == 0:
        b.u(sps["log2_max_poc_lsb"])
        if pps["bottom_field_poc_present"]:
            b.se()
    elif sps["poc_type"] == 1:
        raise UnsupportedStream("picture order count type 1")
    if pps["redundant_pic_cnt"]:
        b.ue()
    h["num_ref_idx"] = pps["num_ref_idx_default"]
    if h["type"] == 0:
        if b.u(1):
            h["num_ref_idx"] = b.ue() + 1
        if b.u(1):
            raise UnsupportedStream("reference list modification")
    if h["num_ref_idx"] != 1 and h["type"] == 0:
        raise UnsupportedStream("more than one reference frame")
    if nal_ref_idc:
        if h["idr"]:
            b.u(2)
        elif b.u(1):
            raise UnsupportedStream("memory management control operations")
    h["qp"] = pps["init_qp"] + b.se()
    if pps["deblocking_control"]:
        if b.ue() != 1:
            raise UnsupportedStream("deblocking")
    else:
        raise UnsupportedStream("deblocking")
    h["data_bit"] = b.pos
    return h


# --- MP4 -------------------------------------------------------------------------------------------------------


def _boxes(data, start, end):
    while start + 8 <= end:
        size = int.from_bytes(data[start : start + 4], "big")
        kind = bytes(data[start + 4 : start + 8])
        head = 8
        if size == 1:
            size, head = int.from_bytes(data[start + 8 : start + 16], "big"), 16
        elif size == 0:
            size = end - start
        yield kind, start + head, start + size
        start += size


def _find(data, start, end, path):
    for kind, a, b in _boxes(data, start, end):
        if kind == path[0]:
            return (a, b) if len(path) == 1 else _find(data, a, b, path[1:])
    return None


def mp4_samples(data):
    """(SPS NAL, PPS NAL, NAL length size, [(offset, size)] per frame) of an MP4's first video track."""
    moov = _find(data, 0, len(data), [b"moov"])
    for kind, a, b in _boxes(data, *moov):
        if kind != b"trak":
            continue
        stbl = _find(data, a, b, [b"mdia", b"minf", b"stbl"])
        stsd = _find(data, *stbl, [b"stsd"])
        entry = stsd[0] + 8  # version/flags, entry count
        if bytes(data[entry + 4 : entry + 8]) != b"avc1":
            continue
        avcc = _find(data, entry + 8 + 78, int.from_bytes(data[entry : entry + 4], "big") + entry, [b"avcC"])
        c = data[avcc[0] : avcc[1]]
        nal_len = (c[4] & 3) + 1
        p = 6
        n = int.from_bytes(c[p : p + 2], "big")
        sps = bytes(c[p + 2 : p + 2 + n])
        p += 2 + n + 1
        n = int.from_bytes(c[p : p + 2], "big")
        pps = bytes(c[p + 2 : p + 2 + n])
        a2, _ = _find(data, *stbl, [b"stsz"])
        fixed, count = int.from_bytes(data[a2 + 4 : a2 + 8], "big"), int.from_bytes(data[a2 + 8 : a2 + 12], "big")
        sizes = (
            [fixed] * count
            if fixed
            else [int.from_bytes(data[a2 + 12 + 4 * i : a2 + 16 + 4 * i], "big") for i in range(count)]
        )
        co = _find(data, *stbl, [b"stco"])
        w = 4
        if co is None:
            co, w = _find(data, *stbl, [b"co64"]), 8
        nchunks = int.from_bytes(data[co[0] + 4 : co[0] + 8], "big")
        chunk_off = [int.from_bytes(data[co[0] + 8 + w * i : co[0] + 8 + w * (i + 1)], "big") for i in range(nchunks)]
        a3, _ = _find(data, *stbl, [b"stsc"])
        runs = [[int.from_bytes(data[a3 + 8 + 12 * i + 4 * k : a3 + 12 + 12 * i + 4 * k], "big") for k in range(3)]
                for i in range(int.from_bytes(data[a3 + 4 : a3 + 8], "big"))]  # fmt: skip
        samples, s = [], 0
        for ci in range(nchunks):
            per = [r[1] for r in runs if r[0] <= ci + 1][-1]
            off = chunk_off[ci]
            for _ in range(per):
                samples.append((off, sizes[s]))
                off += sizes[s]
                s += 1
        return sps, pps, nal_len, samples
    raise ValueError("no H.264 video track")


def sample_nals(data, off, size, nal_len):
    out, p = [], off
    while p < off + size:
        n = int.from_bytes(data[p : p + nal_len], "big")
        out.append(data[p + nal_len : p + nal_len + n])
        p += nal_len + n
    return out


# --- macroblock parsing ----------------------------------------------------------------------------------------

I16_INFO = [None] + [
    ((i - 1) % 4, ((i - 1) // 4) % 3 * 16 + (15 if i >= 13 else 0)) for i in range(1, 25)
]  # mode, cbp


def _blk(x, y):
    """4x4 luma block index (decoding order) of the block at (x, y) in a macroblock."""
    return (x >> 1) * 4 + (y >> 1) * 8 + (x & 1) + (y & 1) * 2


BLK_XY = [None] * 16
for _y in range(4):
    for _x in range(4):
        BLK_XY[_blk(_x, _y)] = (_x, _y)


class Frame:
    """Per-frame macroblock state: what parsing produces and reconstruction consumes."""

    def __init__(self, sps):
        W, H = sps["mb_w"], sps["mb_h"]
        self.W, self.H = W, H
        self.slice = np.full((H, W), -1, np.int64)  # slice number of each macroblock (-1: not decoded)
        self.mbtype = np.zeros((H, W), np.int64)  # 0 P_Skip, 1 inter, 2 I4x4, 3 I16x16, 4 I_PCM
        self.nnz = np.zeros((H * 4, W * 4), np.int64)  # luma non-zero counts per 4x4 block
        self.nnzc = np.zeros((2, H * 2, W * 2), np.int64)  # chroma AC non-zero counts per 4x4 block
        self.i4mode = np.full((H * 4, W * 4), -1, np.int64)  # intra 4x4 modes (-1: not an I4x4 block)
        self.mv = np.zeros((H * 4, W * 4, 2), np.int64)
        self.ref = np.full((H * 4, W * 4), -1, np.int64)  # reference index per 4x4 block (-1: intra)
        self.mbs = {}  # (mby, mbx) -> parsed macroblock: mode, qp, coefficients


class SliceParser:
    def __init__(self, frame, header, rbsp, sps, pps, slice_num):
        self.f, self.h, self.sps, self.pps = frame, header, sps, pps
        self.b = Bits(rbsp, header["data_bit"])
        self.num = slice_num
        self.qp = header["qp"]

    # neighbour availability: inside the frame, in this slice, already decoded
    def avail_mb(self, mby, mbx):
        return 0 <= mby < self.f.H and 0 <= mbx < self.f.W and self.f.slice[mby, mbx] == self.num

    def avail_blk(self, by, bx, cur, done):
        """Whether 4x4 block (by, bx) (frame coordinates) is available to the current macroblock cur = (mby, mbx);
        done: the current macroblock's decoded blocks (4x4 bool)."""
        mby, mbx = by >> 2, bx >> 2
        if (mby, mbx) == cur:
            return bool(done[by & 3, bx & 3])
        if not self.avail_mb(mby, mbx):
            return False
        return (mby, mbx) < cur  # decoded before the current macroblock (raster order)

    def nC(self, plane, by, bx, cur):
        """Predicted non-zero count from the left and top 4x4 blocks (9.2.1)."""
        nnz = self.f.nnz if plane < 0 else self.f.nnzc[plane]
        scale = 4 if plane < 0 else 2
        ok = []
        for yy, xx in ((by, bx - 1), (by - 1, bx)):
            mby, mbx = yy // scale, xx // scale
            if yy < 0 or xx < 0:
                ok.append(None)
            elif (mby, mbx) == cur or self.avail_mb(mby, mbx) and (mby, mbx) < cur:
                ok.append(int(nnz[yy, xx]))
            else:
                ok.append(None)
        a, b = ok
        if a is not None and b is not None:
            return (a + b + 1) >> 1
        return a if a is not None else b if b is not None else 0

    def residual(self, nc, max_coeff, chroma_dc=False):
        """One CAVLC residual block: (coefficients in scan order (max_coeff long), total_coeff)."""
        b = self.b
        if chroma_dc:
            total, t1 = b.vlc(CHROMA_DC_COEFF_TOKEN)
        else:
            total, t1 = b.vlc(COEFF_TOKEN[0 if nc < 2 else 1 if nc < 4 else 2 if nc < 8 else 3])
        coeffs = [0] * max_coeff
        if total == 0:
            return coeffs, 0
        levels = []
        suffix = 1 if total > 10 and t1 < 3 else 0
        for i in range(total):
            if i < t1:
                levels.append(1 - 2 * b.u(1))
                continue
            prefix = 0
            while not b.u(1):
                prefix += 1
            size = 4 if prefix == 14 and suffix == 0 else prefix - 3 if prefix >= 15 else suffix
            code = (min(15, prefix) << suffix) + (b.u(size) if size else 0)
            if prefix >= 15 and suffix == 0:
                code += 15
            if prefix >= 16:
                code += (1 << (prefix - 3)) - 4096
            if i == t1 and t1 < 3:
                code += 2
            level = (code + 2) >> 1 if code % 2 == 0 else (-code - 1) >> 1
            levels.append(level)
            if suffix == 0:
                suffix = 1
            if abs(level) > (3 << (suffix - 1)) and suffix < 6:
                suffix += 1
        if total < max_coeff:
            zeros = b.vlc(CHROMA_DC_TOTAL_ZEROS[total - 1] if chroma_dc else TOTAL_ZEROS[total - 1])
        else:
            zeros = 0
        runs = []
        for i in range(total - 1):
            r = b.vlc(RUN[min(zeros, 7) - 1]) if zeros > 0 else 0
            runs.append(r)
            zeros -= r
        runs.append(zeros)
        pos = -1
        for i in range(total - 1, -1, -1):
            pos += runs[i] + 1
            coeffs[pos] = levels[i]
        return coeffs, total

    # motion vector prediction (8.4.1.3), on 4x4 block coordinates of the frame
    def _neighbour(self, by, bx, cur, done):
        if not self.avail_blk(by, bx, cur, done):
            return None
        return int(self.f.ref[by, bx]), self.f.mv[by, bx].copy()

    def pred_mv(self, by, bx, w, h, cur, done, shape=None, part=0):
        """Predicted motion vector of the partition at block (by, bx), w x h blocks, reference 0."""
        A = self._neighbour(by, bx - 1, cur, done)
        B = self._neighbour(by - 1, bx, cur, done)
        C = self._neighbour(by - 1, bx + w, cur, done)
        if C is None:
            C = self._neighbour(by - 1, bx - 1, cur, done)
        if shape == "16x8":
            if part == 0 and B is not None and B[0] == 0:
                return B[1]
            if part == 1 and A is not None and A[0] == 0:
                return A[1]
        elif shape == "8x16":
            if part == 0 and A is not None and A[0] == 0:
                return A[1]
            if part == 1 and C is not None and C[0] == 0:
                return C[1]
        if B is None and C is None and A is not None:
            B = C = A
        n = [x if x is not None else (-1, np.zeros(2, np.int64)) for x in (A, B, C)]
        match = [x for x in n if x[0] == 0]
        if len(match) == 1:
            return match[0][1].copy()
        return np.median(np.stack([x[1] for x in n]), axis=0).astype(np.int64)

    def set_part(self, by, bx, w, h, mv, done):
        self.f.mv[by : by + h, bx : bx + w] = mv
        self.f.ref[by : by + h, bx : bx + w] = 0
        done[by & 3 : (by & 3) + h, bx & 3 : (bx & 3) + w] = True

    def macroblock(self, mby, mbx, skip):
        f, b, cur = self.f, self.b, (mby, mbx)
        f.slice[mby, mbx] = self.num
        by, bx = mby * 4, mbx * 4
        done = np.zeros((4, 4), bool)
        mb = {"qp": self.qp}
        f.nnz[by : by + 4, bx : bx + 4] = 0
        f.nnzc[:, mby * 2 : mby * 2 + 2, mbx * 2 : mbx * 2 + 2] = 0
        f.i4mode[by : by + 4, bx : bx + 4] = -1
        if skip:  # P_Skip (8.4.1.1)
            f.mbtype[mby, mbx] = 0
            A = self._neighbour(by, bx - 1, cur, done)
            B = self._neighbour(by - 1, bx, cur, done)
            if A is None or B is None or (A[0] == 0 and not A[1].any()) or (B[0] == 0 and not B[1].any()):
                mv = np.zeros(2, np.int64)
            else:
                mv = self.pred_mv(by, bx, 4, 4, cur, done)
            self.set_part(by, bx, 4, 4, mv, done)
            mb.update(kind="inter", cbp=0)
            f.mbs[cur] = mb
            return
        t = b.ue()
        if self.h["type"] == 0:
            if t < 5:
                return self._inter(mb, t, cur, by, bx, done)
            t -= 5
        f.ref[by : by + 4, bx : bx + 4] = -1
        f.mv[by : by + 4, bx : bx + 4] = 0
        if t == 25:
            raise UnsupportedStream("I_PCM")
        if t == 0:
            f.mbtype[mby, mbx] = 2
            modes = [0] * 16
            for n in range(16):
                x, y = BLK_XY[n]
                left = f.i4mode[by + y, bx + x - 1] if self.avail_blk(by + y, bx + x - 1, cur, done | True) else None
                top = f.i4mode[by + y - 1, bx + x] if self.avail_blk(by + y - 1, bx + x, cur, done | True) else None
                if left is None or top is None:
                    pred = 2
                else:
                    pred = min(left if left >= 0 else 2, top if top >= 0 else 2)
                mode = pred if b.u(1) else (lambda r: r + (r >= pred))(b.u(3))
                modes[n] = mode
                f.i4mode[by + y, bx + x] = mode
            mb.update(kind="i4", modes=modes)
        else:
            f.mbtype[mby, mbx] = 3
            mode, cbp = I16_INFO[t]
            mb.update(kind="i16", mode=mode, cbp=cbp)
        mb["chroma_mode"] = b.ue()
        if mb["kind"] == "i4":
            mb["cbp"] = GOLOMB_TO_INTRA4X4_CBP[b.ue()]
        self._residuals(mb, cur, by, bx)
        f.mbs[cur] = mb

    def _inter(self, mb, t, cur, by, bx, done):
        f, b = self.f, self.b
        f.mbtype[cur] = 1
        if t == 0:
            mv = self.pred_mv(by, bx, 4, 4, cur, done) + [b.se(), b.se()]
            self.set_part(by, bx, 4, 4, mv, done)
        elif t == 1:  # 16x8
            for p in range(2):
                mv = self.pred_mv(by + 2 * p, bx, 4, 2, cur, done, "16x8", p) + [b.se(), b.se()]
                self.set_part(by + 2 * p, bx, 4, 2, mv, done)
        elif t == 2:  # 8x16
            for p in range(2):
                mv = self.pred_mv(by, bx + 2 * p, 2, 4, cur, done, "8x16", p) + [b.se(), b.se()]
                self.set_part(by, bx + 2 * p, 2, 4, mv, done)
        else:  # 8x8 (t 3, or 4: all references 0, which is the only one anyway)
            sub = [b.ue() for _ in range(4)]
            if max(sub) > 3:
                raise UnsupportedStream(f"P sub-macroblock type {max(sub)}")
            for i in range(4):
                y0, x0 = by + 2 * (i // 2), bx + 2 * (i % 2)
                parts = {0: [(0, 0, 2, 2)], 1: [(0, 0, 2, 1), (1, 0, 2, 1)], 2: [(0, 0, 1, 2), (0, 1, 1, 2)],
                         3: [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)]}[sub[i]]  # fmt: skip
                for dy, dx, w, h in parts:
                    mv = self.pred_mv(y0 + dy, x0 + dx, w, h, cur, done) + [b.se(), b.se()]
                    self.set_part(y0 + dy, x0 + dx, w, h, mv, done)
        mb.update(kind="inter", cbp=GOLOMB_TO_INTER_CBP[b.ue()])
        self._residuals(mb, cur, by, bx)
        f.mbs[cur] = mb

    def _residuals(self, mb, cur, by, bx):
        f, b = self.f, self.b
        cbp = mb["cbp"]
        mb["luma"] = np.zeros((16, 16), np.int64)  # per 4x4 block (decoding order): coefficients in raster order
        mb["dc"] = None
        mb["chroma_dc"] = np.zeros((2, 4), np.int64)
        mb["chroma_ac"] = np.zeros((2, 4, 16), np.int64)
        if cbp == 0 and mb["kind"] != "i16":
            mb["qp"] = self.qp
            return
        self.qp = (self.qp + b.se() + 52) % 52
        mb["qp"] = self.qp
        if mb["kind"] == "i16":
            c, _ = self.residual(self.nC(-1, by, bx, cur), 16)
            mb["dc"] = np.zeros(16, np.int64)
            for k in range(16):
                mb["dc"][ZIGZAG[k]] = c[k]
        for n in range(16):
            x, y = BLK_XY[n]
            if not (cbp >> (n // 4)) & 1:
                continue
            if mb["kind"] == "i16":
                c, total = self.residual(self.nC(-1, by + y, bx + x, cur), 15)
                c = [0] + c
            else:
                c, total = self.residual(self.nC(-1, by + y, bx + x, cur), 16)
            f.nnz[by + y, bx + x] = total
            for k in range(16):
                mb["luma"][n, ZIGZAG[k]] = c[k]
        if cbp >> 4:
            for p in range(2):
                c, _ = self.residual(-1, 4, chroma_dc=True)
                mb["chroma_dc"][p] = c
        if cbp >> 5:
            cy, cx = cur[0] * 2, cur[1] * 2
            for p in range(2):
                for n in range(4):
                    x, y = n & 1, n >> 1
                    c, total = self.residual(self.nC(p, cy + y, cx + x, cur), 15)
                    f.nnzc[p, cy + y, cx + x] = total
                    c = [0] + c
                    for k in range(16):
                        mb["chroma_ac"][p, n, ZIGZAG[k]] = c[k]

    def run(self, last_mb):
        """Parse macroblocks first_mb..last_mb (the slice's, in raster order)."""
        W = self.f.W
        mb = self.h["first_mb"]
        while mb <= last_mb:
            if self.h["type"] == 0:
                skip = self.b.ue()
                for _ in range(skip):
                    if mb > last_mb:
                        raise ValueError("skip run past the slice")
                    self.macroblock(mb // W, mb % W, True)
                    mb += 1
                if mb > last_mb:
                    break
            self.macroblock(mb // W, mb % W, False)
            mb += 1


# --- reconstruction --------------------------------------------------------------------------------------------


def idct4(d):
    """Inverse 4x4 transform of scaled coefficients d (4x4, [row][col]), with the final (x + 32) >> 6."""
    d = np.asarray(d, np.int64)
    e0, e1 = d[:, 0] + d[:, 2], d[:, 0] - d[:, 2]
    e2, e3 = (d[:, 1] >> 1) - d[:, 3], d[:, 1] + (d[:, 3] >> 1)
    f = np.stack([e0 + e3, e1 + e2, e1 - e2, e0 - e3], 1)
    g0, g1 = f[0] + f[2], f[0] - f[2]
    g2, g3 = (f[1] >> 1) - f[3], f[1] + (f[3] >> 1)
    h = np.stack([g0 + g3, g1 + g2, g1 - g2, g0 - g3], 0)
    return (h + 32) >> 6


# dequantization factor class of each raster position: 0 both coordinates even, 2 both odd, 1 otherwise
_DQ_CLASS = np.array(
    [[0 if i % 2 == 0 and j % 2 == 0 else 2 if i % 2 and j % 2 else 1 for j in range(4)] for i in range(4)]
)


def scale4(c, qp):
    """Dequantize a 4x4 block of levels ([row][col]), flat scaling: c * v(qp % 6, position) << qp / 6."""
    return (np.asarray(c, np.int64) * np.array(DEQUANT[qp % 6])[_DQ_CLASS]) << (qp // 6)


HADAMARD4 = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1]], np.int64)


def luma_dc(c, qp):
    """Intra 16x16 DC: 4x4 levels -> scaled DC per 4x4 block ([row][col] of blocks)."""
    f = HADAMARD4 @ np.asarray(c, np.int64).reshape(4, 4) @ HADAMARD4
    v = DEQUANT[qp % 6][0] * 16
    if qp >= 36:
        return (f * v) << (qp // 6 - 6)
    return (f * v + (1 << (5 - qp // 6))) >> (6 - qp // 6)


def chroma_dc(c, qp):
    c = np.asarray(c, np.int64).reshape(2, 2)
    h = np.array([[1, 1], [1, -1]], np.int64)
    f = h @ c @ h
    return ((f * DEQUANT[qp % 6][0] * 16) << (qp // 6)) >> 5


def clip(x):
    return np.clip(x, 0, 255)


def intra4(mode, top, left, tl, have_top, have_left, have_tl):
    """4x4 intra prediction (8.3.1.2). top: 8 samples above (4..7 above-right, already substituted), left: 4, tl."""
    p = np.zeros((4, 4), np.int64)
    T = lambda x: int(top[x])  # noqa: E731
    L = lambda y: int(left[y]) if y >= 0 else int(tl)  # noqa: E731
    Tx = lambda x: int(top[x]) if x >= 0 else int(tl)  # noqa: E731
    for y in range(4):
        for x in range(4):
            if mode == 0:
                v = T(x)
            elif mode == 1:
                v = L(y)
            elif mode == 2:
                if have_top and have_left:
                    v = (sum(int(t) for t in top[:4]) + sum(int(t) for t in left) + 4) >> 3
                elif have_left:
                    v = (sum(int(t) for t in left) + 2) >> 2
                elif have_top:
                    v = (sum(int(t) for t in top[:4]) + 2) >> 2
                else:
                    v = 128
            elif mode == 3:
                v = (
                    (T(6) + 3 * T(7) + 2) >> 2
                    if x == 3 and y == 3
                    else (T(x + y) + 2 * T(x + y + 1) + T(x + y + 2) + 2) >> 2
                )
            elif mode == 4:
                if x > y:
                    v = (Tx(x - y - 2) + 2 * Tx(x - y - 1) + T(x - y) + 2) >> 2
                elif x < y:
                    v = (L(y - x - 2) + 2 * L(y - x - 1) + L(y - x) + 2) >> 2
                else:
                    v = (T(0) + 2 * int(tl) + L(0) + 2) >> 2
            elif mode == 5:
                z = 2 * x - y
                if z >= 0 and z % 2 == 0:
                    v = (Tx(x - (y >> 1) - 1) + T(x - (y >> 1)) + 1) >> 1
                elif z >= 0:
                    v = (Tx(x - (y >> 1) - 2) + 2 * Tx(x - (y >> 1) - 1) + T(x - (y >> 1)) + 2) >> 2
                elif z == -1:
                    v = (L(0) + 2 * int(tl) + T(0) + 2) >> 2
                else:
                    v = (L(y - 1) + 2 * L(y - 2) + L(y - 3) + 2) >> 2
            elif mode == 6:
                z = 2 * y - x
                if z >= 0 and z % 2 == 0:
                    v = (L(y - (x >> 1) - 1) + L(y - (x >> 1)) + 1) >> 1
                elif z >= 0:
                    v = (L(y - (x >> 1) - 2) + 2 * L(y - (x >> 1) - 1) + L(y - (x >> 1)) + 2) >> 2
                elif z == -1:
                    v = (L(0) + 2 * int(tl) + T(0) + 2) >> 2
                else:
                    v = (Tx(x - 1) + 2 * Tx(x - 2) + Tx(x - 3) + 2) >> 2
            elif mode == 7:
                k = x + (y >> 1)
                v = (T(k) + T(k + 1) + 1) >> 1 if y % 2 == 0 else (T(k) + 2 * T(k + 1) + T(k + 2) + 2) >> 2
            else:
                z = x + 2 * y
                if z > 5:
                    v = L(3)
                elif z == 5:
                    v = (L(2) + 3 * L(3) + 2) >> 2
                elif z % 2 == 0:
                    v = (L(y + (x >> 1)) + L(y + (x >> 1) + 1) + 1) >> 1
                else:
                    v = (L(y + (x >> 1)) + 2 * L(y + (x >> 1) + 1) + L(y + (x >> 1) + 2) + 2) >> 2
            p[y, x] = v
    return p


def intra16(mode, top, left, tl, have_top, have_left):
    if mode == 0:
        return np.tile(top[:16], (16, 1))
    if mode == 1:
        return np.tile(left[:16, None], (1, 16))
    if mode == 2:
        if have_top and have_left:
            v = (top[:16].sum() + left[:16].sum() + 16) >> 5
        elif have_left:
            v = (left[:16].sum() + 8) >> 4
        elif have_top:
            v = (top[:16].sum() + 8) >> 4
        else:
            v = 128
        return np.full((16, 16), v, np.int64)
    t = np.concatenate(([tl], top[:16])).astype(np.int64)  # t[x + 1] = p[x, -1]
    l_ = np.concatenate(([tl], left[:16])).astype(np.int64)
    H = sum((x + 1) * (t[8 + x + 1] - t[6 - x + 1]) for x in range(8))
    V = sum((y + 1) * (l_[8 + y + 1] - l_[6 - y + 1]) for y in range(8))
    a, bb, c = 16 * (l_[16] + t[16]), (5 * H + 32) >> 6, (5 * V + 32) >> 6
    yy, xx = np.mgrid[:16, :16]
    return clip((a + bb * (xx - 7) + c * (yy - 7) + 16) >> 5)


def intra_chroma(mode, top, left, tl, have_top, have_left):
    if mode == 0:
        p = np.zeros((8, 8), np.int64)
        for yo in (0, 4):
            for xo in (0, 4):
                st, sl = int(top[xo : xo + 4].sum()), int(left[yo : yo + 4].sum())
                if (xo, yo) in ((0, 0), (4, 4)):
                    if have_top and have_left:
                        v = (st + sl + 4) >> 3
                    elif have_top:
                        v = (st + 2) >> 2
                    elif have_left:
                        v = (sl + 2) >> 2
                    else:
                        v = 128
                elif xo == 4:
                    v = (st + 2) >> 2 if have_top else (sl + 2) >> 2 if have_left else 128
                else:
                    v = (sl + 2) >> 2 if have_left else (st + 2) >> 2 if have_top else 128
                p[yo : yo + 4, xo : xo + 4] = v
        return p
    if mode == 1:
        return np.tile(left[:8, None], (1, 8))
    if mode == 2:
        return np.tile(top[:8], (8, 1))
    t = np.concatenate(([tl], top[:8])).astype(np.int64)
    l_ = np.concatenate(([tl], left[:8])).astype(np.int64)
    H = sum((x + 1) * (t[4 + x + 1] - t[2 - x + 1]) for x in range(4))
    V = sum((y + 1) * (l_[4 + y + 1] - l_[2 - y + 1]) for y in range(4))
    a, bb, c = 16 * (l_[8] + t[8]), (34 * H + 32) >> 6, (34 * V + 32) >> 6
    yy, xx = np.mgrid[:8, :8]
    return clip((a + bb * (xx - 3) + c * (yy - 3) + 16) >> 5)


def luma_mc(ref, y0, x0, h, w, mv):
    """Inter prediction of an h x w luma block at (y0, x0) from ref (8.4.2.2.1), mv in quarter samples."""
    H, W = ref.shape
    yi, xi = y0 + (mv[1] >> 2), x0 + (mv[0] >> 2)
    yf, xf = mv[1] & 3, mv[0] & 3
    ys = np.clip(np.arange(yi - 2, yi + h + 3), 0, H - 1)
    xs = np.clip(np.arange(xi - 2, xi + w + 3), 0, W - 1)
    P = ref[np.ix_(ys, xs)].astype(np.int64)  # P[2 + y, 2 + x] = sample (yi + y, xi + x)

    def tap(a, axis):  # six-tap filter along axis, output aligned to positions 2..n-4 (between k and k + 1)
        s = lambda k: np.take(a, np.arange(k, a.shape[axis] - 5 + k), axis=axis)  # noqa: E731
        return s(0) - 5 * s(1) + 20 * s(2) + 20 * s(3) - 5 * s(4) + s(5)

    G = P[2 : 2 + h, 2 : 2 + w]
    b1 = tap(P, 1)  # horizontal half samples: b1[y, x] between x + 2 and x + 3 of P's columns
    h1 = tap(P, 0)  # vertical half samples
    b = clip((b1[2 : 2 + h, 0:w] + 16) >> 5)  # b at (y, x + 1/2)
    hh = clip((h1[0:h, 2 : 2 + w] + 16) >> 5)  # h at (y + 1/2, x)
    j1 = tap(b1, 0)  # centre: vertical filter over the unrounded horizontal half samples
    j = clip((j1[0:h, 0:w] + 512) >> 10)  # j at (y + 1/2, x + 1/2)
    m = clip((h1[0:h, 3 : 3 + w] + 16) >> 5)  # h at x + 1
    s = clip((b1[3 : 3 + h, 0:w] + 16) >> 5)  # b at y + 1
    H1 = P[2 : 2 + h, 3 : 3 + w]  # integer sample at x + 1
    M = P[3 : 3 + h, 2 : 2 + w]  # integer sample at y + 1
    avg = lambda a, c: (a + c + 1) >> 1  # noqa: E731
    table = {
        (0, 0): G, (0, 1): avg(G, hh), (0, 2): hh, (0, 3): avg(M, hh),
        (1, 0): avg(G, b), (1, 1): avg(b, hh), (1, 2): avg(hh, j), (1, 3): avg(hh, s),
        (2, 0): b, (2, 1): avg(b, j), (2, 2): j, (2, 3): avg(j, s),
        (3, 0): avg(H1, b), (3, 1): avg(b, m), (3, 2): avg(j, m), (3, 3): avg(m, s),
    }  # fmt: skip
    return table[(xf, yf)]


def chroma_mc(ref, y0, x0, h, w, mv):
    H, W = ref.shape
    yi, xi = y0 + (mv[1] >> 3), x0 + (mv[0] >> 3)
    yf, xf = mv[1] & 7, mv[0] & 7
    ys = np.clip(np.arange(yi, yi + h + 1), 0, H - 1)
    xs = np.clip(np.arange(xi, xi + w + 1), 0, W - 1)
    P = ref[np.ix_(ys, xs)].astype(np.int64)
    A, B, C, D = P[:h, :w], P[:h, 1:], P[1:, :w], P[1:, 1:]
    return ((8 - xf) * (8 - yf) * A + xf * (8 - yf) * B + (8 - xf) * yf * C + xf * yf * D + 32) >> 6


def reconstruct(frame, ref, chroma_qp_offset):
    """The frame's Y, U, V (coded size) from its parsed macroblocks and the previous frame (Y, U, V) or None."""
    W, H = frame.W, frame.H
    Y = np.zeros((H * 16, W * 16), np.int64)
    C = np.zeros((2, H * 8, W * 8), np.int64)
    for mby in range(H):
        for mbx in range(W):
            mb = frame.mbs[(mby, mbx)]
            cur = (mby, mbx)
            sl = frame.slice[mby, mbx]
            av = lambda yy, xx: 0 <= yy < H and 0 <= xx < W and frame.slice[yy, xx] == sl and (yy, xx) < cur  # noqa: E731
            have_top, have_left, have_tl, have_tr = (
                av(mby - 1, mbx),
                av(mby, mbx - 1),
                av(mby - 1, mbx - 1),
                av(mby - 1, mbx + 1),
            )
            y0, x0 = mby * 16, mbx * 16
            qp = mb["qp"]
            qpc = CHROMA_QP[min(max(qp + chroma_qp_offset, 0), 51)]
            # luma
            if mb["kind"] == "inter":
                for by in range(4):
                    for bx in range(4):
                        mv = frame.mv[mby * 4 + by, mbx * 4 + bx]
                        Y[y0 + 4 * by : y0 + 4 * by + 4, x0 + 4 * bx : x0 + 4 * bx + 4] = luma_mc(
                            ref[0], y0 + 4 * by, x0 + 4 * bx, 4, 4, mv
                        )
                if mb["cbp"] & 15:
                    for n in range(16):
                        x, y = BLK_XY[n]
                        r = idct4(scale4(mb["luma"][n].reshape(4, 4), qp))
                        sy, sx = y0 + 4 * y, x0 + 4 * x
                        Y[sy : sy + 4, sx : sx + 4] = clip(Y[sy : sy + 4, sx : sx + 4] + r)
            elif mb["kind"] == "i16":
                top = Y[y0 - 1, x0 : x0 + 16] if have_top else np.zeros(16, np.int64)
                left = Y[y0 : y0 + 16, x0 - 1] if have_left else np.zeros(16, np.int64)
                tl = Y[y0 - 1, x0 - 1] if have_tl else 0
                pred = intra16(mb["mode"], top, left, tl, have_top, have_left)
                dc = luma_dc(mb["dc"], qp)
                for n in range(16):
                    x, y = BLK_XY[n]
                    c = scale4(mb["luma"][n].reshape(4, 4), qp)
                    c[0, 0] = dc[y, x]
                    r = idct4(c)
                    Y[y0 + 4 * y : y0 + 4 * y + 4, x0 + 4 * x : x0 + 4 * x + 4] = clip(
                        pred[4 * y : 4 * y + 4, 4 * x : 4 * x + 4] + r
                    )
            else:  # i4, block by block in decoding order
                for n in range(16):
                    x, y = BLK_XY[n]
                    sy, sx = y0 + 4 * y, x0 + 4 * x
                    ht = y > 0 or have_top
                    hl = x > 0 or have_left
                    htl = (
                        (x > 0 and y > 0)
                        or (y == 0 and x > 0 and have_top)
                        or (x == 0 and y > 0 and have_left)
                        or (x == 0 and y == 0 and have_tl)
                    )
                    if y == 0:
                        htr = have_top if x < 3 else have_tr
                    elif x == 3:
                        htr = False
                    else:
                        htr = _blk(x + 1, y - 1) < n
                    top = np.zeros(8, np.int64)
                    if ht:
                        top[:4] = Y[sy - 1, sx : sx + 4]
                        top[4:] = Y[sy - 1, sx + 4 : sx + 8] if htr else top[3]
                    left = Y[sy : sy + 4, sx - 1] if hl else np.zeros(4, np.int64)
                    tl = Y[sy - 1, sx - 1] if htl else 0
                    pred = intra4(mb["modes"][n], top, left, tl, ht, hl, htl)
                    r = idct4(scale4(mb["luma"][n].reshape(4, 4), qp)) if (mb["cbp"] >> (n // 4)) & 1 else 0
                    Y[sy : sy + 4, sx : sx + 4] = clip(pred + r)
            # chroma
            cy0, cx0 = mby * 8, mbx * 8
            for p in range(2):
                if mb["kind"] == "inter":
                    pred = np.zeros((8, 8), np.int64)
                    for by in range(4):
                        for bx in range(4):
                            mv = frame.mv[mby * 4 + by, mbx * 4 + bx]
                            pred[2 * by : 2 * by + 2, 2 * bx : 2 * bx + 2] = chroma_mc(
                                ref[1 + p], cy0 + 2 * by, cx0 + 2 * bx, 2, 2, mv
                            )
                else:
                    top = C[p, cy0 - 1, cx0 : cx0 + 8] if have_top else np.zeros(8, np.int64)
                    left = C[p, cy0 : cy0 + 8, cx0 - 1] if have_left else np.zeros(8, np.int64)
                    tl = C[p, cy0 - 1, cx0 - 1] if have_tl else 0
                    pred = intra_chroma(mb["chroma_mode"], top, left, tl, have_top, have_left)
                if mb["cbp"] >> 4:
                    dc = chroma_dc(mb["chroma_dc"][p], qpc)
                    for n in range(4):
                        x, y = n & 1, n >> 1
                        c = scale4(mb["chroma_ac"][p, n].reshape(4, 4), qpc)
                        c[0, 0] = dc[y, x]
                        pred[4 * y : 4 * y + 4, 4 * x : 4 * x + 4] = clip(
                            pred[4 * y : 4 * y + 4, 4 * x : 4 * x + 4] + idct4(c)
                        )
                C[p, cy0 : cy0 + 8, cx0 : cx0 + 8] = pred
    return Y, C[0], C[1]


def decode_frame(nals, sps, pps, ref):
    """One frame from its slice NAL units (unescaped), the previous frame (Y, U, V) or None: (Y, U, V) coded size."""
    frame = Frame(sps)
    headers = [parse_slice_header(n, sps, pps) for n in nals]
    order = sorted(range(len(nals)), key=lambda i: headers[i]["first_mb"])
    total = sps["mb_w"] * sps["mb_h"]
    for k, i in enumerate(order):
        last = (headers[order[k + 1]]["first_mb"] if k + 1 < len(order) else total) - 1
        if headers[i]["type"] == 0 and ref is None:
            raise ValueError("P slice without a reference frame")
        SliceParser(frame, headers[i], nals[i], sps, pps, k).run(last)
    return reconstruct(frame, ref, pps["chroma_qp_offset"])


def decode_mp4(data, frames=None):
    """Yield each frame of an MP4 (bytes) as (Y, U, V) uint8, cropped to the display size."""
    sps_nal, pps_nal, nal_len, samples = mp4_samples(data)
    sps, pps = parse_sps(unescape(sps_nal)), parse_pps(unescape(pps_nal))
    cl, cr, ct, cb = sps["crop"]
    ref = None
    for k, (off, size) in enumerate(samples):
        if frames is not None and k >= frames:
            break
        nals = [unescape(n) for n in sample_nals(data, off, size, nal_len) if n[0] & 31 in (1, 5)]
        if nals[0][0] & 31 == 5:
            ref = None
        Y, U, V = decode_frame(nals, sps, pps, ref)
        ref = (Y, U, V)
        h, w = Y.shape
        yield (
            Y[2 * ct : h - 2 * cb, 2 * cl : w - 2 * cr].astype(np.uint8),
            U[ct : h // 2 - cb, cl : w // 2 - cr].astype(np.uint8),
            V[ct : h // 2 - cb, cl : w // 2 - cr].astype(np.uint8),
        )
