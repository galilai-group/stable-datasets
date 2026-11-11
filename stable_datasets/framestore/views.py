"""Training views of decoded clips, fused: every view's crop resampled, flipped, colour-jittered and normalized in one
pass over the uint8 frames, on the GPU, for whole views (dense) or only the patches a model keeps.

    p = draw_views(B, H, W, views=4, size=96, scale=(0.02, 0.4), jitter=(0.4, 0.4, 0.2))   # per (view, clip)
    x = view_frames(clip, p)                        # (4 * B, T, 3, 96, 96) float32, normalized: any model
    x = view_patches(clip, p, idx)                  # (4 * B, S, 768): only the patches idx keeps (token dropping)

clip: the loaders' (B, T, H, W, 3) uint8 CUDA batch. The ops are torchvision's / LeVJEPA v2's ViewBuilder (whose
recipe draw_views defaults to): roi_align(aligned=True) of a random box (area share `scale`, log-uniform aspect
`ratio`) with `sampling_ratio` samples per output pixel and axis, a horizontal flip, brightness, contrast, saturation
and grayscale, then (x - mean) / std. Contrast pivots on each frame's mean over the whole crop (pivot="frame", v2's;
exact for view_frames), or, for view_patches, which never builds the whole crop, on the kept patches' mean of each
frame ("frame") or of the whole view ("view", the token loader's C2).

Each 16 x 16 output patch is one Triton program per channel: its source window (KK x KK pixels around the patch,
loaded once) resampled as two small matrix products, Ry @ window @ Rx^T, with roi_align's own sampling and border
rules (stable_pretraining.data.levjepa.kernels.fused does the same from partly decoded JPEGs). A view with jitter needs a second,
elementwise pass for the contrast pivot. All parameters stay on the GPU: the whole thing is CUDA-graph capturable.
"""

import math
from dataclasses import dataclass

import torch


try:
    import triton
    import triton.language as tl
except ImportError:  # the parameter draw still imports
    triton = None

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PATCH = 16


@dataclass
class Views:
    """One view type's parameters for B clips, row v * B + b: box (rows, 4) x1, y1, x2, y2 in frame pixels; flip
    (rows,) bool; jitter (rows, 4) brightness, contrast, saturation factors (1: none) and grayscale (0 / 1), or None;
    size: the view's side; sampling_ratio: roi_align's samples per output pixel and axis; max_side: the longest box
    side the draw allows (sizes the kernel's source window without reading the boxes back)."""

    box: torch.Tensor
    flip: torch.Tensor
    jitter: torch.Tensor | None
    size: int
    sampling_ratio: int
    B: int
    max_side: float


def draw_views(
    B,
    H,
    W,
    views=1,
    size=224,
    scale=(0.8, 1.0),
    ratio=(3 / 4, 4 / 3),
    flip_p=0.5,
    jitter=None,
    jitter_p=0.8,
    gray_p=0.2,
    device="cuda",
    generator=None,
):
    """Random views of B clips of H x W frames, as ViewBuilder.crop / jitter draw them (one draw per view and clip,
    shared by its frames): jitter (brightness, contrast, saturation) strengths, or None for none."""
    n = views * B

    def u(lo=0.0, hi=1.0):
        return torch.empty(n, device=device).uniform_(lo, hi, generator=generator)

    area = H * W * u(*scale)
    r = u(math.log(ratio[0]), math.log(ratio[1])).exp()
    w = (area * r).sqrt().clamp(max=W)
    h = (area / r).sqrt().clamp(max=H)
    x1 = u() * (W - w)
    y1 = u() * (H - h)
    box = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)
    flip = u() < flip_p
    jit = None
    if jitter is not None:
        apply = u() < jitter_p
        jit = torch.stack(
            [torch.where(apply, u(1 - s, 1 + s), torch.ones(n, device=device)) for s in jitter]
            + [(u() < gray_p).float()],
            dim=1,
        )
    max_side = math.sqrt(scale[1] * H * W * ratio[1])  # ViewBuilder.crop's, for its sampling ratio
    return Views(box, flip, jit, size, max(1, math.ceil(max_side / size)), B, min(max_side, max(H, W)))


if triton is not None:

    @triton.jit
    def _roi_rows(start, bin_size, src, KK: tl.constexpr, SR: tl.constexpr):
        """(16, KK) roi_align weights of a window's pixels for output rows src (16,): SR samples per output pixel,
        each bilinear with torchvision's border rules (zero beyond [-1, KK], clamped to the last pixel), averaged."""
        j = tl.arange(0, KK)[None, :]
        r = tl.zeros((16, KK), tl.float32)
        for s in tl.static_range(SR):
            y = start + src[:, None] * bin_size + (s + 0.5) * bin_size / SR
            valid = (y >= -1.0) & (y <= KK)
            y = tl.maximum(y, 0.0)
            lo = y.to(tl.int32)
            top = lo >= KK - 1
            lo = tl.where(top, KK - 1, lo)
            hi = tl.where(top, KK - 1, lo + 1)
            y = tl.where(top, lo.to(tl.float32), y)
            ly = y - lo.to(tl.float32)
            w = tl.where(j == lo, 1.0 - ly, 0.0) + tl.where(j == hi, ly, 0.0)
            r += tl.where(valid, w, 0.0)
        return r / SR

    @triton.jit
    def _where(q, idx, S, T: tl.constexpr, G: tl.constexpr, SIZE: tl.constexpr, DENSE: tl.constexpr):
        """Output patch q -> (view row, frame, patch row, patch col, address of its channel 0 in out, row stride,
        channel stride, pivot group (frame))."""
        if DENSE:
            row = q // (T * G * G)
            i = q % (T * G * G)
        else:
            row = q // S
            i = tl.load(idx + q).to(tl.int32)
        t = i // (G * G)
        n = i % (G * G)
        prow = n // G
        pcol = n % G
        if DENSE:  # (rows, T, 3, SIZE, SIZE)
            base = ((row * T + t) * 3).to(tl.int64) * (SIZE * SIZE) + prow * 16 * SIZE + pcol * 16
            rs = SIZE
            cs = SIZE * SIZE
        else:  # (rows, S, 3 * 16 * 16), PatchEmbed's (c p1 p2)
            base = q.to(tl.int64) * 768
            rs = 16
            cs = 256
        return row, t, prow, pcol, base, rs, cs

    @triton.jit
    def _view_kernel(
        frames,
        sf,
        sh,
        H,
        W,
        box,
        flip,
        idx,
        jit,
        gsum,
        out,
        B,
        S,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        T: tl.constexpr,
        G: tl.constexpr,
        SIZE: tl.constexpr,
        SR: tl.constexpr,
        KK: tl.constexpr,
        DENSE: tl.constexpr,
        JITTER: tl.constexpr,
        PIVOT_FRAME: tl.constexpr,
    ):
        q = tl.program_id(0)
        ch = tl.program_id(1)
        row, t, prow, pcol, base, rs, cs = _where(q, idx, S, T, G, SIZE, DENSE)
        fl = tl.load(flip + row) != 0
        scol = tl.where(fl, G - 1 - pcol, pcol)  # the flip: output column pcol samples the crop's mirrored column
        x1 = tl.load(box + row * 4)
        y1 = tl.load(box + row * 4 + 1)
        bw = (tl.load(box + row * 4 + 2) - x1) / SIZE * 16  # frame pixels under 16 output pixels
        bh = (tl.load(box + row * 4 + 3) - y1) / SIZE * 16
        px = x1 + scol * bw
        py = y1 + prow * bh
        f = (row % B) * T + t
        # the window: clamped to the frame, a pixel past every bilinear tap from its edges (the frame's own edges
        # where it is clamped, so roi_align's border rules apply unchanged)
        r0 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.floor(py - 0.5) - 1, 0.0), (H - KK) * 1.0).to(tl.int32)
        c0 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.floor(px - 0.5) - 1, 0.0), (W - KK) * 1.0).to(tl.int32)
        ys = r0 + tl.arange(0, KK)
        xs = c0 + tl.arange(0, KK)
        win = tl.load(frames + f.to(tl.int64) * sf + ys[:, None] * sh + xs[None, :] * 3 + ch).to(tl.float32)
        a = tl.arange(0, 16).to(tl.float32)
        ry = _roi_rows(py - r0.to(tl.float32) - 0.5, bh / 16, a, KK, SR)  # (16, KK)
        hi = (ry.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
        z = tl.dot(hi, win, input_precision="tf32")  # win is integer valued (exact in tf32): split only the weights
        z = tl.dot(ry - hi, win, z, input_precision="tf32")  # (16, KK)
        rx = _roi_rows(px - c0.to(tl.float32) - 0.5, bw / 16, tl.where(fl, 15.0 - a, a), KK, SR)  # (16, KK)
        p = tl.dot(z, tl.trans(rx), input_precision="tf32x3") * (1.0 / 255.0)  # (16, 16)
        if JITTER:  # brightness; contrast etc. in _jitter_kernel, around the pivot group's mean
            p = tl.minimum(tl.maximum(p * tl.load(jit + row * 4), 0.0), 1.0)
            if PIVOT_FRAME:
                grp = row * T + t
            else:
                grp = row
            tl.atomic_add(gsum + grp * 2, tl.sum(p))
            tl.atomic_add(gsum + grp * 2 + 1, 256.0)  # the group's element count (kept patches per frame vary)
        else:
            mu = tl.where(ch == 0, mean0, tl.where(ch == 1, mean1, mean2))
            sd = tl.where(ch == 0, std0, tl.where(ch == 1, std1, std2))
            p = (p - mu) / sd
        tl.store(out + base + ch * cs + tl.arange(0, 16)[:, None] * rs + tl.arange(0, 16)[None, :], p)

    @triton.jit
    def _jitter_kernel(
        out,
        idx,
        jit,
        gsum,
        S,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        T: tl.constexpr,
        G: tl.constexpr,
        SIZE: tl.constexpr,
        DENSE: tl.constexpr,
        PIVOT_FRAME: tl.constexpr,
    ):
        """ViewBuilder.jitter after brightness: contrast around the pivot group's mean, saturation, grayscale, then
        normalize, one output patch (all three channels) per program."""
        q = tl.program_id(0)
        row, t, prow, pcol, base, rs, cs = _where(q, idx, S, T, G, SIZE, DENSE)
        if PIVOT_FRAME:
            grp = row * T + t
        else:
            grp = row
        mu = tl.load(gsum + grp * 2) / tl.load(gsum + grp * 2 + 1)
        con = tl.load(jit + row * 4 + 1)
        sat = tl.load(jit + row * 4 + 2)
        gray = tl.load(jit + row * 4 + 3) != 0
        o = out + base + tl.arange(0, 16)[:, None] * rs + tl.arange(0, 16)[None, :]
        r = tl.load(o)
        g = tl.load(o + cs)
        b = tl.load(o + 2 * cs)
        r = tl.minimum(tl.maximum((r - mu) * con + mu, 0.0), 1.0)
        g = tl.minimum(tl.maximum((g - mu) * con + mu, 0.0), 1.0)
        b = tl.minimum(tl.maximum((b - mu) * con + mu, 0.0), 1.0)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = tl.minimum(tl.maximum((r - lum) * sat + lum, 0.0), 1.0)
        g = tl.minimum(tl.maximum((g - lum) * sat + lum, 0.0), 1.0)
        b = tl.minimum(tl.maximum((b - lum) * sat + lum, 0.0), 1.0)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = tl.where(gray, lum, r)
        g = tl.where(gray, lum, g)
        b = tl.where(gray, lum, b)
        tl.store(o, (r - mean0) / std0)
        tl.store(o + cs, (g - mean1) / std1)
        tl.store(o + 2 * cs, (b - mean2) / std2)


def _window(p, H, W):
    """The source window side the kernel loads around a patch: the frame pixels under 16 output pixels at the
    largest box the draw allows, plus bilinear taps."""
    side = 16 * p.max_side / p.size
    for kk in (32, 64):
        if side + 4 <= kk and kk <= min(H, W):
            return kk
    raise ValueError(f"a {p.size}-px view's patches span {side:.0f} frame pixels: more than a 64-px window holds")


def _run(clip, p, idx, out, pivot, mean, std):
    B, T, H, W, C = clip.shape
    if C != 3 or clip.dtype != torch.uint8 or not clip.is_contiguous():
        raise ValueError("clip: a contiguous (B, T, H, W, 3) uint8 tensor")
    if p.size % PATCH:
        raise ValueError(f"view size {p.size}: not a multiple of {PATCH}")
    G = p.size // PATCH
    rows = p.box.shape[0]
    dense = idx is None
    S = T * G * G if dense else idx.shape[1]
    kk = _window(p, H, W)
    jitter = p.jitter is not None
    frame = pivot == "frame"
    if pivot not in ("frame", "view"):
        raise ValueError(f"pivot {pivot!r}: frame or view")
    gsum = torch.zeros(rows * T if frame else rows, 2, device=clip.device) if jitter else out
    idx_ = out if dense else idx
    flip = p.flip.view(torch.uint8)
    jit = p.jitter if jitter else p.box
    n = rows * S
    meta = {"T": T, "G": G, "SIZE": p.size, "DENSE": dense, "PIVOT_FRAME": frame}
    _view_kernel[(n, 3)](
        clip, clip.stride(1), clip.stride(2), H, W, p.box, flip, idx_, jit, gsum, out, p.B, S, *mean, *std,
        SR=p.sampling_ratio, KK=kk, JITTER=jitter, num_warps=2, **meta,
    )  # fmt: skip
    if jitter:
        _jitter_kernel[(n,)](out, idx_, p.jitter, gsum, S, *mean, *std, num_warps=4, **meta)
    return out


def view_frames(clip, p, out=None, mean=MEAN, std=STD):
    """Whole views of every clip: (rows, T, 3, size, size) float32, normalized (rows = views * B, row v * B + b)."""
    T, rows = clip.shape[1], p.box.shape[0]
    out = out if out is not None else torch.empty(rows, T, 3, p.size, p.size, device=clip.device)
    return _run(clip, p, None, out, "frame", mean, std)


def view_patches(clip, p, idx, out=None, pivot="frame", mean=MEAN, std=STD):
    """Only the patches idx (rows, S) keeps of each view (indices into its T * G * G patches, G = size / 16):
    (rows, S, 768) float32 in PatchEmbed's (c p1 p2) order, normalized. pivot: the contrast pivot's group, each frame's
    kept patches ("frame") or the whole view's ("view")."""
    rows, S = idx.shape
    out = out if out is not None else torch.empty(rows, S, 768, device=clip.device)
    return _run(clip, p, idx.contiguous(), out, pivot, mean, std)
