"""Transform ops: each draws its random parameters for a batch first, then applies them.

Drawing first lets a reader fetch and decode only the input region the ops need (``region``), and lets a fused
kernel take the same parameters as the reference ``apply``, so their outputs can be compared exactly.

A sample is a torch tensor ``(..., C, *spatial)``: (C, H, W) for an image, (T, C, H, W) for a clip (every frame
gets the same parameters), (C, D, H, W) for a volume. Boxes are starts then stops per spatial axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Op:
    """One transform. ``draw`` returns per-sample parameter arrays (first axis: sample); ``out_shape`` gives the
    spatial shape each sample has after the op; ``apply`` is the reference implementation for one sample. Chain
    adds ``_in``, the spatial shape of each sample's input, to every op's parameters.

    ``crops``: the op reads a box of its input (``region``), so a reader can fetch just that box. ``pointwise``: each
    output value depends on its input value alone, so the op can run on any region of its input."""

    ndim = 2
    crops = False
    pointwise = False

    def draw(self, rng: np.random.Generator, shapes: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def out_shape(self, params: dict, shapes: np.ndarray) -> np.ndarray:
        return shapes

    def region(self, params: dict, shapes: np.ndarray) -> np.ndarray:
        """(n, 2 * ndim) input boxes the op reads, clipped to the input."""
        return np.concatenate([np.zeros_like(shapes), shapes], 1)

    def apply(self, x, params: dict, i: int, origin=None):
        """Sample ``i`` with its parameters. ``origin``: where ``x`` starts in the full input, when ``x`` is only
        the op's region of it."""
        raise NotImplementedError

    def signature(self) -> tuple:
        return (type(self).__name__,)


def _sizes(size, ndim):
    return (size,) * ndim if isinstance(size, int) else tuple(size)


@dataclass(frozen=True)
class RandomCrop(Op):
    """A crop of ``size`` at a uniform random position; inputs smaller than ``size`` on an axis are zero-padded at
    the end. N-D: ``size`` has one entry per spatial axis."""

    size: tuple | int
    crops = True

    def _size(self, ndim):
        return np.array(_sizes(self.size, ndim), np.int64)

    def draw(self, rng, shapes):
        size = self._size(shapes.shape[1])
        room = np.maximum(shapes - size, 0)
        return {"start": np.floor(rng.random(shapes.shape) * (room + 1)).astype(np.int64)}

    def out_shape(self, params, shapes):
        return np.broadcast_to(self._size(shapes.shape[1]), shapes.shape).copy()

    def region(self, params, shapes):
        start = params["start"]
        return np.concatenate([start, np.minimum(start + self._size(shapes.shape[1]), shapes)], 1)

    def apply(self, x, params, i, origin=None):
        import torch.nn.functional as F

        nd = len(params["start"][i])
        size = self._size(nd)
        start = params["start"][i] - (0 if origin is None else np.asarray(origin))
        spatial = np.array(x.shape[-nd:])
        stop = np.minimum(start + size, spatial)
        out = x[(..., *[slice(int(a), int(b)) for a, b in zip(start, stop)])]
        pad = size - (stop - start)
        if pad.any():
            out = F.pad(out, [p for q in reversed(pad.tolist()) for p in (0, q)])
        return out

    def signature(self):
        return ("RandomCrop", self.size)


@dataclass(frozen=True)
class RandomResizedCrop(Op):
    """A box of random area (``scale`` of the input's) and aspect ratio (``ratio``), resampled to ``size`` x
    ``size`` by roi_align (aligned, the mean of sampling_ratio^2 bilinear samples per output pixel), output float in
    [0, 1] from uint8 input. The draw is LeVJEPA's ViewBuilder.crop (no rejection: sides clamped to the input),
    which framestore.views fuses with the ops after it. From only its region, the output equals the whole
    input's to float rounding (about 1e-6): roi_align's sample positions are computed from region coordinates."""

    size: int = 224
    scale: tuple = (0.08, 1.0)
    ratio: tuple = (3 / 4, 4 / 3)
    crops = True

    def draw(self, rng, shapes):
        n = len(shapes)
        H, W = shapes[:, 0].astype(np.float64), shapes[:, 1].astype(np.float64)
        area = H * W * rng.uniform(*self.scale, n)
        r = np.exp(rng.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]), n))
        w = np.minimum(np.sqrt(area * r), W)
        h = np.minimum(np.sqrt(area / r), H)
        x0 = rng.random(n) * (W - w)
        y0 = rng.random(n) * (H - h)
        # float32, the precision roi_align and the fused kernel compute in
        return {"box": np.stack([y0, x0, y0 + h, x0 + w], 1).astype(np.float32)}

    def sampling_ratio(self, shape) -> int:
        max_side = math.sqrt(self.scale[1] * shape[0] * shape[1] * self.ratio[1])
        return max(1, math.ceil(max_side / self.size))

    def out_shape(self, params, shapes):
        return np.full_like(shapes, self.size)

    def region(self, params, shapes):
        """Every pixel a bilinear tap reaches, plus a margin so roi_align's border rules see no false edge."""
        b = params["box"]
        lo = np.floor(b[:, :2] - 0.5).astype(np.int64) - 1
        hi = np.floor(b[:, 2:] - 0.5).astype(np.int64) + 3
        return np.concatenate([np.clip(lo, 0, shapes), np.clip(hi, 0, shapes)], 1)

    def apply(self, x, params, i, origin=None):
        import torch
        from torchvision.ops import roi_align

        y0, x0, y1, x1 = params["box"][i]
        if origin is not None:
            y0, y1, x0, x1 = y0 - origin[0], y1 - origin[0], x0 - origin[1], x1 - origin[1]
        lead = x.shape[:-3]
        frames = x.reshape(-1, *x.shape[-3:])
        frames = frames.float() if frames.is_floating_point() else frames.float().div(255)
        rois = torch.tensor([[k, x0, y0, x1, y1] for k in range(len(frames))], dtype=torch.float32, device=x.device)
        out = roi_align(frames, rois, self.size, 1.0, self.sampling_ratio(params["_in"][i]), aligned=True)
        return out.reshape(*lead, *out.shape[-3:])

    def signature(self):
        return ("RandomResizedCrop", self.size, tuple(self.scale), tuple(self.ratio))


@dataclass(frozen=True)
class Resize(Op):
    """Bilinear resize (antialiased) to ``size``; output float in [0, 1] from uint8 input."""

    size: tuple | int

    def out_shape(self, params, shapes):
        return np.broadcast_to(np.array(_sizes(self.size, shapes.shape[1])), shapes.shape).copy()

    def apply(self, x, params, i, origin=None):
        import torch.nn.functional as F

        size = _sizes(self.size, 2)
        y = x.float().div(255) if not x.is_floating_point() else x.float()
        lead = y.shape[:-3]
        y = y.reshape(-1, *y.shape[-3:])
        y = F.interpolate(y, size=size, mode="bilinear", antialias=True, align_corners=False)
        return y.reshape(*lead, *y.shape[-3:])

    def signature(self):
        return ("Resize", _sizes(self.size, 2))


@dataclass(frozen=True)
class Flip(Op):
    """Mirror the last spatial axis with probability ``p``."""

    p: float = 0.5

    def draw(self, rng, shapes):
        return {"flip": rng.random(len(shapes)) < self.p}

    def apply(self, x, params, i, origin=None):
        return x.flip(-1) if params["flip"][i] else x


@dataclass(frozen=True)
class ColorJitter(Op):
    """With probability ``p``: brightness, contrast (around the mean of each frame after brightness) and saturation
    factors uniform in [1 - s, 1 + s]; then grayscale with probability ``gray_p``. Input float in [0, 1].
    LeVJEPA's ViewBuilder.jitter."""

    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    p: float = 0.8
    gray_p: float = 0.2

    def draw(self, rng, shapes):
        n = len(shapes)
        on = rng.random(n) < self.p
        f = [
            np.where(on, rng.uniform(1 - s, 1 + s, n), 1.0) for s in (self.brightness, self.contrast, self.saturation)
        ]
        return {"jitter": np.stack(f + [(rng.random(n) < self.gray_p).astype(np.float64)], 1)}

    def apply(self, x, params, i, origin=None):
        j = params["jitter"][i]

        def lum(y):
            return 0.299 * y[..., 0:1, :, :] + 0.587 * y[..., 1:2, :, :] + 0.114 * y[..., 2:3, :, :]

        if j[0] != 1:
            x = (x * float(j[0])).clamp(0, 1)
        if j[1] != 1:
            mu = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = ((x - mu) * float(j[1]) + mu).clamp(0, 1)
        if j[2] != 1:
            lu = lum(x)
            x = ((x - lu) * float(j[2]) + lu).clamp(0, 1)
        return lum(x).expand_as(x) if j[3] else x

    def signature(self):
        return ("ColorJitter", self.brightness, self.contrast, self.saturation, self.p, self.gray_p)


@dataclass(frozen=True)
class Normalize(Op):
    """(x - mean) / std per channel (the channel axis is -3)."""

    mean: tuple = field(default=MEAN)
    std: tuple = field(default=STD)
    pointwise = True

    def apply(self, x, params, i, origin=None):
        import torch

        m = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        s = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        return (x - m) / s

    def signature(self):
        return ("Normalize", tuple(self.mean), tuple(self.std))
