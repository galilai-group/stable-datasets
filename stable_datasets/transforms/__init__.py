"""Transforms that draw their random parameters before reading any data.

    tf = Chain(RandomResizedCrop(224, scale=(0.3, 1.0)), Flip(), ColorJitter(), Normalize())
    params = tf.draw(rng, shapes)          # per-op parameter arrays for a batch, shapes (n, 2): each input's H, W
    boxes = tf.region(params)              # the input box each sample needs: read and decode only that
    y = tf.apply(region_pixels, params, i, origin=boxes[i, :2])

The reference ``apply`` of each op is plain torch; fast paths (fused GPU kernels) take the same parameters and are
tested against it.
"""

from .chain import Chain
from .ops import MEAN, STD, ColorJitter, Flip, Normalize, Op, RandomCrop, RandomResizedCrop, Resize


__all__ = [
    "MEAN",
    "STD",
    "Chain",
    "ColorJitter",
    "Flip",
    "Normalize",
    "Op",
    "RandomCrop",
    "RandomResizedCrop",
    "Resize",
]
