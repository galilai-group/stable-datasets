"""ViT factory: thin wrapper around timm.create_model.

Backbone names follow timm conventions (e.g. ``vit_small_patch16_224``) and
are passed straight through. ``dynamic_img_size=True`` enables pos-embed
interpolation so a 224-named ViT can run at any image size.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def create_vit(
    name: str,
    img_size: int | tuple[int, int] | None = None,
    in_chans: int = 3,
    **kwargs,
) -> nn.Module:
    extra = {}
    if "vit" in name:
        extra["dynamic_img_size"] = True
        if img_size is not None:
            extra["img_size"] = img_size
    return timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        drop_path_rate=0.0,
        **extra,
        **kwargs,
    )
