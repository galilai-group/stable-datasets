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
    patch_size: int | None = None,
    **kwargs,
) -> nn.Module:
    extra = {}
    if "vit" in name:
        extra["dynamic_img_size"] = True
        if img_size is not None:
            extra["img_size"] = img_size
        if patch_size is not None:
            extra["patch_size"] = patch_size
    return timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        drop_path_rate=0.0,
        **extra,
        **kwargs,
    )


def create_resnet(name: str, in_chans: int = 3, **kwargs) -> nn.Module:
    """CNN factory: timm ResNet with a global-average-pooled feature output.

    ``num_classes=0`` drops the classifier so the model returns a pooled
    ``[N, num_features]`` vector (2048 for resnet50) — the same 2D shape the
    SSL projectors and probes expect from a ViT with ``num_classes=0``. ViT-only
    kwargs (``img_size``/``patch_size``/``dynamic_img_size``) do not apply.
    """
    return timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
        **kwargs,
    )
