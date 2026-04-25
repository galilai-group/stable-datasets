"""LeJEPA: multi-view invariance + Epps-Pulley SIGReg (canonical).

Thin wrapper around :class:`benchmarks.models._lejepa_canonical.LeJEPA`
(vendored from upstream stable-pretraining). This module contributes:

* DINO-style multi-crop: 2 global views (full ``ds_config.image_size``) +
  6 local views (half resolution) — matching the canonical LeJEPA recipe.
* A backbone pooler that adapts ``spt.backbone.vit_hf`` (HF-style output
  with ``.last_hidden_state``) into the ``(B, embed_dim)`` tensor the
  canonical ``LeJEPA.forward`` expects, and enables
  ``interpolate_pos_encoding`` so local views work with pretrained-shape
  position embeddings.
* A forward function that unpacks ``global_*`` / ``local_*`` named views
  from ``collate_multicrop`` batches and delegates to ``LeJEPA.forward``.
"""

from __future__ import annotations

import stable_pretraining as spt
import torch
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_by_prefix
from torch import nn

from benchmarks.models import (
    build_optim_config,
    collate_multicrop,
    get_embedding_dim,
    val_transform,
)
from benchmarks.models._lejepa_canonical import LeJEPA


NUM_GLOBAL = 2
NUM_LOCAL = 6


# Transforms


def _photometric(ds_config) -> list:
    ops = [transforms.RandomHorizontalFlip(p=0.5)]
    if ds_config.channels != 1:
        ops.extend(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
    ops.append(transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5))
    if ds_config.channels != 1:
        ops.append(transforms.RandomSolarize(threshold=128, p=0.2))
    return ops


def _crop_transform(ds_config, size, scale):
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(size, scale=scale),
        *_photometric(ds_config),
        transforms.ToImage(mean=ds_config.mean, std=ds_config.std),
    )


def create_transforms(ds_config):
    h, w = ds_config.image_size
    local_size = (max(h // 2, 32), max(w // 2, 32))

    global_aug = _crop_transform(ds_config, (h, w), scale=(0.3, 1.0))
    local_aug = _crop_transform(ds_config, local_size, scale=(0.05, 0.3))

    transform_dict = {
        **{f"global_{i + 1}": global_aug for i in range(NUM_GLOBAL)},
        **{f"local_{i + 1}": local_aug for i in range(NUM_LOCAL)},
    }
    train = transforms.MultiViewTransform(transform_dict)
    return train, val_transform(ds_config), collate_multicrop


# Backbone adapter: HF ViT → (B, embed_dim) with interpolated pos-encoding


class _CLSPooledViT(nn.Module):
    """Wrap ``spt.backbone.vit_hf`` to return CLS-pooled ``(B, embed_dim)``.

    Always passes ``interpolate_pos_encoding=True`` so local (smaller) crops
    reuse the same pos-embed interpolated to their patch grid.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.bb = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bb(x, interpolate_pos_encoding=True)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        if out.dim() == 3:
            return out[:, 0, :]
        return out


# Forward


def forward(self, batch, stage):
    out = {}
    global_views, local_views, all_views = _get_views_by_prefix(
        batch, global_prefix="global", local_prefix="local"
    )

    if all_views is None:
        # Eval path: single-view batch from val loader.
        output = self.model(images=batch["image"])
        out["embedding"] = output.embedding
        if "label" in batch:
            out["label"] = batch["label"]
        return out

    # Training path: multi-view.
    if "label" in all_views[0]:
        out["label"] = torch.cat([v["label"] for v in global_views], dim=0)

    global_imgs = [v["image"] for v in global_views]
    local_imgs = [v["image"] for v in local_views]

    output = self.model(global_views=global_imgs, local_views=local_imgs)

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(f"{stage}/sigreg", output.sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


# Builder


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    h = ds_config.image_size[0]
    hf_backbone = spt.backbone.vit_hf(
        size=cfg.backbone.size,
        patch_size=cfg.backbone.patch_size,
        image_size=h,
        pretrained=False,
    )
    embed_dim = get_embedding_dim(hf_backbone)
    backbone = _CLSPooledViT(hf_backbone)

    model = LeJEPA(
        backbone=backbone,
        embed_dim=embed_dim,
        n_slices=cfg.model.loss.num_slices,
        t_max=cfg.model.loss.t_max,
        n_points=cfg.model.loss.n_points,
        lamb=cfg.model.loss.lamb,
    )

    module = spt.Module(
        model=model,
        forward=forward,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim
