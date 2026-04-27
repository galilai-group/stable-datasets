"""Barlow Twins: redundancy-reduction with 2 augmented views."""

from __future__ import annotations

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from torch import nn

from benchmarks.models import (
    build_optim_config,
    collate_multiview,
    create_backbone,
    create_projector,
    get_embedding_dim,
    ssl_augmentation,
    val_transform,
)


NUM_VIEWS = 2


def create_transforms(ds_config, model_cfg=None):
    """Returns (train_transform, val_transform, collate_fn)."""
    h, w = ds_config.image_size
    view = ssl_augmentation(ds_config, (h, w), crop_scale=(0.08, 1.0))
    train = transforms.MultiViewTransform([view] * NUM_VIEWS)
    return train, val_transform(ds_config), collate_multiview


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim)
    barlow_loss = spt.losses.BarlowTwinsLoss(lambd=cfg.model.loss.lambda_coeff)
    # Replace LazyBatchNorm1d with concrete BatchNorm1d to avoid
    # UninitializedParameter errors (e.g. numel() calls before first forward)
    barlow_loss.bn = nn.BatchNorm1d(cfg.model.projector.output_dim)
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.barlow_twins_forward,
        barlow_loss=barlow_loss,
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
