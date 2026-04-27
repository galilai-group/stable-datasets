"""NNCLR: nearest-neighbor contrastive learning with 2 augmented views."""

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
    proj_out = cfg.model.projector.output_dim
    projector = create_projector(embed_dim, cfg.model.projector.hidden_dim, proj_out)
    predictor = nn.Sequential(
        nn.Linear(proj_out, cfg.model.predictor.hidden_dim),
        nn.BatchNorm1d(cfg.model.predictor.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.model.predictor.hidden_dim, cfg.model.predictor.output_dim),
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        forward=forward.nnclr_forward,
        nnclr_loss=spt.losses.NTXEntLoss(temperature=cfg.model.loss.temperature),
        hparams={
            "support_set_size": cfg.model.queue_size,
            "projection_dim": proj_out,
        },
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
