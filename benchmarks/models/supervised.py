"""Supervised: standard classification baseline with the same ViT backbone."""

from __future__ import annotations

import stable_pretraining as spt
from torch import nn

from benchmarks.models import (
    build_optim_config,
    collate_single,
    create_backbone,
    get_embedding_dim,
    ssl_augmentation,
    val_transform,
)


def create_transforms(ds_config):
    """Returns (train_transform, val_transform, collate_fn).

    Uses the same augmentation pipeline as the SSL models so the
    supervised baseline is a fair comparison.
    """
    h, w = ds_config.image_size
    train = ssl_augmentation(ds_config, (h, w), crop_scale=(0.08, 1.0))
    return train, val_transform(ds_config), collate_single


def forward(self, batch, stage):
    """Supervised forward: backbone → classifier → cross-entropy."""
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])
        out["label"] = batch["label"]
        self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)

    return out


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    classifier = nn.Linear(embed_dim, ds_config.num_classes)

    module = spt.Module(
        backbone=backbone,
        classifier=classifier,
        supervised_loss=nn.CrossEntropyLoss(),
        forward=forward,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim
