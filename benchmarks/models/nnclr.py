"""NNCLR: nearest-neighbor contrastive learning with 2 augmented views."""

from __future__ import annotations

import stable_pretraining as spt
import torch
from stable_pretraining.callbacks.queue import OnlineQueue, find_or_create_queue_callback
from stable_pretraining.data import transforms
from stable_pretraining.forward import _find_nearest_neighbors, _get_views_list
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


def forward(self, batch, stage):
    """NNCLR forward — fork of ``spt.forward.nnclr_forward`` with a fix that
    moves the lazily-created support-set queue onto the model's device.

    The upstream implementation creates the OnlineQueue's underlying buffer
    at ``setup()`` time *after* the parent module has already been moved to
    the GPU. Lightning does not retroactively move children added after the
    initial ``.to(device)``, so the queue ends up on CPU and the matmul in
    ``_find_nearest_neighbors`` fails with a device mismatch. We move it
    explicitly the first time we touch it.
    """
    out = {}
    views = _get_views_list(batch)

    if views is None:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]
        return out

    if len(views) != 2:
        raise ValueError(f"NNCLR requires exactly 2 views, got {len(views)}.")

    embeddings = [self.backbone(view["image"]) for view in views]
    out["embedding"] = torch.cat(embeddings, dim=0)
    if "label" in views[0]:
        out["label"] = torch.cat([view["label"] for view in views], dim=0)

    if not self.training:
        return out

    if not hasattr(self, "_nnclr_queue_callback"):
        cb = find_or_create_queue_callback(
            self.trainer,
            key="nnclr_support_set",
            queue_length=self.hparams.support_set_size,
            dim=self.hparams.projection_dim,
        )
        # Move the lazily-created shared queue onto the model's device.
        queue_mod = OnlineQueue._shared_queues.get(cb.key)
        if queue_mod is not None:
            queue_mod.to(self.device)
        self._nnclr_queue_callback = cb

    queue_callback = self._nnclr_queue_callback
    projections = [self.projector(emb) for emb in embeddings]
    proj_q, proj_k = projections[0], projections[1]
    support_set = OnlineQueue._shared_queues.get(queue_callback.key).get()

    if support_set is not None and len(support_set) > 0:
        pred_q = self.predictor(proj_q)
        pred_k = self.predictor(proj_k)
        nn_k = _find_nearest_neighbors(proj_k, support_set).detach()
        nn_q = _find_nearest_neighbors(proj_q, support_set).detach()
        loss_a = self.nnclr_loss(pred_q, nn_k)
        loss_b = self.nnclr_loss(pred_k, nn_q)
        out["loss"] = (loss_a + loss_b) / 2.0
    else:
        # Queue not yet populated → fall back to SimCLR-style positive pair.
        out["loss"] = self.nnclr_loss(proj_q, proj_k)

    self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)
    out["nnclr_support_set"] = torch.cat(projections)
    return out


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
        forward=forward,
        nnclr_loss=spt.losses.NTXEntLoss(temperature=cfg.model.loss.temperature),
        hparams={
            "support_set_size": cfg.model.queue_size,
            "projection_dim": proj_out,
        },
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
