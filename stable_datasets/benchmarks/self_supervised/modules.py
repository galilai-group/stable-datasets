"""Module builders for SSL benchmarks.

Registry of per-model builder functions that create spt.Module instances.
Each builder is self-contained (~25-35 lines) and uses shared utilities
for backbone creation, projectors, and evaluation callbacks.
"""

from __future__ import annotations

import logging
import re

import stable_pretraining as spt
import torch
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from stable_pretraining import forward
from torch import nn

from .dataset import DatasetConfig
from .lejepa_losses import EppsPulley, SlicingUnivariateTest
from .models import create_vit

log = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 512


# =============================================================================
# Shared Utilities
# =============================================================================


def get_embedding_dim(backbone: nn.Module) -> int:
    """Get embedding dimension from a backbone, trying multiple conventions."""
    if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
        return backbone.config.hidden_size
    for attr in ("num_features", "embed_dim"):
        if hasattr(backbone, attr):
            return getattr(backbone, attr)
    if hasattr(backbone, "fc") and hasattr(backbone.fc, "in_features"):
        return backbone.fc.in_features
    return DEFAULT_EMBED_DIM


def create_backbone(backbone_cfg, ds_config: DatasetConfig) -> nn.Module:
    """Create a backbone that outputs flat embeddings (CLS token for ViT).

    For DINO (HF ViT) and MAE (MaskedEncoder), see their model-specific builders.
    """
    if backbone_cfg.type == "resnet":
        backbone = spt.backbone.from_torchvision(
            backbone_cfg.model_name, low_resolution=ds_config.low_resolution
        )
        backbone.fc = nn.Identity()
        return backbone

    if backbone_cfg.type == "vit":
        return create_vit(
            size=backbone_cfg.size,
            img_size=ds_config.image_size,
            patch_size=backbone_cfg.patch_size,
        )

    raise ValueError(f"Unknown backbone type: {backbone_cfg.type}")


def create_projector(embed_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Create a 3-layer MLP projection head with BatchNorm."""
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def build_optim_config(model_cfg) -> dict:
    """Build optimizer config dict from model config."""
    optim = {
        "optimizer": {
            "type": model_cfg.optimizer.type,
            "lr": model_cfg.optimizer.lr,
            "weight_decay": model_cfg.optimizer.weight_decay,
        },
        "scheduler": {"type": model_cfg.scheduler.type},
        "interval": "epoch",
    }
    if hasattr(model_cfg.optimizer, "betas"):
        optim["optimizer"]["betas"] = tuple(model_cfg.optimizer.betas)
    return optim


def create_eval_callbacks(
    module: spt.Module, ds_config: DatasetConfig, embed_dim: int
) -> list:
    """Create linear probe and KNN evaluation callbacks."""
    num_classes = ds_config.num_classes
    callbacks = []

    if num_classes > 0:
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(
                num_classes, top_k=min(5, num_classes)
            ),
        }
        linear_probe = spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(embed_dim, num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics=metrics,
        )
        callbacks.append(linear_probe)

        knn_probe = spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=20000,
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                "top5": torchmetrics.classification.MulticlassAccuracy(
                    num_classes, top_k=min(5, num_classes)
                ),
            },
            input_dim=embed_dim,
            k=10,
        )
        callbacks.append(knn_probe)

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


# =============================================================================
# Custom Forward Functions (for models not in stable-pretraining)
# =============================================================================


def _mae_patchify(images, patch_size):
    """Convert images to patches [N, T, P*C]."""
    N, C, H, W = images.shape
    p = patch_size
    h, w = H // p, W // p
    x = images.reshape(N, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(N, h * w, p * p * C)


def _mae_forward(self, batch, stage):
    """MAE forward: masked patch reconstruction."""
    images = batch["image"]
    encoder_out = self.backbone(images)
    out = {"embedding": encoder_out.encoded[:, 0]}

    if "label" in batch:
        out["label"] = batch["label"]

    if self.training:
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        visible_patches = encoder_out.encoded[:, num_prefix:]
        pred = self.decoder(visible_patches, encoder_out.mask)
        target = _mae_patchify(images, self.patch_size)
        loss = spt.losses.mae(target, pred, encoder_out.mask)
        out["loss"] = loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    return out


def _lejepa_forward(self, batch, stage):
    """LeJEPA forward: multi-view invariance + SIGReg regularization."""
    out = {}
    views = batch.get("views")

    if views is not None:
        V, N = len(views), views[0]["image"].size(0)
        all_images = torch.cat([v["image"] for v in views], dim=0)
        all_emb = self.backbone(all_images)
        out["embedding"] = all_emb

        if "label" in views[0]:
            out["label"] = torch.cat([v["label"] for v in views], dim=0)

        if self.training:
            all_proj = self.projector(all_emb)
            proj_stacked = all_proj.reshape(V, N, -1)
            view_mean = proj_stacked.mean(0)
            inv_loss = (view_mean - proj_stacked).square().mean()

            if isinstance(self.sigreg_loss, SlicingUnivariateTest) and isinstance(
                self.sigreg_loss.univariate_test, EppsPulley
            ):
                sigreg_loss = self.sigreg_loss(proj_stacked)
            else:
                sigreg_loss = self.sigreg_loss(
                    proj_stacked.reshape(-1, proj_stacked.size(-1))
                )

            lamb = getattr(self, "lamb", 0.02)
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            out["loss"] = lejepa_loss

            self.log(f"{stage}/sigreg", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv", inv_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/lejepa", lejepa_loss, on_step=True, on_epoch=True, sync_dist=True)
    else:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


# =============================================================================
# Per-Model Builders
# =============================================================================


def build_simclr(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(
        embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr_forward,
        simclr_loss=spt.losses.NTXEntLoss(temperature=cfg.model.loss.temperature),
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim


def build_barlow_twins(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(
        embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.barlow_twins_forward,
        barlow_loss=spt.losses.BarlowTwinsLoss(lambd=cfg.model.loss.lambda_coeff),
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim


def build_nnclr(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    proj_out = cfg.model.projector.output_dim
    projector = create_projector(
        embed_dim, cfg.model.projector.hidden_dim, proj_out
    )
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


def build_lejepa(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(
        embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim
    )
    univariate_test = EppsPulley(
        t_max=cfg.model.loss.t_max, n_points=cfg.model.loss.n_points
    )
    sigreg_loss = SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=cfg.model.loss.num_slices,
        reduction="mean",
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        sigreg_loss=sigreg_loss,
        lamb=cfg.model.loss.lamb,
        forward=_lejepa_forward,
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim


def build_dino(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    h = ds_config.image_size[0]
    backbone = spt.backbone.vit_hf(
        size=cfg.backbone.size,
        patch_size=cfg.backbone.patch_size,
        image_size=h,
        pretrained=False,
    )
    embed_dim = get_embedding_dim(backbone)
    backbone_wrapper = spt.backbone.TeacherStudentWrapper(
        student=backbone, base_ema_coefficient=cfg.model.momentum_teacher
    )

    projector = nn.Sequential(
        nn.Linear(embed_dim, cfg.model.projector.hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.model.projector.hidden_dim, cfg.model.projector.hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.model.projector.hidden_dim, cfg.model.projector.bottleneck_dim),
        spt.utils.nn_modules.L2Norm(),
        nn.Linear(cfg.model.projector.bottleneck_dim, cfg.model.projector.output_dim, bias=False),
    )
    projector_wrapper = spt.backbone.TeacherStudentWrapper(
        student=projector, base_ema_coefficient=cfg.model.momentum_teacher
    )

    dino_loss = spt.losses.DINOv1Loss(
        temperature_student=cfg.model.loss.temperature_student,
        center_momentum=0.9,
    )

    module = spt.Module(
        backbone=backbone_wrapper,
        projector=projector_wrapper,
        dino_loss=dino_loss,
        temperature_teacher=cfg.model.loss.temperature_teacher,
        warmup_temperature_teacher=cfg.model.loss.warmup_temperature_teacher,
        warmup_epochs_temperature_teacher=cfg.model.loss.warmup_epochs_temperature_teacher,
        forward=forward.dino_forward,
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim


def _adjust_mae_params_for_low_resolution(ds_config, patch_size, decoder_embed_dim, decoder_depth):
    """Scale down MAE params for small images."""
    h, w = ds_config.image_size
    if not ds_config.low_resolution:
        return patch_size, decoder_embed_dim, decoder_depth
    return (
        min(patch_size, min(h, w) // 4),
        min(decoder_embed_dim, 256),
        min(decoder_depth, 4),
    )


def build_mae(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    h, w = ds_config.image_size
    patch_size = cfg.backbone.patch_size
    decoder_embed_dim = cfg.model.decoder.embed_dim
    decoder_depth = cfg.model.decoder.depth

    patch_size, decoder_embed_dim, decoder_depth = _adjust_mae_params_for_low_resolution(
        ds_config, patch_size, decoder_embed_dim, decoder_depth
    )

    grid_size = (h // patch_size, w // patch_size)
    num_patches = grid_size[0] * grid_size[1]
    output_dim = patch_size * patch_size * 3

    vit_model = create_vit(
        size=cfg.backbone.size,
        img_size=(h, w),
        patch_size=patch_size,
    )
    masking = spt.backbone.PatchMasking(mask_ratio=cfg.model.mask_ratio)
    encoder = spt.backbone.MaskedEncoder(model_or_model_name=vit_model, masking=masking)
    encoder_embed_dim = getattr(encoder, "embed_dim", 768)

    decoder = spt.backbone.MAEDecoder(
        embed_dim=encoder_embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        output_dim=output_dim,
        num_patches=num_patches,
        grid_size=grid_size,
        depth=decoder_depth,
        num_heads=cfg.model.decoder.num_heads,
    )

    module = spt.Module(
        backbone=encoder,
        decoder=decoder,
        forward=_mae_forward,
        patch_size=patch_size,
        optim=build_optim_config(cfg.model),
    )
    return module, encoder_embed_dim


# =============================================================================
# Registry
# =============================================================================

MODEL_BUILDERS = {
    "simclr": build_simclr,
    "dino": build_dino,
    "mae": build_mae,
    "lejepa": build_lejepa,
    "nnclr": build_nnclr,
    "barlow_twins": build_barlow_twins,
}


def build_module(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    """Build a module from Hydra config using the model registry.

    Returns:
        Tuple of (module, embedding_dim).
    """
    model_name = cfg.model.name
    if model_name not in MODEL_BUILDERS:
        available = ", ".join(MODEL_BUILDERS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_BUILDERS[model_name](cfg, ds_config)
