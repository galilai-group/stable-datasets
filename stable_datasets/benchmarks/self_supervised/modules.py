"""Module builders for SSL benchmarks.

Registry of per-model builder functions that create spt.Module instances.
Each builder is self-contained (~25-35 lines) and uses shared utilities
for backbone creation, projectors, and evaluation callbacks.
"""

from __future__ import annotations

import logging

import stable_pretraining as spt
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from stable_pretraining import forward
from stable_pretraining.forward import _get_views_by_prefix
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
    """Create a ViT backbone that outputs flat embeddings (CLS token).

    For DINO (HF ViT) and MAE (MaskedEncoder), see their model-specific builders.
    """
    if backbone_cfg.type == "vit":
        return create_vit(
            size=backbone_cfg.size,
            img_size=ds_config.image_size,
            patch_size=backbone_cfg.patch_size,
        )

    raise ValueError(f"Unknown backbone type: {backbone_cfg.type}. Only 'vit' is supported.")


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


def build_optim_config(model_cfg, backbone_cfg) -> dict:
    """Build optimizer config dict from model config."""
    if hasattr(model_cfg, "vit_optimizer"):
        opt_cfg = model_cfg.vit_optimizer
    else:
        opt_cfg = model_cfg.optimizer

    scheduler_cfg = {"type": model_cfg.scheduler.type}
    # Use explicitly computed total_steps instead of estimated_stepping_batches
    # to avoid double-counting accumulate_grad_batches (batch_size is already
    # divided by accum for memory, and the Module frequency mechanism also
    # divides by accum).
    if hasattr(model_cfg, "_total_steps"):
        total_steps = int(model_cfg._total_steps)
        scheduler_cfg["total_steps"] = total_steps
        scheduler_cfg["peak_step"] = max(1, int(0.01 * total_steps))

    optim = {
        "optimizer": {
            "type": opt_cfg.type,
            "lr": opt_cfg.lr,
            "weight_decay": opt_cfg.weight_decay,
        },
        "scheduler": scheduler_cfg,
        "interval": "step",
    }
    if hasattr(opt_cfg, "betas"):
        optim["optimizer"]["betas"] = tuple(opt_cfg.betas)
    # Apply per-dataset LR override if present
    if hasattr(model_cfg, "_lr_override"):
        optim["optimizer"]["lr"] = model_cfg._lr_override
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
        norm_pix = getattr(self, "norm_pix_loss", True)
        loss = spt.losses.mae(target, pred, encoder_out.mask, norm_pix_loss=norm_pix)
        out["loss"] = loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    return out


def _dino_forward(self, batch, stage):
    """DINO forward with accumulation-aware centering.

    When using gradient accumulation (batch_size=32, accum=8 for effective 256),
    the vanilla DINO center update applies one EMA step per micro-batch. This
    causes the center to update 8x per effective batch with 8x noisier estimates,
    destabilizing the center and leading to mode collapse.

    Fix: accumulate center contributions across micro-batches, apply ONE EMA
    update per effective batch.
    """
    out = {}

    global_views, local_views, all_views = _get_views_by_prefix(
        batch, global_prefix="global", local_prefix="local"
    )

    if all_views is not None:
        n_global = len(global_views)
        n_local = len(local_views)
    else:
        images = batch["image"]
        if "label" in batch:
            out["label"] = batch["label"]
        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    batch_size = all_views[0]["image"].shape[0]

    if "label" in all_views[0]:
        if self.training:
            out["label"] = torch.cat([view["label"] for view in global_views], dim=0)
        else:
            out["label"] = torch.cat([view["label"] for view in all_views], dim=0)

    if not self.training:
        all_images = torch.cat([view["image"] for view in all_views], dim=0)
        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(all_images)
        out["embedding"] = teacher_features.last_hidden_state[:, 0].detach()
        return out

    # --- Training ---
    global_images = torch.cat([view["image"] for view in global_views], dim=0)

    with torch.no_grad():
        teacher_features = self.backbone.forward_teacher(global_images)
        if hasattr(teacher_features, "last_hidden_state"):
            teacher_cls_features = teacher_features.last_hidden_state[:, 0, :]
        else:
            teacher_cls_features = (
                teacher_features[:, 0, :]
                if teacher_features.ndim == 3
                else teacher_features
            )
        teacher_logits = self.projector.forward_teacher(teacher_cls_features)
        teacher_logits = teacher_logits.view(n_global, batch_size, -1)

    # Student: all views
    student_logits_list = []
    student_features = self.backbone.forward_student(global_images)
    if hasattr(student_features, "last_hidden_state"):
        student_cls_features = student_features.last_hidden_state[:, 0, :]
    else:
        student_cls_features = (
            student_features[:, 0, :]
            if student_features.ndim == 3
            else student_features
        )
    student_global_logits = self.projector.forward_student(student_cls_features)
    student_global_logits = student_global_logits.view(n_global, batch_size, -1)
    student_logits_list.append(student_global_logits)

    if n_local > 0:
        local_images = torch.cat([view["image"] for view in local_views], dim=0)
        student_features = self.backbone.forward_student(
            local_images, interpolate_pos_encoding=True
        )
        if hasattr(student_features, "last_hidden_state"):
            student_local_cls = student_features.last_hidden_state[:, 0, :]
        else:
            student_local_cls = (
                student_features[:, 0, :]
                if student_features.ndim == 3
                else student_features
            )
        student_local_logits = self.projector.forward_student(student_local_cls)
        student_local_logits = student_local_logits.view(n_local, batch_size, -1)
        student_logits_list.append(student_local_logits)

    student_logits = torch.cat(student_logits_list, dim=0)

    # Temperature scheduling
    if (
        hasattr(self, "warmup_epochs_temperature_teacher")
        and hasattr(self, "warmup_temperature_teacher")
        and hasattr(self, "temperature_teacher")
    ):
        if self.current_epoch < self.warmup_epochs_temperature_teacher:
            progress = self.current_epoch / self.warmup_epochs_temperature_teacher
            temperature_teacher = self.warmup_temperature_teacher + progress * (
                self.temperature_teacher - self.warmup_temperature_teacher
            )
        else:
            temperature_teacher = self.temperature_teacher
    else:
        temperature_teacher = getattr(self, "temperature_teacher", 0.07)

    # --- Accumulation-aware centering ---
    accum = getattr(self.trainer, "accumulate_grad_batches_",
                    getattr(self.trainer, "accumulate_grad_batches", 1))
    accum = max(int(accum), 1)

    # Apply the accumulated center from PREVIOUS effective batch (on first micro-batch)
    if not hasattr(self, "_dino_accum_count"):
        self._dino_accum_count = 0
        self._dino_accum_center = None

    if self._dino_accum_count == 0:
        # First micro-batch of this effective batch: apply any pending accumulated
        # center update from the previous effective batch
        self.dino_loss.apply_center_update()

    # Compute teacher probs using the current (stable) center — do NOT
    # call apply_center_update again within this effective batch
    center = self.dino_loss.center
    if center is not None:
        teacher_probs = F.softmax(
            (teacher_logits - center) / temperature_teacher, dim=-1
        )
    else:
        teacher_probs = F.softmax(teacher_logits / temperature_teacher, dim=-1)

    # Compute loss
    loss = self.dino_loss(student_logits, teacher_probs)

    # Accumulate center contribution from this micro-batch
    with torch.no_grad():
        # Mean over batch dim, sum over views → [1, out_dim]
        micro_center = torch.sum(teacher_logits.mean(1), dim=0, keepdim=True)
        n_views = len(teacher_logits)

        if self._dino_accum_center is None:
            self._dino_accum_center = micro_center.clone()
            self._dino_accum_n_views = n_views
        else:
            self._dino_accum_center += micro_center

        self._dino_accum_count += 1

        if self._dino_accum_count >= accum:
            # Last micro-batch: queue ONE center update for the full effective batch.
            # Mimic update_center but with accumulated values.
            self.dino_loss.updated = False
            self.dino_loss.len_teacher_output = n_views  # same across micro-batches
            # Average across accumulation steps
            self.dino_loss.async_batch_center = self._dino_accum_center / accum

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.dino_loss.reduce_handle = torch.distributed.all_reduce(
                    self.dino_loss.async_batch_center, async_op=True
                )
            else:
                self.dino_loss.reduce_handle = None

            # Reset accumulation state
            self._dino_accum_count = 0
            self._dino_accum_center = None

    out["embedding"] = teacher_cls_features.detach()
    out["loss"] = loss
    self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)
    return out


def _lejepa_forward(self, batch, stage):
    """LeJEPA forward: multi-view invariance + SIGReg regularization."""
    out = {}
    views = batch.get("views")

    if views is not None:
        V, N = len(views), views[0]["image"].size(0)
        all_images = torch.cat([v["image"] for v in views], dim=0)
        all_emb = self.backbone(all_images)
        # Only pass first view's embedding to probes (avoid noisy augmented views)
        out["embedding"] = all_emb[:N]

        if "label" in views[0]:
            out["label"] = views[0]["label"]

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
            self.log(f"{stage}/loss", lejepa_loss, on_step=True, on_epoch=True, sync_dist=True)
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
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim


def build_barlow_twins(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(
        embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim
    )
    barlow_loss = spt.losses.BarlowTwinsLoss(lambd=cfg.model.loss.lambda_coeff)
    # Replace LazyBatchNorm1d with concrete BatchNorm1d to avoid
    # UninitializedParameter errors (e.g. numel() calls before first forward)
    barlow_loss.bn = nn.BatchNorm1d(cfg.model.projector.output_dim)
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.barlow_twins_forward,
        barlow_loss=barlow_loss,
        optim=build_optim_config(cfg.model, cfg.backbone),
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
        optim=build_optim_config(cfg.model, cfg.backbone),
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
        optim=build_optim_config(cfg.model, cfg.backbone),
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
        forward=_dino_forward,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim


def build_mae(cfg, ds_config: DatasetConfig) -> tuple[spt.Module, int]:
    h, w = ds_config.image_size
    patch_size = cfg.backbone.patch_size
    decoder_embed_dim = cfg.model.decoder.embed_dim
    decoder_depth = cfg.model.decoder.depth
    mask_ratio = cfg.model.mask_ratio

    grid_size = (h // patch_size, w // patch_size)
    num_patches = grid_size[0] * grid_size[1]
    output_dim = patch_size * patch_size * 3

    vit_model = create_vit(
        size=cfg.backbone.size,
        img_size=(h, w),
        patch_size=patch_size,
    )
    log.info(
        f"MAE config: patch_size={patch_size}, grid={grid_size}, "
        f"num_patches={num_patches}, mask_ratio={mask_ratio:.2f}, "
        f"visible={int(num_patches * (1 - mask_ratio))}"
    )

    masking = spt.backbone.PatchMasking(mask_ratio=mask_ratio)
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

    norm_pix_loss = cfg.model.get("norm_pix_loss", True)
    module = spt.Module(
        backbone=encoder,
        decoder=decoder,
        forward=_mae_forward,
        patch_size=patch_size,
        norm_pix_loss=norm_pix_loss,
        optim=build_optim_config(cfg.model, cfg.backbone),
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
