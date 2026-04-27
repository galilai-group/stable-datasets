"""DINO: self-distillation with multi-crop (2 global + 6 local views)."""

from __future__ import annotations

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_by_prefix
from torch import nn

from benchmarks.models import (
    build_optim_config,
    create_backbone,
    collate_multicrop,
    get_embedding_dim,
    ssl_augmentation,
    val_transform,
)


def create_transforms(ds_config, model_cfg=None):
    """Returns (train_transform, val_transform, collate_fn).

    Builds DINO's asymmetric multi-crop:
      - global_1: blur p=1.0, no solarize
      - global_2: blur p=0.1, solarize p=0.2
      - local_*:  smaller crop, blur p=0.5
    All resizes use BICUBIC to match the canonical DINO recipe.
    Crop counts are read from ``model_cfg.transforms.{num_global,num_local}``.
    """
    h, w = ds_config.image_size

    tcfg = getattr(model_cfg, "transforms", None) if model_cfg is not None else None
    num_global = int(getattr(tcfg, "num_global", 2)) if tcfg is not None else 2
    num_local = int(getattr(tcfg, "num_local", 6)) if tcfg is not None else 6

    bicubic = transforms.InterpolationMode.BICUBIC
    global_aug_1 = ssl_augmentation(
        ds_config, (h, w), crop_scale=(0.4, 1.0), blur_p=1.0, solarize_p=0.0, interpolation=bicubic
    )
    global_aug_2 = ssl_augmentation(
        ds_config, (h, w), crop_scale=(0.4, 1.0), blur_p=0.1, solarize_p=0.2, interpolation=bicubic
    )
    globals_ = [global_aug_1, global_aug_2]

    transform_dict = {}
    for i in range(num_global):
        # If num_global > 2 (non-canonical), cycle through the asymmetric pair.
        transform_dict[f"global_{i + 1}"] = globals_[i % 2]

    local_size = (max(h // 2, 32), max(w // 2, 32))
    local_aug = ssl_augmentation(
        ds_config, local_size, crop_scale=(0.05, 0.4), blur_p=0.5, interpolation=bicubic
    )
    for i in range(num_local):
        transform_dict[f"local_{i + 1}"] = local_aug

    train = transforms.MultiViewTransform(transform_dict)
    return train, val_transform(ds_config), collate_multicrop


# Forward


def forward(self, batch, stage):
    """DINO forward with accumulation-aware centering.

    When using gradient accumulation (batch_size=32, accum=8 for effective 256),
    the vanilla DINO center update applies one EMA step per micro-batch. This
    causes the center to update 8x per effective batch with 8x noisier estimates,
    destabilizing the center and leading to mode collapse.

    Fix: accumulate center contributions across micro-batches, apply ONE EMA
    update per effective batch.
    """
    out = {}

    global_views, local_views, all_views = _get_views_by_prefix(batch, global_prefix="global", local_prefix="local")

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
            teacher_cls_features = teacher_features[:, 0, :] if teacher_features.ndim == 3 else teacher_features
        teacher_logits = self.projector.forward_teacher(teacher_cls_features)
        teacher_logits = teacher_logits.view(n_global, batch_size, -1)

    # Student: all views
    student_logits_list = []
    student_features = self.backbone.forward_student(global_images)
    if hasattr(student_features, "last_hidden_state"):
        student_cls_features = student_features.last_hidden_state[:, 0, :]
    else:
        student_cls_features = student_features[:, 0, :] if student_features.ndim == 3 else student_features
    student_global_logits = self.projector.forward_student(student_cls_features)
    student_global_logits = student_global_logits.view(n_global, batch_size, -1)
    student_logits_list.append(student_global_logits)

    if n_local > 0:
        local_images = torch.cat([view["image"] for view in local_views], dim=0)
        student_features = self.backbone.forward_student(local_images, interpolate_pos_encoding=True)
        if hasattr(student_features, "last_hidden_state"):
            student_local_cls = student_features.last_hidden_state[:, 0, :]
        else:
            student_local_cls = student_features[:, 0, :] if student_features.ndim == 3 else student_features
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

    accum = getattr(self.trainer, "accumulate_grad_batches_", getattr(self.trainer, "accumulate_grad_batches", 1))
    accum = max(int(accum), 1)

    if getattr(self, "use_sinkhorn_knopp", False):
        # Sinkhorn-Knopp: optimal-transport target with uniform prototype mass.
        # Batch-size-invariant — no centering EMA, no accumulation bookkeeping.
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        n_views = teacher_logits.shape[0]
        num_samples = n_views * batch_size * world_size
        teacher_probs = self.dino_loss.sinkhorn_knopp_teacher(
            teacher_logits, teacher_temp=temperature_teacher, num_samples=num_samples
        )
        loss = self.dino_loss(student_logits, teacher_probs)
    else:
        # --- Accumulation-aware centering ---
        if not hasattr(self, "_dino_accum_count"):
            self._dino_accum_count = 0
            self._dino_accum_center = None

        if self._dino_accum_count == 0:
            self.dino_loss.apply_center_update()

        center = self.dino_loss.center
        if center is not None:
            teacher_probs = F.softmax((teacher_logits - center) / temperature_teacher, dim=-1)
        else:
            teacher_probs = F.softmax(teacher_logits / temperature_teacher, dim=-1)

        loss = self.dino_loss(student_logits, teacher_probs)

        with torch.no_grad():
            micro_center = torch.sum(teacher_logits.mean(1), dim=0, keepdim=True)
            n_views = len(teacher_logits)

            if self._dino_accum_center is None:
                self._dino_accum_center = micro_center.clone()
                self._dino_accum_n_views = n_views
            else:
                self._dino_accum_center += micro_center

            self._dino_accum_count += 1

            if self._dino_accum_count >= accum:
                self.dino_loss.updated = False
                self.dino_loss.len_teacher_output = n_views
                self.dino_loss.async_batch_center = self._dino_accum_center / accum

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    self.dino_loss.reduce_handle = torch.distributed.all_reduce(
                        self.dino_loss.async_batch_center, async_op=True
                    )
                else:
                    self.dino_loss.reduce_handle = None

                self._dino_accum_count = 0
                self._dino_accum_center = None

    out["embedding"] = teacher_cls_features.detach()
    # spt's manual_backward does not divide by accumulate_grad_batches, so the
    # accumulated gradient at the optimizer step is accum× the single-batch
    # mean-gradient — effective LR scales with accum. Divide here so bs=B/accum=K
    # is equivalent to bs=K*B/accum=1 as expected.
    out["loss"] = loss / accum
    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


# Builder


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    h = ds_config.image_size[0]
    backbone = create_backbone(cfg.backbone, ds_config)
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
        center_momentum=cfg.model.get("center_momentum", 0.9),
    )

    module = spt.Module(
        backbone=backbone_wrapper,
        projector=projector_wrapper,
        dino_loss=dino_loss,
        temperature_teacher=cfg.model.loss.temperature_teacher,
        warmup_temperature_teacher=cfg.model.loss.warmup_temperature_teacher,
        warmup_epochs_temperature_teacher=cfg.model.loss.warmup_epochs_temperature_teacher,
        use_sinkhorn_knopp=bool(cfg.model.get("sinkhorn_knopp", False)),
        forward=forward,
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
