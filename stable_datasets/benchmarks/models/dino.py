"""DINO: self-distillation with multi-crop (2 global + 6 local views)."""

from __future__ import annotations

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from stable_pretraining.data import transforms
from stable_pretraining.forward import _get_views_by_prefix
from torch import nn

from stable_datasets.benchmarks.models import (
    build_optim_config, collate_multicrop, get_embedding_dim,
    ssl_augmentation, val_transform,
)

NUM_GLOBAL = 2
NUM_LOCAL = 6


def create_transforms(ds_config):
    """Returns (train_transform, val_transform, collate_fn)."""
    h, w = ds_config.image_size
    global_aug = ssl_augmentation(ds_config, (h, w), crop_scale=(0.4, 1.0), blur_p=1.0)

    transform_dict = {}
    for i in range(NUM_GLOBAL):
        transform_dict[f"global_{i + 1}"] = global_aug

    local_size = (max(h // 2, 32), max(w // 2, 32))
    local_aug = ssl_augmentation(ds_config, local_size, crop_scale=(0.05, 0.4), blur_p=0.5)
    for i in range(NUM_LOCAL):
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

    if not hasattr(self, "_dino_accum_count"):
        self._dino_accum_count = 0
        self._dino_accum_center = None

    if self._dino_accum_count == 0:
        self.dino_loss.apply_center_update()

    center = self.dino_loss.center
    if center is not None:
        teacher_probs = F.softmax(
            (teacher_logits - center) / temperature_teacher, dim=-1
        )
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
    out["loss"] = loss
    self.log(f"{stage}/loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True)
    return out


# Builder


def build(cfg, ds_config) -> tuple[spt.Module, int]:
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
        forward=forward,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim
