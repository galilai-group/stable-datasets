"""SSL model builders for benchmarks.

Each model file (simclr.py, dino.py, etc.) exposes:
  - ``build(cfg, ds_config)`` → ``(spt.Module, embed_dim)``
  - ``create_transforms(ds_config)`` → ``(train_transform, val_transform, collate_fn)``

Shared helpers live here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import stable_pretraining as spt
import torch
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from stable_pretraining.callbacks.lidar import LiDAR
from stable_pretraining.callbacks.rankme import RankMe
from stable_pretraining.data import transforms
from torch import nn

from benchmarks.models.vit import create_vit


log = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 512


# Backbone


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


def _cfg_value(cfg, key, default=None):
    """Read a config/object attribute without assuming a concrete config type."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_backbone_name(backbone_cfg, ds_config=None) -> str:
    """Resolve a backbone config/object into a concrete timm model name."""
    if isinstance(backbone_cfg, str):
        return backbone_cfg

    name = _cfg_value(backbone_cfg, "name")
    if isinstance(name, str):
        if "_patch" in name:
            return name
        patch_size = _cfg_value(backbone_cfg, "patch_size")
        image_size = None
        if ds_config is not None:
            image_size = ds_config.image_size[0]
        if patch_size is not None and image_size is not None:
            return f"{name}_patch{patch_size}_{image_size}"
        return name

    backbone_type = _cfg_value(backbone_cfg, "type")
    size = _cfg_value(backbone_cfg, "size")
    patch_size = _cfg_value(backbone_cfg, "patch_size")
    if backbone_type == "vit" and size and patch_size:
        image_size = 224 if ds_config is None else ds_config.image_size[0]
        return f"vit_{size}_patch{patch_size}_{image_size}"

    raise TypeError(f"Unsupported backbone config: {backbone_cfg!r}")


def create_backbone(
    backbone_cfg,
    ds_config,
    patch_size: int | None = None,
    img_size: int | tuple[int, int] | None = None,
) -> nn.Module:
    """Create a backbone from either a timm model name or a structured config.

    Always builds a 3-channel patch embed: the SSL transform pipeline starts
    with ``transforms.RGB()`` which upcasts grayscale source images to 3
    channels before they reach the model.

    ``patch_size`` and ``img_size`` override the defaults derived from the
    backbone name / ds_config when a dataset needs a non-standard ViT config
    (e.g. small native resolution with a finer patch grid).
    """
    name = resolve_backbone_name(backbone_cfg, ds_config)
    effective_img_size = img_size if img_size is not None else ds_config.image_size
    return create_vit(name, img_size=effective_img_size, in_chans=3, patch_size=patch_size)


# Projector


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


# Optimizer config


def build_optim_config(model_cfg) -> dict:
    """Build optimizer config dict from model config."""
    if hasattr(model_cfg, "vit_optimizer"):
        opt_cfg = model_cfg.vit_optimizer
    else:
        opt_cfg = model_cfg.optimizer

    scheduler_cfg = {"type": model_cfg.scheduler.type}
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
    if hasattr(model_cfg, "_lr_override"):
        optim["optimizer"]["lr"] = model_cfg._lr_override
    return optim


# Evaluation callbacks


def create_eval_callbacks(module: spt.Module, ds_config, embed_dim: int) -> list:
    """Create linear probe and KNN evaluation callbacks."""
    num_classes = ds_config.num_classes
    callbacks = []

    if num_classes > 0:
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=min(5, num_classes)),
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
                "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=min(5, num_classes)),
            },
            input_dim=embed_dim,
            k=10,
        )
        callbacks.append(knn_probe)

    # Representation-geometry monitors. RankMe is label-free; LiDAR uses
    # sequential surrogate classes from the queue (paper-native mode).
    callbacks.append(
        RankMe(
            name="rankme",
            target="embedding",
            queue_length=2048,
            target_shape=embed_dim,
        )
    )
    if num_classes > 0:
        lidar_n_classes = min(100, num_classes)
        callbacks.append(
            LiDAR(
                name="lidar",
                target="embedding",
                queue_length=max(2048, lidar_n_classes * 10),
                target_shape=embed_dim,
                n_classes=lidar_n_classes,
                samples_per_class=10,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


# Shared transforms & collation


def val_transform(ds_config) -> transforms.Compose:
    """Validation transform shared by all models: resize + normalize."""
    h, w = ds_config.image_size
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((h, w)),
        transforms.ToImage(mean=ds_config.mean, std=ds_config.std),
    )


def ssl_augmentation(
    ds_config,
    crop_size: tuple[int, int],
    crop_scale: tuple[float, float],
    blur_p: float = 0.5,
    solarize_p: float = 0.0,
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
) -> transforms.Compose:
    """Standard SSL augmentation pipeline (crop → flip → color → blur → solarize → normalize).

    Shared by multiview and multicrop models.  Only the crop params, blur
    probability, solarization probability, and resize interpolation differ
    between them. Solarize is gated to RGB inputs.
    """
    ops = [
        transforms.RGB(),
        transforms.RandomResizedCrop(crop_size, scale=crop_scale, interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if ds_config.channels != 1:
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
    ops.append(transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=blur_p))
    if solarize_p > 0.0 and ds_config.channels != 1:
        ops.append(transforms.RandomSolarize(threshold=128, p=solarize_p))
    ops.append(transforms.ToImage(mean=ds_config.mean, std=ds_config.std))
    return transforms.Compose(*ops)


def collate_single(batch):
    """Collate for single-view batches: {"image": tensor, "label": int}."""
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch]),
    }


def collate_multiview(batch):
    """Collate for list-based multi-view: {"views": [view0, view1, ...]}."""
    first = batch[0]
    num_views = len(first["views"])
    views_list = []
    for v in range(num_views):
        view_dict = {"image": torch.stack([s["views"][v]["image"] for s in batch])}
        if "label" in first["views"][v]:
            view_dict["label"] = torch.tensor([s["views"][v]["label"] for s in batch])
        views_list.append(view_dict)
    return {"views": views_list}


def collate_multicrop(batch):
    """Collate for dict-based multi-view (DINO): {"global_1": {...}, ...}."""
    first = batch[0]
    collated = {}
    for view_name in first.keys():
        if isinstance(first[view_name], dict) and "image" in first[view_name]:
            view_dict = {"image": torch.stack([s[view_name]["image"] for s in batch])}
            if "label" in first[view_name]:
                view_dict["label"] = torch.tensor([s[view_name]["label"] for s in batch])
            collated[view_name] = view_dict
    return collated


# Model registry


def _get_model_module(model_name: str):
    """Import and return the model module by name."""
    from benchmarks.models import barlow_twins, dino, lejepa, mae, nnclr, simclr, supervised

    _MODULES = {
        "simclr": simclr,
        "dino": dino,
        "mae": mae,
        "lejepa": lejepa,
        "nnclr": nnclr,
        "barlow_twins": barlow_twins,
        "supervised": supervised,
    }
    if model_name not in _MODULES:
        available = ", ".join(_MODULES.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return _MODULES[model_name]


def build_module(cfg, ds_config) -> tuple[spt.Module, int]:
    """Build a module from Hydra config using the model registry.

    Returns (module, embedding_dim).
    """
    return _get_model_module(cfg.model.name).build(cfg, ds_config)


def get_transforms(model_name: str, ds_config, model_cfg=None):
    """Get (train_transform, val_transform, collate_fn) for a model.

    Each model defines its own ``create_transforms(ds_config, model_cfg=None)``
    that returns the appropriate transforms and collation function. ``model_cfg``
    is the per-model OmegaConf node (``cfg.model``) — models that need to read
    transform-specific knobs (e.g. DINO's ``num_global``/``num_local``) consume it.
    """
    return _get_model_module(model_name).create_transforms(ds_config, model_cfg)
