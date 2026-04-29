"""LeJEPA: invariance + sliced Epps-Pulley regularization (multi-crop)."""

from __future__ import annotations

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput

from benchmarks.models import (
    build_optim_config,
    collate_multicrop,
    resolve_backbone_name,
    val_transform,
)


NUM_GLOBAL = 2
NUM_LOCAL = 6


def _photometric(ds_config) -> list:
    ops = [transforms.RandomHorizontalFlip(p=0.5)]
    if ds_config.channels != 1:
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
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


def create_transforms(ds_config, model_cfg=None):
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


def forward(self, batch, stage):
    """LeJEPA forward: multi-view invariance + Epps-Pulley goodness-of-fit (SIGReg).

    Expects ``self`` to have attributes:
        - ``backbone``: Feature extraction network
        - ``projector``: Projection head
        - ``sigreg``: :class:`SlicedEppsPulley` module
        - ``lamb``: SIGReg weight λ

    Batch format:
        - Training: dict of named views (``"global_0"``, ``"local_2"``, etc.)
        - Eval: single dict with ``"image"`` key

    Args:
        self: Module instance (automatically bound).
        batch: Named view dict or single-image dict.
        stage: Training stage ('train', 'val', or 'test').

    Returns:
        Dictionary with ``"loss"``, ``"embedding"``, and optionally ``"label"``.
    """
    out = {}

    images = batch.get("image")
    if stage == "fit":
        global_views = [batch[key]["image"] for key in batch if key.startswith("global")]
        local_views = [batch[key]["image"] for key in batch if key.startswith("local")]
        labels = next(batch[key]["label"] for key in batch if key.startswith("global") or key.startswith("local"))

        output: LeJEPAOutput = self.model.forward(global_views=global_views, local_views=local_views, images=images)
        out["label"] = labels.repeat(len(global_views))
    else:
        output: LeJEPAOutput = self.model.forward(images=images)
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(
        f"{stage}/sigreg",
        output.sigreg_loss,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    backbone_name = resolve_backbone_name(cfg.backbone, ds_config)
    lejepa = LeJEPA(
        encoder_name=backbone_name,
        n_slices=cfg.model.loss.num_slices,
        t_max=cfg.model.loss.t_max,
        n_points=cfg.model.loss.n_points,
        lamb=cfg.model.loss.lamb,
        pretrained=False,
        drop_path_rate=0.0,
    )
    embed_dim = lejepa.embed_dim
    module = spt.Module(
        model=lejepa,
        backbone=lejepa.backbone,
        projector=lejepa.projector,
        forward=forward,
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
