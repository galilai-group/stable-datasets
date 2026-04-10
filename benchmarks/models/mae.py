"""MAE (Masked Autoencoder): single-view masked patch reconstruction."""

from __future__ import annotations

import logging

import stable_pretraining as spt
from stable_pretraining.data import transforms

from stable_datasets.benchmarks.models import build_optim_config, collate_single, val_transform
from stable_datasets.benchmarks.models.vit import create_vit


log = logging.getLogger(__name__)


def create_transforms(ds_config):
    """Returns (train_transform, val_transform, collate_fn).

    MAE uses minimal augmentation (no color jitter, no blur) — just
    random crop + flip so the masking task carries the learning signal.
    """
    h, w = ds_config.image_size
    stats = {"mean": ds_config.mean, "std": ds_config.std}
    train = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((h, w), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**stats),
    )
    return train, val_transform(ds_config), collate_single


# Forward


def _patchify(images, patch_size):
    """Convert images to patches [N, T, P*C]."""
    N, C, H, W = images.shape
    p = patch_size
    h, w = H // p, W // p
    x = images.reshape(N, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(N, h * w, p * p * C)


def forward(self, batch, stage):
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
        target = _patchify(images, self.patch_size)
        norm_pix = getattr(self, "norm_pix_loss", True)
        loss = spt.losses.mae(target, pred, encoder_out.mask, norm_pix_loss=norm_pix)
        out["loss"] = loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    return out


# Builder


def build(cfg, ds_config) -> tuple[spt.Module, int]:
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
        forward=forward,
        patch_size=patch_size,
        norm_pix_loss=norm_pix_loss,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, encoder_embed_dim
