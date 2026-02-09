"""Experiment factory for SSL baselines.

This module provides utilities to create SSL experiments (SimCLR, MAE)
using stable-pretraining's Module and Manager.
"""

from __future__ import annotations

import lightning as pl
import stable_pretraining as spt
import torch
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from stable_pretraining import forward
from torch import nn

from .dataset_factory import DatasetConfig
from .models import create_vit_base
from .lejepa_losses import EppsPulley, SlicingUnivariateTest


DEFAULT_EMBED_DIM = 512


def get_embedding_dim(backbone: nn.Module) -> int:
    """Get the embedding dimension from a backbone."""
    for attr in ("num_features", "embed_dim"):
        if hasattr(backbone, attr):
            return getattr(backbone, attr)

    if hasattr(backbone, "fc") and hasattr(backbone.fc, "in_features"):
        return backbone.fc.in_features

    return DEFAULT_EMBED_DIM


def _create_projection_head(embed_dim: int, hidden_dim: int, projection_dim: int) -> nn.Module:
    """Create a 3-layer MLP projection head with BatchNorm."""
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, projection_dim),
    )


def create_simclr_module(
    config: DatasetConfig,
    backbone_name: str = "resnet18",
    projection_dim: int = 256,
    hidden_dim: int = 2048,
    temperature: float = 0.5,
    learning_rate: float = 5.0,
    weight_decay: float = 1e-6,
) -> spt.Module:
    """Create a SimCLR module.

    Args:
        config: Dataset configuration
        backbone_name: Name of the backbone (e.g., "resnet18", "resnet50")
        projection_dim: Output dimension of the projection head
        hidden_dim: Hidden dimension of the projection head
        temperature: Temperature for NT-Xent loss
        learning_rate: Learning rate (LARS optimizer)
        weight_decay: Weight decay

    Returns:
        spt.Module configured for SimCLR training
    """
    backbone = spt.backbone.from_torchvision(backbone_name, low_resolution=config.low_resolution)
    backbone.fc = nn.Identity()
    embed_dim = get_embedding_dim(backbone)

    projector = _create_projection_head(embed_dim, hidden_dim, projection_dim)

    return spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr_forward,
        simclr_loss=spt.losses.NTXEntLoss(temperature=temperature),
        optim={
            "optimizer": {
                "type": "LARS",
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
            },
            "interval": "epoch",
        },
    )


def _adjust_mae_params_for_low_resolution(
    config: DatasetConfig,
    patch_size: int,
    embed_dim: int,
    depth: int,
    decoder_embed_dim: int,
    decoder_depth: int,
) -> tuple[int, int, int, int, int]:
    """Adjust MAE parameters for low resolution images."""
    h, w = config.image_size
    if not config.low_resolution:
        return patch_size, embed_dim, depth, decoder_embed_dim, decoder_depth

    return (
        min(patch_size, min(h, w) // 4),
        min(embed_dim, 384),
        min(depth, 6),
        min(decoder_embed_dim, 256),
        min(decoder_depth, 4),
    )


def _mae_patchify(images, patch_size):
    """Convert images to patches in MAE format [N, T, P*C]."""
    N, C, H, W = images.shape
    p = patch_size
    h, w = H // p, W // p
    # [N, C, H, W] -> [N, C, h, p, w, p] -> [N, h, w, p, p, C] -> [N, T, P*C]
    x = images.reshape(N, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # [N, h, w, p, p, C]
    x = x.reshape(N, h * w, p * p * C)
    return x


def _mae_forward(self, batch, stage):
    """MAE forward function for training and inference."""
    images = batch["image"]
    encoder_out = self.backbone(images)

    # Extract CLS token embedding for downstream tasks
    out = {"embedding": encoder_out.encoded[:, 0]}

    if "label" in batch:
        out["label"] = batch["label"]

    if self.training:
        # Strip prefix tokens (CLS, etc.) - decoder expects only visible patch tokens
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        visible_patches = encoder_out.encoded[:, num_prefix:]

        pred = self.decoder(visible_patches, encoder_out.mask)
        target = _mae_patchify(images, self.patch_size)
        loss = spt.losses.mae(target, pred, encoder_out.mask)

        out["loss"] = loss
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    return out


def _lejepa_forward(self, batch, stage):
    """LeJEPA forward pass with multi-view invariance and SIGReg losses."""
    out = {}
    views = batch.get("views")

    if views is not None:
        # Multi-view training
        V = len(views)
        N = views[0]["image"].size(0)

        # Single forward pass for all views
        all_images = torch.cat([view["image"] for view in views], dim=0)
        all_emb = self.backbone(all_images)
        out["embedding"] = all_emb

        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            all_proj = self.projector(all_emb)
            proj_stacked = all_proj.reshape(V, N, -1)  # (V, N, proj_dim)

            # Invariance loss: compare each view to mean
            view_mean = proj_stacked.mean(0)
            inv_loss = (view_mean - proj_stacked).square().mean()

            # SIGReg loss: check if loss supports 3D input (V, N, D)
            if isinstance(self.sigreg_loss, SlicingUnivariateTest) and isinstance(
                self.sigreg_loss.univariate_test, EppsPulley
            ):
                sigreg_loss = self.sigreg_loss(proj_stacked)
            else:
                sigreg_loss = self.sigreg_loss(proj_stacked.reshape(-1, proj_stacked.size(-1)))

            lamb = getattr(self, "lamb", 0.02)
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            out["loss"] = lejepa_loss

            self.log(f"{stage}/sigreg", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv", inv_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/lejepa", lejepa_loss, on_step=True, on_epoch=True, sync_dist=True)
    else:
        # Single-view (validation)
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def create_mae_module(
    config: DatasetConfig,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    decoder_embed_dim: int = 512,
    decoder_depth: int = 8,
    decoder_num_heads: int = 16,
    mask_ratio: float = 0.75,
    learning_rate: float = 1.5e-4,
    weight_decay: float = 0.05,
) -> spt.Module:
    """Create an MAE (Masked Autoencoder) module.

    Args:
        config: Dataset configuration
        patch_size: Size of image patches
        embed_dim: Encoder embedding dimension
        depth: Encoder depth
        num_heads: Encoder attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Decoder depth
        decoder_num_heads: Decoder attention heads
        mask_ratio: Ratio of patches to mask
        learning_rate: Learning rate
        weight_decay: Weight decay

    Returns:
        spt.Module configured for MAE training
    """
    h, w = config.image_size

    patch_size, _, depth, decoder_embed_dim, decoder_depth = _adjust_mae_params_for_low_resolution(
        config, patch_size, embed_dim, depth, decoder_embed_dim, decoder_depth
    )

    grid_size = (h // patch_size, w // patch_size)
    num_patches = grid_size[0] * grid_size[1]
    # Always 3 channels because transforms.RGB() converts all images to RGB
    output_dim = patch_size * patch_size * 3

    masking = spt.backbone.PatchMasking(mask_ratio=mask_ratio)
    vit_model = create_vit_base(patch_size=patch_size, img_size=(h, w), pretrained=False)
    encoder = spt.backbone.MaskedEncoder(
        model_or_model_name=vit_model,
        masking=masking,
    )

    # Get actual embed_dim from the encoder (may differ from requested due to model config)
    encoder_embed_dim = getattr(encoder, "embed_dim", embed_dim)

    decoder = spt.backbone.MAEDecoder(
        embed_dim=encoder_embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        output_dim=output_dim,
        num_patches=num_patches,
        grid_size=grid_size,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
    )

    return spt.Module(
        backbone=encoder,
        decoder=decoder,
        forward=_mae_forward,
        patch_size=patch_size,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": (0.9, 0.95),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                # peak_step as fraction of total_steps (default 0.01 = 1% warmup)
            },
            "interval": "epoch",
        },
    )


def create_lejepa_module(
    config: DatasetConfig,
    backbone_name: str = "resnet18",
    projection_dim: int = 128,
    hidden_dim: int = 2048,
    lamb: float = 0.02,
    t_max: float = 3.0,
    n_points: int = 17,
    num_slices: int = 256,
    learning_rate: float = 5.0,
    weight_decay: float = 1e-6,
) -> spt.Module:
    """Create a LeJEPA module.

    Args:
        config: Dataset configuration
        backbone_name: Name of the backbone (e.g., "resnet18", "resnet50")
        projection_dim: Output dimension of the projection head
        hidden_dim: Hidden dimension of the projection head
        lamb: Regularization weight (SIGReg vs invariance loss)
        t_max: Maximum integration point for EppsPulley test
        n_points: Number of integration points for EppsPulley test
        num_slices: Number of random slices for SlicingUnivariateTest
        learning_rate: Learning rate (LARS optimizer)
        weight_decay: Weight decay

    Returns:
        spt.Module configured for LeJEPA training
    """
    backbone = spt.backbone.from_torchvision(backbone_name, low_resolution=config.low_resolution)
    backbone.fc = nn.Identity()
    embed_dim = get_embedding_dim(backbone)

    projector = _create_projection_head(embed_dim, hidden_dim, projection_dim)

    # Create SIGReg loss: SlicingUnivariateTest with EppsPulley
    univariate_test = EppsPulley(t_max=t_max, n_points=n_points)
    sigreg_loss = SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=num_slices,
        reduction="mean",
    )

    return spt.Module(
        backbone=backbone,
        projector=projector,
        sigreg_loss=sigreg_loss,
        lamb=lamb,
        forward=_lejepa_forward,
        optim={
            "optimizer": {
                "type": "LARS",
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
            },
            "interval": "epoch",
        },
    )


def create_dino_module(
    config: DatasetConfig,
    backbone_name: str = "vit_tiny_patch16_224",
    projection_dim: int = 65536,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
    temperature_student: float = 0.1,
    temperature_teacher: float = 0.04,
    warmup_temperature_teacher: float = 0.04,
    warmup_epochs_temperature_teacher: int = 30,
    momentum_teacher: float = 0.996,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.04,
) -> spt.Module:
    """Create a DINO module.

    Args:
        config: Dataset configuration
        backbone_name: Name of the backbone (ViT model)
        projection_dim: Output dimension of the projection head (prototypes)
        hidden_dim: Hidden dimension of the projection head
        bottleneck_dim: Bottleneck dimension before final layer
        temperature_student: Temperature for student softmax
        temperature_teacher: Final temperature for teacher softmax
        warmup_temperature_teacher: Starting temperature for teacher
        warmup_epochs_temperature_teacher: Epochs to warm up teacher temperature
        momentum_teacher: EMA momentum for teacher network
        learning_rate: Learning rate
        weight_decay: Weight decay

    Returns:
        spt.Module configured for DINO training
    """
    # Use ViT backbone (DINO works best with ViTs)
    h, w = config.image_size
    backbone = spt.backbone.vit_timm(backbone_name, img_size=(h, w))
    embed_dim = get_embedding_dim(backbone)

    # Wrap backbone in TeacherStudentWrapper
    backbone_wrapper = spt.backbone.TeacherStudentWrapper(
        student=backbone, momentum=momentum_teacher
    )

    # Create DINO projection head (3-layer MLP + normalization + prototypes)
    projector_head = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, bottleneck_dim),
    )

    # Wrap projector with normalization and prototypes
    projector = spt.backbone.DINOProjector(
        projector=projector_head,
        input_dim=bottleneck_dim,
        output_dim=projection_dim,
        norm_last_layer=True,
    )

    # Wrap in TeacherStudentWrapper
    projector_wrapper = spt.backbone.TeacherStudentWrapper(
        student=projector, momentum=momentum_teacher
    )

    # Create DINO loss
    dino_loss = spt.losses.DINOv1Loss(
        temperature_student=temperature_student,
        center_momentum=0.9,
    )

    return spt.Module(
        backbone=backbone_wrapper,
        projector=projector_wrapper,
        dino_loss=dino_loss,
        temperature_teacher=temperature_teacher,
        warmup_temperature_teacher=warmup_temperature_teacher,
        warmup_epochs_temperature_teacher=warmup_epochs_temperature_teacher,
        forward=forward.dino_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
            },
            "interval": "epoch",
        },
    )


def create_evaluation_callbacks(
    module: spt.Module,
    config: DatasetConfig,
    embed_dim: int,
) -> list:
    """Create evaluation callbacks (linear probe, KNN probe)."""
    num_classes = config.num_classes

    evals, metrics = [], {}

    if num_classes > 0:
        metrics = {
            "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
            "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=min(5, num_classes)),
        }

    if num_classes > 0:
        linear_probe = spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(embed_dim, num_classes),
            loss_fn=nn.CrossEntropyLoss(),
            metrics=metrics,
        )
        evals.append(linear_probe)

    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics=metrics,
        input_dim=embed_dim,
        k=10,
    )

    evals.append(knn_probe)
    return evals


AVAILABLE_MODELS = ("simclr", "mae", "lejepa", "dino")
DEFAULT_MAE_EMBED_DIM = 768


def _create_module_and_get_embed_dim(model_name: str, config: DatasetConfig, **model_kwargs) -> tuple[spt.Module, int]:
    """Create the SSL module and return it along with its embedding dimension."""
    if model_name == "simclr":
        module = create_simclr_module(config, **model_kwargs)
        embed_dim = get_embedding_dim(module.backbone)
        return module, embed_dim

    if model_name == "mae":
        module = create_mae_module(config, **model_kwargs)
        embed_dim = getattr(module.backbone, "embed_dim", DEFAULT_MAE_EMBED_DIM)
        return module, embed_dim

    if model_name == "lejepa":
        module = create_lejepa_module(config, **model_kwargs)
        embed_dim = get_embedding_dim(module.backbone)
        return module, embed_dim

    if model_name == "dino":
        module = create_dino_module(config, **model_kwargs)
        # DINO uses TeacherStudentWrapper, get embed_dim from teacher
        backbone = getattr(module.backbone, "teacher", module.backbone)
        embed_dim = get_embedding_dim(backbone)
        return module, embed_dim

    available = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unknown model: {model_name}. Available: {available}")


def create_experiment(
    model_name: str,
    data_module: spt.data.DataModule,
    config: DatasetConfig,
    max_epochs: int = 100,
    precision: str = "16-mixed",
    **model_kwargs,
) -> tuple[spt.Manager, dict]:
    """Create a complete experiment (module + trainer + callbacks).

    Args:
        model_name: Name of the SSL method ("simclr", "mae", "lejepa", or "dino")
        data_module: Data module with train/val dataloaders
        config: Dataset configuration
        max_epochs: Maximum training epochs
        precision: Training precision
        **model_kwargs: Additional kwargs passed to the model factory

    Returns:
        Tuple of (Manager, experiment_info dict)
    """
    model_name_lower = model_name.lower()
    module, embed_dim = _create_module_and_get_embed_dim(model_name_lower, config, **model_kwargs)

    callbacks = create_evaluation_callbacks(module, config, embed_dim)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Disable validation if no validation dataloader exists
    has_val = data_module.val is not None
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if has_val else 0,
        callbacks=callbacks,
        precision=precision,
        enable_checkpointing=False,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data_module)

    experiment_info = {
        "model": model_name_lower,
        "dataset": config.name,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "max_epochs": max_epochs,
    }

    return manager, experiment_info
