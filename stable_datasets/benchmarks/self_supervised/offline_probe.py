"""Offline linear probe evaluation for SSL benchmark checkpoints.

Standard protocol: freeze backbone with spt.backbone.EvalOnly, train nn.Linear
for 100 epochs with SGD + cosine annealing using spt.forward.supervised_forward.

Usage:
    # Single checkpoint
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --checkpoint checkpoints/simclr_vit_small_cifar10/last.ckpt

    # Batch: all checkpoints under a directory
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --checkpoint-dir checkpoints/

    # Custom settings
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --checkpoint checkpoints/dino_vit_small_flowers102/last.ckpt \
        --epochs 200 --lr 0.1 --batch-size 256

    # Without wandb
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --checkpoint-dir checkpoints/ --no-wandb

Checkpoint directory convention (from main.py):
    {checkpoint.dir}/{model}_{backbone}_{dataset}/last.ckpt
    e.g. checkpoints/simclr_vit_small_cifar10/last.ckpt

The model, backbone, and dataset are parsed from the directory name.
Results are saved as JSON and logged to wandb.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import torch
import torchmetrics
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchvision.transforms import v2 as transforms

from stable_datasets.benchmarks.self_supervised.dataset import (
    DatasetConfig,
    _get_normalization_stats,
    CACHE_DIR_DEFAULT,
)
from stable_datasets.benchmarks.self_supervised.models import create_vit, VIT_CONFIGS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ============================================================================
# Backbone output normalisation
# ============================================================================


class FlatEmbeddingBackbone(nn.Module):
    """Wraps any backbone to always return flat [B, D] CLS token embeddings.

    Handles:
      - HuggingFace ViT (BaseModelOutputWithPooling) → .last_hidden_state[:, 0]
      - MaskedEncoder (has .encoded) → .encoded[:, 0]
      - timm ViT [B, T, D] → [:, 0]
      - Already flat [B, D] → passthrough
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        out = self.backbone(images)
        if hasattr(out, "last_hidden_state"):  # HF ViT
            return out.last_hidden_state[:, 0]
        if hasattr(out, "encoded"):  # MaskedEncoder
            return out.encoded[:, 0]
        if out.ndim == 3:  # [B, T, D] → CLS token
            return out[:, 0]
        return out  # already [B, D]


# ============================================================================
# Backbone extraction
# ============================================================================


def load_backbone_from_checkpoint(
    ckpt_path: str, model_name: str, backbone_size: str = "small"
) -> tuple[nn.Module, int]:
    """Load a frozen backbone from a Lightning checkpoint.

    Rebuilds the backbone architecture, loads matching weights, and wraps
    with spt.backbone.EvalOnly to freeze it permanently.

    Returns (frozen_backbone, embed_dim).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Collect all backbone.* keys
    backbone_keys = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}

    if not backbone_keys:
        raise ValueError(f"No backbone.* keys found in checkpoint: {ckpt_path}")

    embed_dim = VIT_CONFIGS[backbone_size]["embed_dim"]

    if model_name == "dino":
        # DINO: TeacherStudentWrapper(vit_hf(...)) — extract student
        backbone = spt.backbone.vit_hf(size=backbone_size, pretrained=False)
        prefix = "backbone.student."
        sub_dict = {
            k[len(prefix):]: v for k, v in backbone_keys.items()
            if k.startswith(prefix)
        }
        if not sub_dict:
            prefix = "backbone."
            sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        msg = backbone.load_state_dict(sub_dict, strict=False)
        log.info(f"DINO backbone load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    elif model_name == "mae":
        # MAE: MaskedEncoder wrapping timm ViT
        from stable_pretraining.backbone.vit import MaskedEncoder
        from stable_pretraining.backbone import PatchMasking
        backbone = MaskedEncoder(
            model_or_model_name=f"vit_{backbone_size}_patch16_224",
            masking=PatchMasking(mask_ratio=0.0),
            pretrained=False,
            img_size=224,
        )
        prefix = "backbone."
        sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        msg = backbone.load_state_dict(sub_dict, strict=False)
        log.info(f"MAE backbone load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    elif model_name == "lejepa":
        # LeJEPA: TeacherStudentWrapper(create_vit(...)) — extract student
        backbone = create_vit(size=backbone_size, img_size=(224, 224), patch_size=16)
        prefix = "backbone.student."
        sub_dict = {
            k[len(prefix):]: v for k, v in backbone_keys.items()
            if k.startswith(prefix)
        }
        if not sub_dict:
            prefix = "backbone."
            sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        msg = backbone.load_state_dict(sub_dict, strict=False)
        log.info(f"LeJEPA backbone load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    else:
        # SimCLR, Barlow Twins, VICReg — plain create_vit()
        backbone = create_vit(size=backbone_size, img_size=(224, 224), patch_size=16)
        prefix = "backbone."
        sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        msg = backbone.load_state_dict(sub_dict, strict=False)
        log.info(f"{model_name} backbone load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    if msg.missing_keys:
        log.warning(f"Missing keys (first 5): {msg.missing_keys[:5]}")

    # Normalise output to flat [B, D], then freeze with EvalOnly
    frozen = spt.backbone.EvalOnly(FlatEmbeddingBackbone(backbone))
    return frozen, embed_dim


# ============================================================================
# Dataset loading (simple single-view transforms for linear eval)
# ============================================================================


def create_eval_transforms(mean, std):
    """Standard linear eval transforms: light augmentation for train, deterministic for val."""
    train_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, val_transform


def load_eval_dataset(dataset_name: str, batch_size: int, num_workers: int = 8,
                      cache_dir: str | None = CACHE_DIR_DEFAULT):
    """Load dataset with eval transforms from cached memmaps."""
    import numpy as np
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader

    cache_path = Path(cache_dir) / dataset_name if cache_dir else None
    metadata_path = cache_path / "metadata.json" if cache_path else None

    if not (metadata_path and metadata_path.exists()):
        raise FileNotFoundError(
            f"No cached dataset at {cache_path}. Run cache_datasets.py first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    mean, std = _get_normalization_stats(dataset_name, metadata["original_channels"])
    train_tf, val_tf = create_eval_transforms(mean, std)

    class CachedEvalDataset(Dataset):
        def __init__(self, images_path, labels_path, transform):
            self.images = np.load(images_path, mmap_mode="r")
            self.labels = np.load(labels_path)
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            img = PILImage.fromarray(self.images[idx].copy())
            img = self.transform(img)
            return {"image": img, "label": self.labels[idx]}

    train_ds = CachedEvalDataset(cache_path / "train_images.npy", cache_path / "train_labels.npy", train_tf)
    val_ds = CachedEvalDataset(cache_path / "val_images.npy", cache_path / "val_labels.npy", val_tf)

    ds_config = DatasetConfig(
        name=dataset_name,
        num_classes=metadata["num_classes"],
        channels=metadata["original_channels"],
        image_size=(metadata["image_size"], metadata["image_size"]),
        mean=mean,
        std=std,
        num_frames=1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    return spt.data.DataModule(train=train_loader, val=val_loader), ds_config


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_checkpoint(
    ckpt_path: str,
    model_name: str,
    backbone_size: str,
    dataset_name: str,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 256,
    cache_dir: str | None = CACHE_DIR_DEFAULT,
    use_wandb: bool = True,
    wandb_entity: str | None = None,
    wandb_project: str = "stable-datasets-benchmarks",
    gpu_id: int | None = None,
) -> dict:
    """Run offline linear probe on a single checkpoint."""

    # Pin to a single GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log.info(f"Pinned to GPU {gpu_id}")

    log.info(f"Evaluating: {ckpt_path}")
    log.info(f"  model={model_name}, backbone=vit_{backbone_size}, dataset={dataset_name}")

    # 1. Load frozen backbone
    frozen_backbone, embed_dim = load_backbone_from_checkpoint(ckpt_path, model_name, backbone_size)

    # 2. Load dataset
    data, ds_config = load_eval_dataset(dataset_name, batch_size, cache_dir=cache_dir)

    # 3. Build module — spt.forward.supervised_forward + spt.backbone.EvalOnly
    num_classes = ds_config.num_classes
    classifier = nn.Linear(embed_dim, num_classes)

    ckpt_name = Path(ckpt_path).stem
    run_name = f"probe_{model_name}_vit_{backbone_size}_{dataset_name}_{ckpt_name}"

    module = spt.Module(
        backbone=frozen_backbone,
        classifier=classifier,
        forward=spt.forward.supervised_forward,
        supervised_loss=nn.CrossEntropyLoss(),
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": lr,
                "momentum": 0.9,
                "weight_decay": 0,
            },
            "scheduler": {
                "type": "CosineAnnealingLR",
            },
            "interval": "epoch",
        },
        hparams={
            "probe_type": "offline_linear",
            "ssl_model": model_name,
            "backbone": f"vit_{backbone_size}",
            "dataset": dataset_name,
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_name,
            "probe_epochs": epochs,
            "probe_lr": lr,
            "batch_size": batch_size,
        },
    )

    # 4. Callbacks — lightweight accuracy tracker (not OnlineProbe, which
    #    requires a trainable probe and would fail with nn.Identity())
    class AccuracyTracker(pl.Callback):
        def __init__(self, n_classes):
            super().__init__()
            self.val_top1 = torchmetrics.classification.MulticlassAccuracy(n_classes)
            self.val_top5 = torchmetrics.classification.MulticlassAccuracy(
                n_classes, top_k=min(5, n_classes)
            )

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if "logits" in outputs and "label" in batch:
                logits = outputs["logits"].detach()
                labels = batch["label"]
                self.val_top1.to(logits.device).update(logits, labels)
                self.val_top5.to(logits.device).update(logits, labels)

        def on_validation_epoch_end(self, trainer, pl_module):
            t1 = self.val_top1.compute()
            t5 = self.val_top5.compute()
            pl_module.log("eval/top1", t1, sync_dist=True)
            pl_module.log("eval/top5", t5, sync_dist=True)
            self.val_top1.reset()
            self.val_top5.reset()

    callbacks = [AccuracyTracker(num_classes), LearningRateMonitor(logging_interval="epoch")]

    # 5. Logger
    logger = False
    if use_wandb:
        logger = WandbLogger(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            id=wandb.util.generate_id(),
            log_model=False,
            tags=["offline_probe"],
            config={
                "probe_type": "offline_linear",
                "ssl_model": model_name,
                "backbone": f"vit_{backbone_size}",
                "dataset": dataset_name,
                "checkpoint": str(ckpt_path),
                "checkpoint_name": ckpt_name,
                "probe_epochs": epochs,
                "probe_lr": lr,
                "batch_size": batch_size,
                "embed_dim": embed_dim,
                "num_classes": num_classes,
            },
        )

    # 6. Train (single GPU — round-robin assigned by caller)
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=1,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

    # 7. Collect results
    metric_dict = trainer.callback_metrics
    top1 = metric_dict.get("eval/top1", torch.tensor(0.0)).item()
    top5 = metric_dict.get("eval/top5", torch.tensor(0.0)).item()

    results = {
        "checkpoint": str(ckpt_path),
        "checkpoint_name": ckpt_name,
        "model": model_name,
        "backbone": f"vit_{backbone_size}",
        "dataset": dataset_name,
        "epochs": epochs,
        "top1": top1,
        "top5": top5,
    }

    log.info(f"  → top1={top1:.4f}, top5={top5:.4f}")

    if use_wandb:
        wandb.summary["best_top1"] = top1
        wandb.summary["best_top5"] = top5
        wandb.finish()

    return results


# ============================================================================
# Checkpoint discovery
# ============================================================================


def parse_checkpoint_info(ckpt_path: str) -> tuple[str, str, str] | None:
    """Parse model, backbone size, dataset from checkpoint directory name.

    Expects: .../simclr_vit_small_cifar10/last.ckpt
    Returns: ("simclr", "small", "cifar10") or None.
    """
    parent = Path(ckpt_path).parent.name
    parts = parent.split("_")
    if len(parts) >= 4 and parts[1] == "vit":
        model = parts[0]
        size = parts[2]
        dataset = "_".join(parts[3:])
        return model, size, dataset
    return None


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Offline linear probe evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a single checkpoint")
    group.add_argument("--checkpoint-dir", type=str, default="/mnt/data/sami/stable-datasets/.pretrain_checkpoints", help="Directory to glob for checkpoints")

    parser.add_argument("--model", type=str, help="Model name (auto-detected from path if omitted)")
    parser.add_argument("--backbone-size", type=str, default="small")
    parser.add_argument("--dataset", type=str, help="Dataset name (auto-detected from path if omitted)")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)

    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-entity", type=str, default="stable-ssl")
    parser.add_argument("--wandb-project", type=str, default="stable-datasets-benchmarks")

    parser.add_argument("--output", type=str, default="offline_probe_results.json",
                        help="Output JSON for batch results")

    args = parser.parse_args()

    # Collect checkpoints
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = sorted(
            glob.glob(os.path.join(args.checkpoint_dir, "**/last.ckpt"), recursive=True)
            + glob.glob(os.path.join(args.checkpoint_dir, "**/best*.ckpt"), recursive=True)
        )
        log.info(f"Found {len(ckpt_paths)} checkpoints under {args.checkpoint_dir}")

    if not ckpt_paths:
        log.error("No checkpoints found")
        sys.exit(1)

    # Round-robin GPU assignment
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        log.info(f"Found {num_gpus} GPU(s) — assigning probes round-robin")
    else:
        log.info("No GPUs found — running on CPU")

    all_results = []

    for job_idx, ckpt_path in enumerate(ckpt_paths):
        parsed = parse_checkpoint_info(ckpt_path)

        if parsed:
            model, size, dataset = parsed
        else:
            model, size, dataset = None, args.backbone_size, None

        model = args.model or model
        size = args.backbone_size or size
        dataset = args.dataset or dataset

        if not model or not dataset:
            log.warning(
                f"Skipping {ckpt_path}: could not determine model/dataset. "
                f"Use --model and --dataset, or follow naming: "
                f"{{model}}_vit_{{size}}_{{dataset}}/last.ckpt"
            )
            continue

        gpu_id = job_idx % num_gpus if num_gpus > 0 else None

        results = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            model_name=model,
            backbone_size=size,
            dataset_name=dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            use_wandb=not args.no_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            gpu_id=gpu_id,
        )

        all_results.append(results)

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    log.info("=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    for r in all_results:
        log.info(
            f"  {r['model']:>12s} | {r['dataset']:>20s} | "
            f"{r['checkpoint_name']:>15s} | top1={r['top1']:.4f} | top5={r['top5']:.4f}"
        )
    log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()