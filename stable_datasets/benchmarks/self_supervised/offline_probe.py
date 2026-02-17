"""Offline linear probe evaluation for SSL benchmark checkpoints.

Standard protocol: freeze backbone, train nn.Linear for 100 epochs
with SGD + cosine annealing on extracted features.

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

Checkpoint directory convention (from main.py):
    {checkpoint.dir}/{model}_{backbone}_{dataset}/last.ckpt
    e.g. checkpoints/simclr_vit_small_cifar10/last.ckpt

The model, backbone, and dataset are parsed from the directory name.
Results are saved as JSON next to each checkpoint.
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
# Backbone extraction
# ============================================================================


class FeatureExtractor(nn.Module):
    """Wraps any SSL backbone to return flat [B, D] embeddings.

    Handles all model types:
      - Plain ViT (SimCLR, Barlow, VICReg): returns CLS token directly
      - TeacherStudentWrapper (DINO, LeJEPA): unwraps student
      - MaskedEncoder (MAE): extracts encoded[:, 0]
      - HF ViT: extracts last_hidden_state[:, 0]
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        # Unwrap TeacherStudentWrapper → take student
        if hasattr(backbone, "student"):
            self.backbone = backbone.student
        else:
            self.backbone = backbone

        self.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        # Always stay in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.backbone(images)

        # HF ViT → output.last_hidden_state[:, 0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]

        # MaskedEncoderOutput → output.encoded[:, 0]
        if hasattr(out, "encoded"):
            return out.encoded[:, 0]

        # Plain tensor — could be [B, D] (custom ViT) or [B, T, D]
        if out.ndim == 3:
            return out[:, 0]

        return out


def load_backbone_from_checkpoint(
    ckpt_path: str, model_name: str, backbone_size: str = "small"
) -> tuple[FeatureExtractor, int]:
    """Load a frozen backbone from a Lightning checkpoint.

    Rebuilds the backbone architecture, loads matching weights from the
    checkpoint state_dict, and wraps in FeatureExtractor.

    Returns (feature_extractor, embed_dim).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Collect all backbone.* keys from checkpoint
    backbone_keys = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}

    if not backbone_keys:
        raise ValueError(f"No backbone.* keys found in checkpoint: {ckpt_path}")

    # Detect model structure from key patterns
    has_teacher_student = any(".student." in k or ".teacher." in k for k in backbone_keys)
    has_masked_encoder = any(k.startswith("backbone.encoder.") for k in backbone_keys)

    embed_dim = VIT_CONFIGS[backbone_size]["embed_dim"]

    if model_name == "dino":
        # DINO uses TeacherStudentWrapper(vit_hf(...)) — HuggingFace ViT
        # Student keys: backbone.student.vit.encoder.layer.0...
        backbone = spt.backbone.vit_hf(size=backbone_size, pretrained=False)
        prefix = "backbone.student."
        sub_dict = {
            k[len(prefix):]: v for k, v in backbone_keys.items()
            if k.startswith(prefix)
        }
        if not sub_dict:
            # Maybe saved without wrapper (fallback)
            prefix = "backbone."
            sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        backbone.load_state_dict(sub_dict, strict=False)

    elif model_name == "mae":
        # MAE uses MaskedEncoder wrapping a timm ViT
        from stable_pretraining.backbone.vit import MaskedEncoder, PatchMasking
        backbone = MaskedEncoder(
            model_or_model_name=f"vit_{backbone_size}_patch16_224",
            masking=PatchMasking(mask_ratio=0.0),  # No masking at eval
            pretrained=False,
            img_size=224,
        )
        prefix = "backbone."
        sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        backbone.load_state_dict(sub_dict, strict=False)

    elif model_name == "lejepa":
        # LeJEPA uses TeacherStudentWrapper(create_vit(...))
        raw_vit = create_vit(size=backbone_size, img_size=(224, 224), patch_size=16)
        prefix = "backbone.student."
        sub_dict = {
            k[len(prefix):]: v for k, v in backbone_keys.items()
            if k.startswith(prefix)
        }
        if not sub_dict:
            prefix = "backbone."
            sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        raw_vit.load_state_dict(sub_dict, strict=False)
        backbone = raw_vit

    else:
        # SimCLR, Barlow Twins, VICReg — plain create_vit()
        backbone = create_vit(size=backbone_size, img_size=(224, 224), patch_size=16)
        prefix = "backbone."
        sub_dict = {k[len(prefix):]: v for k, v in backbone_keys.items()}
        backbone.load_state_dict(sub_dict, strict=False)

    extractor = FeatureExtractor(backbone)
    return extractor, embed_dim


# ============================================================================
# Dataset loading (simple single-view transforms for linear eval)
# ============================================================================


def create_eval_transforms(mean, std):
    """Standard linear eval transforms — no multi-view, no heavy augmentation."""
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
                      cache_dir: str | None = CACHE_DIR_DEFAULT,
                      data_dir: str | None = None):
    """Load dataset with simple eval transforms, reusing cached memmaps if available."""
    import numpy as np
    from PIL import Image as PILImage
    from torch.utils.data import Dataset, DataLoader

    # Try loading from cache
    cache_path = Path(cache_dir) / dataset_name if cache_dir else None
    metadata_path = cache_path / "metadata.json" if cache_path else None

    if metadata_path and metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        mean, std = _get_normalization_stats(dataset_name)
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

        train_ds = CachedEvalDataset(
            cache_path / "train_images.npy",
            cache_path / "train_labels.npy",
            train_tf,
        )
        val_ds = CachedEvalDataset(
            cache_path / "val_images.npy",
            cache_path / "val_labels.npy",
            val_tf,
        )

        ds_config = DatasetConfig(
            name=dataset_name,
            num_classes=metadata["num_classes"],
            channels=metadata["original_channels"],
            image_size=(metadata["image_size"], metadata["image_size"]),
            mean=mean,
            std=std,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
        )

        data = spt.data.DataModule(train=train_loader, val=val_loader)
        return data, ds_config

    else:
        # Fallback: load from HF (slow path) — reuse create_dataset with supervised transforms
        raise FileNotFoundError(
            f"No cached dataset found at {cache_path}. "
            f"Run cache_datasets.py first, or implement HF fallback."
        )


# ============================================================================
# Probe forward function
# ============================================================================


def probe_forward(self, batch, stage):
    """Supervised forward that handles frozen backbone output."""
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])
        self.log(f"{stage}/loss", out["loss"], on_step=False, on_epoch=True, sync_dist=True)

    return out


# ============================================================================
# Main evaluation logic
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
) -> dict:
    """Run offline linear probe on a single checkpoint. Returns results dict."""

    log.info(f"Evaluating: {ckpt_path}")
    log.info(f"  model={model_name}, backbone={backbone_size}, dataset={dataset_name}")

    # 1. Load frozen backbone
    extractor, embed_dim = load_backbone_from_checkpoint(ckpt_path, model_name, backbone_size)

    # 2. Load dataset
    data, ds_config = load_eval_dataset(dataset_name, batch_size, cache_dir=cache_dir)

    # 3. Build supervised module
    num_classes = ds_config.num_classes
    classifier = nn.Linear(embed_dim, num_classes)

    module = spt.Module(
        backbone=extractor,
        classifier=classifier,
        forward=probe_forward,
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
    )

    # Metrics callback
    metrics = {
        "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        "top5": torchmetrics.classification.MulticlassAccuracy(
            num_classes, top_k=min(5, num_classes)
        ),
    }
    probe_cb = spt.callbacks.OnlineProbe(
        module,
        name="offline_probe",
        input="embedding",
        target="label",
        probe=nn.Identity(),  # We're already classifying in the forward
        loss=None,  # No separate probe loss
        metrics=metrics,
    )

    # Actually, simpler: just use the built-in logging from probe_forward
    # and track val accuracy via a simple callback
    class AccuracyTracker(pl.Callback):
        def __init__(self, num_classes):
            self.top1 = torchmetrics.classification.MulticlassAccuracy(num_classes)
            self.top5 = torchmetrics.classification.MulticlassAccuracy(
                num_classes, top_k=min(5, num_classes)
            )
            self.best_top1 = 0.0
            self.best_top5 = 0.0

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if "logits" in outputs and "label" in batch:
                logits = outputs["logits"].detach()
                labels = batch["label"]
                self.top1.update(logits, labels)
                self.top5.update(logits, labels)

        def on_validation_epoch_end(self, trainer, pl_module):
            t1 = self.top1.compute().item()
            t5 = self.top5.compute().item()
            self.best_top1 = max(self.best_top1, t1)
            self.best_top5 = max(self.best_top5, t5)
            pl_module.log("eval/top1", t1, sync_dist=True)
            pl_module.log("eval/top5", t5, sync_dist=True)
            self.top1.reset()
            self.top5.reset()

    tracker = AccuracyTracker(num_classes)

    # 4. Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="16-mixed",
        callbacks=[tracker],
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        accelerator="auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

    results = {
        "checkpoint": str(ckpt_path),
        "model": model_name,
        "backbone": backbone_size,
        "dataset": dataset_name,
        "epochs": epochs,
        "top1": tracker.best_top1,
        "top5": tracker.best_top5,
    }

    log.info(f"  → top1={results['top1']:.4f}, top5={results['top5']:.4f}")
    return results


def parse_checkpoint_info(ckpt_path: str) -> tuple[str, str, str] | None:
    """Parse model_backbone_dataset from checkpoint directory name.

    Expects: .../simclr_vit_small_cifar10/last.ckpt
    Returns: ("simclr", "small", "cifar10") or None if can't parse.
    """
    parent = Path(ckpt_path).parent.name
    # Convention: {model}_{vit}_{size}_{dataset}
    # e.g. simclr_vit_small_cifar10, dino_vit_small_flowers102
    parts = parent.split("_")
    if len(parts) >= 4 and parts[1] == "vit":
        model = parts[0]
        size = parts[2]
        dataset = "_".join(parts[3:])  # Handle dataset names with underscores
        return model, size, dataset
    return None


def main():
    parser = argparse.ArgumentParser(description="Offline linear probe evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a single checkpoint")
    group.add_argument("--checkpoint-dir", type=str, default="/mnt/data/sami/stable-datasets/.pretrain_checkpoints", help="Directory to glob for checkpoints")

    parser.add_argument("--model", type=str, help="Model name (auto-detected from path if omitted)")
    parser.add_argument("--backbone-size", type=str, default="small", help="ViT size (default: small)")
    parser.add_argument("--dataset", type=str, help="Dataset name (auto-detected from path if omitted)")

    parser.add_argument("--epochs", type=int, default=100, help="Probe training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--output", type=str, default="offline_probe_results.json",
                        help="Output JSON file for batch results")

    args = parser.parse_args()

    # Collect checkpoints to evaluate
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        # Glob for last.ckpt and best*.ckpt under the directory
        ckpt_paths = sorted(
            glob.glob(os.path.join(args.checkpoint_dir, "**/last.ckpt"), recursive=True)
            + glob.glob(os.path.join(args.checkpoint_dir, "**/best*.ckpt"), recursive=True)
        )
        log.info(f"Found {len(ckpt_paths)} checkpoints under {args.checkpoint_dir}")

    if not ckpt_paths:
        log.error("No checkpoints found")
        sys.exit(1)

    all_results = []

    for ckpt_path in ckpt_paths:
        # Auto-detect model/dataset from path, or use CLI args
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
                f"Use --model and --dataset flags, or follow naming convention "
                f"{{model}}_vit_{{size}}_{{dataset}}/last.ckpt"
            )
            continue

        results = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            model_name=model,
            backbone_size=size,
            dataset_name=dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )

        all_results.append(results)

        # Save incrementally (in case of crash)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for r in all_results:
        log.info(f"  {r['model']:>12s} | {r['dataset']:>20s} | top1={r['top1']:.4f} | top5={r['top5']:.4f}")

    log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()