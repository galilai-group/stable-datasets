"""Offline linear probe evaluation for SSL benchmark checkpoints.

Standard protocol: freeze backbone with spt.backbone.EvalOnly, train nn.Linear
for 100 epochs with SGD + cosine annealing using spt.forward.supervised_forward.

Uses Hydra for configuration (mirrors main.py). Parallel GPU execution via
the joblib launcher with round-robin GPU assignment.

Usage:
    # Single probe
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        ssl_model=simclr dataset=cifar10

    # Sweep all models × datasets (sequential)
    python -m stable_datasets.benchmarks.self_supervised.offline_probe --multirun \
        ssl_model=simclr,dino,mae,lejepa dataset=cifar10,stl10,flowers102

    # Parallel across local GPUs (one probe per GPU, like local_parallel)
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --multirun --config-name offline_probe_parallel \
        ssl_model=simclr,dino,mae,lejepa dataset=cifar10,stl10,flowers102

    # Override GPU count
    NUM_GPUS=4 python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --multirun --config-name offline_probe_parallel \
        ssl_model=simclr,dino,mae,lejepa dataset=cifar10,stl10

    # Override probe hyperparams
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        ssl_model=dino dataset=flowers102 probe.epochs=200 probe.lr=0.05

    # ALL checkpoints in the directory, parallel across GPUs
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --multirun --config-name offline_probe_parallel \
        ssl_model=all dataset=all

    # All models on specific datasets
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        --multirun --config-name offline_probe_parallel \
        ssl_model=all dataset=cifar10,stl10

    # Without wandb
    python -m stable_datasets.benchmarks.self_supervised.offline_probe \
        ssl_model=simclr dataset=cifar10 wandb.enabled=false

Checkpoint directory convention (from main.py):
    {checkpoint.dir}/{model}_{backbone}_{dataset}/{checkpoint.name}.ckpt
    e.g. checkpoints/simclr_vit_small_cifar10/last.ckpt

Results are logged to wandb (tagged "offline_probe") and saved as JSON
in Hydra's output directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import torchmetrics
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch import nn
from torchvision.transforms import v2 as transforms

from stable_datasets.benchmarks.self_supervised.dataset import (
    DatasetConfig,
    _get_normalization_stats,
    CACHE_DIR_DEFAULT,
)
from stable_datasets.benchmarks.self_supervised.models import create_vit, VIT_CONFIGS

log = logging.getLogger(__name__)

# Default checkpoint dir (must match offline_probe.yaml)
_DEFAULT_CKPT_DIR = "/mnt/data/sami/stable-datasets/.pretrain_checkpoints"


# ============================================================================
# argv expansion: ssl_model=all / dataset=all  (mirrors main.py dataset=all)
# ============================================================================


def _get_argv_value(key: str, default: str) -> str:
    """Extract key=value from sys.argv, or return default."""
    prefix = f"{key}="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return default


def _scan_checkpoint_dir(ckpt_dir: str) -> tuple[list[str], list[str]]:
    """Scan checkpoint directory for available (model, dataset) combos.

    Expects subdirs named {model}_vit_{size}_{dataset}/ containing *.ckpt.
    Returns (sorted_models, sorted_datasets).
    """
    models, datasets = set(), set()
    if not os.path.isdir(ckpt_dir):
        return [], []
    for entry in os.listdir(ckpt_dir):
        subdir = os.path.join(ckpt_dir, entry)
        if not os.path.isdir(subdir):
            continue
        parts = entry.split("_")
        # {model}_vit_{size}_{dataset...}
        if len(parts) >= 4 and parts[1] == "vit":
            models.add(parts[0])
            datasets.add("_".join(parts[3:]))
    return sorted(models), sorted(datasets)


def _expand_all_in_argv():
    """Rewrite sys.argv before Hydra sees it: ssl_model=all / dataset=all → discovered values."""
    has_model_all = any(
        arg.startswith("ssl_model=") and arg.split("=", 1)[1].lower() == "all"
        for arg in sys.argv
    )
    has_dataset_all = any(
        arg.startswith("dataset=") and arg.split("=", 1)[1].lower() == "all"
        for arg in sys.argv
    )
    if not has_model_all and not has_dataset_all:
        return

    ckpt_dir = _get_argv_value("checkpoint.dir", _DEFAULT_CKPT_DIR)
    models, datasets = _scan_checkpoint_dir(ckpt_dir)

    for i, arg in enumerate(sys.argv):
        if arg.startswith("ssl_model=") and arg.split("=", 1)[1].lower() == "all":
            sys.argv[i] = f"ssl_model={','.join(models)}"
        elif arg.startswith("dataset=") and arg.split("=", 1)[1].lower() == "all":
            sys.argv[i] = f"dataset={','.join(datasets)}"

    if "--multirun" not in sys.argv and "-m" not in sys.argv:
        sys.argv.append("--multirun")


_expand_all_in_argv()


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
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, drop_last=True,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context="fork" if num_workers > 0 else None,
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
) -> dict:
    """Run offline linear probe on a single checkpoint."""

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

    # 6. Train (single GPU — assigned by _assign_gpu via Hydra job.num)
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

    trainer.fit(module, datamodule=data)

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
# GPU assignment (mirrors main.py)
# ============================================================================


def _assign_gpu():
    """Pin this job to a single GPU via round-robin over available devices."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return
    try:
        job_num = HydraConfig.get().job.num
    except ValueError:
        return  # not a multirun
    gpu_id = job_num % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(f"Job #{job_num} pinned to GPU {gpu_id}/{num_gpus}")


# ============================================================================
# Hydra entry point
# ============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="offline_probe")
def main(cfg: DictConfig) -> None:
    # Pin each multirun job to a different GPU
    if cfg.get("distribute_gpus", False):
        _assign_gpu()

    # Construct checkpoint path from config
    backbone_size = cfg.ssl_backbone.replace("vit_", "")
    ckpt_dir = os.path.join(
        cfg.checkpoint.dir, f"{cfg.ssl_model}_{cfg.ssl_backbone}_{cfg.dataset}"
    )
    ckpt_name = cfg.checkpoint.get("name", "last")
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.ckpt")

    if not os.path.exists(ckpt_path):
        log.warning(f"Checkpoint not found: {ckpt_path}, skipping")
        return

    result = evaluate_checkpoint(
        ckpt_path=ckpt_path,
        model_name=cfg.ssl_model,
        backbone_size=backbone_size,
        dataset_name=cfg.dataset,
        epochs=cfg.probe.epochs,
        lr=cfg.probe.lr,
        batch_size=cfg.probe.batch_size,
        cache_dir=cfg.get("cache_dir", CACHE_DIR_DEFAULT),
        use_wandb=cfg.wandb.enabled,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
    )

    # Save result JSON to Hydra's output directory
    out_path = os.path.join(os.getcwd(), "probe_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
