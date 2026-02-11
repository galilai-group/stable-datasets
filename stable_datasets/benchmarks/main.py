"""SSL Benchmarks for stable-datasets.

Hydra-driven entry point that runs any combination of
{model, backbone, dataset} for self-supervised learning evaluation.

Incompatible combos (e.g. MAE + ResNet) are skipped with a warning.

Usage:
    # Single run
    python -m stable_datasets.benchmarks.main dataset=cifar10 model=simclr backbone=resnet18

    # Sweep all models x backbones on one dataset (local)
    python -m stable_datasets.benchmarks.main --multirun \\
        dataset=cifar10 model=simclr,dino,mae,lejepa backbone=resnet18,vit_tiny

    # Sweep across multiple datasets
    python -m stable_datasets.benchmarks.main --multirun \\
        dataset=cifar10,stl10,flowers102 model=simclr,dino backbone=resnet18,vit_tiny

    # Sweep (SLURM) — each combo becomes a separate job
    python -m stable_datasets.benchmarks.main --multirun --config-name slurm \\
        dataset=cifar10,stl10 model=simclr,dino,mae,lejepa backbone=resnet18,vit_tiny

    # Override hyperparameters for a specific run
    python -m stable_datasets.benchmarks.main dataset=cifar10 model=simclr backbone=resnet50 \\
        model.optimizer.lr=0.3 training.max_epochs=200

    # LR sweep (useful for tuning)
    python -m stable_datasets.benchmarks.main --multirun \\
        dataset=cifar10 model=simclr backbone=resnet18 \\
        model.optimizer.lr=0.1,1.0,5.0,10.0
"""

from __future__ import annotations

import logging

import hydra
import lightning as pl
import stable_pretraining as spt
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from stable_datasets.benchmarks.dataset import create_dataset
from stable_datasets.benchmarks.modules import build_module, create_eval_callbacks

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Validate backbone x model compatibility
    if cfg.model.requires_vit and cfg.backbone.type != "vit":
        log.warning(
            f"Skipping {cfg.model.name} + {cfg.backbone.name}: "
            f"{cfg.model.name} requires a ViT backbone."
        )
        return

    log.info(
        f"Running {cfg.model.name} | backbone={cfg.backbone.name} | dataset={cfg.dataset}"
    )

    # Load dataset with config-driven transforms
    data, ds_config = create_dataset(cfg.dataset, cfg.model.transforms, cfg.training)

    # Build module from registry
    module, embed_dim = build_module(cfg, ds_config)

    # Evaluation callbacks
    callbacks = create_eval_callbacks(module, ds_config, embed_dim)

    # Logger
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=f"{cfg.model.name}_{cfg.backbone.name}_{cfg.dataset}",
            log_model=False,
            config={
                "model": cfg.model.name,
                "backbone": cfg.backbone.name,
                "dataset": cfg.dataset,
            },
        )

    # Trainer
    has_val = data.val is not None
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if has_val else 0,
        enable_checkpointing=False,
        accelerator="auto",
    )

    # Run
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
