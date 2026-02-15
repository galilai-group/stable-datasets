"""SSL Benchmarks for stable-datasets.

Hydra-driven entry point that runs any combination of
{model, backbone, dataset} for self-supervised learning evaluation.

Incompatible combos (e.g. MAE + ResNet) are skipped with a warning.

Usage:
    # Single run
    python -m stable_datasets.benchmarks.self_supervised.main dataset=cifar10 model=simclr backbone=resnet18

    # Sweep all models x backbones on one dataset (local)
    python -m stable_datasets.benchmarks.self_supervised.main --multirun \\
        dataset=cifar10 model=simclr,dino,mae,lejepa backbone=resnet18,vit_tiny

    # Sweep across multiple datasets
    python -m stable_datasets.benchmarks.self_supervised.main --multirun \\
        dataset=cifar10,stl10,flowers102 model=simclr,dino backbone=resnet18,vit_tiny

    # Sweep (SLURM) — each combo becomes a separate job
    python -m stable_datasets.benchmarks.self_supervised.main --multirun --config-name slurm \\
        dataset=cifar10,stl10 model=simclr,dino,mae,lejepa backbone=resnet18,vit_tiny

    # Override hyperparameters for a specific run
    python -m stable_datasets.benchmarks.self_supervised.main dataset=cifar10 model=simclr backbone=resnet50 \\
        model.optimizer.lr=0.3 training.max_epochs=200

    # Run on ALL datasets
    python -m stable_datasets.benchmarks.self_supervised.main dataset=all model=simclr backbone=resnet18

    # LR sweep (useful for tuning)
    python -m stable_datasets.benchmarks.self_supervised.main --multirun \\
        dataset=cifar10 model=simclr backbone=resnet18 \\
        model.optimizer.lr=0.1,1.0,5.0,10.0

    # Parallel across local GPUs (round-robin assignment)
    python -m stable_datasets.benchmarks.self_supervised.main --multirun --config-name local_parallel \\
        dataset=cifar10,stl10,svhn,cifar100 model=simclr backbone=resnet18

    # Override GPU count for local parallel
    NUM_GPUS=4 python -m stable_datasets.benchmarks.self_supervised.main --multirun --config-name local_parallel \\
        dataset=all model=lejepa backbone=resnet18
"""

from __future__ import annotations

import logging
import os

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict

import stable_datasets as sds

from stable_datasets.benchmarks.self_supervised.dataset import create_dataset
from stable_datasets.benchmarks.self_supervised.modules import build_module, create_eval_callbacks

log = logging.getLogger(__name__)

# Datasets to always skip:
#   - cars196: download issues
#   - cifar10c/cifar100c: corruption benchmarks, not standard SSL datasets
#   - clevrer: video dataset, no label field
#   - dsprites*: multi-factor labels (Sequence), no ClassLabel/num_classes
#   - cars3d, shapes3d, smallnorb: same multi-factor label issue
#   - facepointing, celeba: no label field
SKIP_DATASETS = {
    "cars196", "cifar10c", "cifar100c", "clevrer",
    "dsprites", "dspritescolor", "dspritesnoise", "dspritesscream",
    "cars3d", "shapes3d", "smallnorb",
    "facepointing", "celeba",
}


def _parse_skip_datasets_from_argv() -> set[str]:
    """Extract skip_datasets from sys.argv before Hydra parses it."""
    import sys

    for arg in sys.argv:
        if arg.startswith("skip_datasets="):
            val = arg.split("=", 1)[1]
            # Handle both skip_datasets=[a,b] and skip_datasets=a,b
            val = val.strip("[]")
            if val:
                return {s.strip().lower() for s in val.split(",")}
    return set()


def _get_all_dataset_names(extra_skip: set[str] = None) -> list[str]:
    """Discover all available dataset names from stable_datasets.images."""
    skip = SKIP_DATASETS | (extra_skip or set())
    return sorted(
        name.lower()
        for name, cls in vars(sds.images).items()
        if (
            isinstance(cls, type)
            and issubclass(cls, sds.BaseDatasetBuilder)
            and name.lower() not in skip
        )
    )


def _expand_dataset_all_in_argv():
    """Rewrite sys.argv before Hydra sees it: dataset=all -> dataset=name1,name2,..."""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg.startswith("dataset=") and arg.split("=", 1)[1].lower() == "all":
            extra_skip = _parse_skip_datasets_from_argv()
            all_names = ",".join(_get_all_dataset_names(extra_skip))
            sys.argv[i] = f"dataset={all_names}"
            if "--multirun" not in sys.argv and "-m" not in sys.argv:
                sys.argv.append("--multirun")
            log.info(f"dataset=all expanded to {all_names}")
            break


_expand_dataset_all_in_argv()


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Pin each multirun job to a different GPU
    if cfg.get("distribute_gpus", False):
        _assign_gpu()

    # Skip datasets in the skip list
    skip_list = {s.lower() for s in cfg.get("skip_datasets", [])}
    if cfg.dataset.lower() in skip_list:
        log.warning(f"Skipping dataset '{cfg.dataset}' (in skip_datasets list)")
        return

    # Validate backbone x model compatibility
    if cfg.model.requires_vit and cfg.backbone.type != "vit":
        log.warning(
            f"Skipping {cfg.model.name} + {cfg.backbone.name}: "
            f"{cfg.model.name} requires a ViT backbone."
        )
        return

    # Resolve per-model, per-dataset params (batch_size, max_epochs, lr)
    params = cfg.model.get("params", {})
    ds_params = params.get(cfg.dataset, {})
    default_params = params.get("default", {})

    def _resolve(key, fallback):
        """Check dataset-specific params, then default params, then global fallback."""
        if key in ds_params:
            return ds_params[key]
        if key in default_params:
            return default_params[key]
        return fallback

    with open_dict(cfg):
        cfg.training.batch_size = _resolve("batch_size", cfg.training.batch_size)
        cfg.training.max_epochs = _resolve("max_epochs", cfg.training.max_epochs)
        # LR override stored for optim builder
        lr_override = ds_params.get("lr", default_params.get("lr", None))
        if lr_override is not None:
            cfg.model._lr_override = float(lr_override)

    log.info(
        f"Running {cfg.model.name} | backbone={cfg.backbone.name} "
        f"| dataset={cfg.dataset} | batch_size={cfg.training.batch_size} "
        f"| max_epochs={cfg.training.max_epochs}"
    )

    # Load dataset with config-driven transforms
    data, ds_config = create_dataset(
        cfg.dataset, cfg.model.transforms, cfg.training, data_dir=cfg.get("data_dir"),
    )

    # Build module from registry
    module, embed_dim = build_module(cfg, ds_config)

    # Evaluation callbacks
    callbacks = create_eval_callbacks(module, ds_config, embed_dim)

    # Logger — each multirun job gets its own wandb run
    logger = None
    if cfg.wandb.enabled:
        run_name = f"{cfg.model.name}_{cfg.backbone.name}_{cfg.dataset}"
        logger = WandbLogger(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=run_name,
            id=wandb.util.generate_id(),
            log_model=False,
            save_dir=os.getcwd(),
            config={
                "model": cfg.model.name,
                "backbone": cfg.backbone.name,
                "dataset": cfg.dataset,
            },
        )

    # Trainer - use Hydra's output directory (benchmark-runs/...)
    has_val = data.val is not None
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir if hydra_cfg.runtime.output_dir else os.getcwd()

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if has_val else 0,
        enable_checkpointing=False,
        accelerator="auto",
        default_root_dir=output_dir,
    )

    # Run
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

    # Close wandb run so the next multirun job gets a fresh run
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
