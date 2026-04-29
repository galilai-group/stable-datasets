"""Hydra entry point for benchmark runs.

Runs any combination of {model, backbone, dataset} with config-driven
hyperparameters.  See launch.sh for usage examples.
"""

from __future__ import annotations

import logging
import math
import os

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict

import wandb
from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import build_module, create_eval_callbacks, get_transforms


log = logging.getLogger(__name__)


# Param resolution


def _resolve_params(cfg: DictConfig) -> None:
    """Merge per-dataset params from model config into training config.

    Priority: CLI overrides > dataset-specific > model defaults > config.yaml.
    Mutates cfg in place.
    """
    params = cfg.model.get("params", {})
    ds_params = params.get(cfg.dataset, {})
    default_params = params.get("default", {})

    cli_overrides = set()
    try:
        for override in HydraConfig.get().overrides.task:
            cli_overrides.add(override.split("=")[0])
    except ValueError:
        pass

    def _pick(key, cfg_key, fallback):
        if cfg_key in cli_overrides:
            return fallback
        return ds_params.get(key, default_params.get(key, fallback))

    with open_dict(cfg):
        cfg.training.batch_size = _pick("batch_size", "training.batch_size", cfg.training.batch_size)
        cfg.training.max_epochs = _pick("max_epochs", "training.max_epochs", cfg.training.max_epochs)
        cfg.training.accumulate_grad_batches = _pick(
            "accumulate_grad_batches",
            "training.accumulate_grad_batches",
            cfg.training.get("accumulate_grad_batches", 1),
        )
        accum = cfg.training.accumulate_grad_batches
        if accum > 1:
            cfg.training.batch_size = cfg.training.batch_size // accum

        lr_override = ds_params.get("lr", default_params.get("lr", None))
        if lr_override is not None:
            cfg.model._lr_override = float(lr_override)

        # Per-dataset model-level overrides (DINO centering/teacher EMA + SK flag).
        # Raised momenta or SK help stabilize centering on small-batch/few-class
        # datasets.
        for key, caster in (
            ("center_momentum", float),
            ("momentum_teacher", float),
            ("sinkhorn_knopp", bool),
        ):
            if key in cli_overrides or f"model.{key}" in cli_overrides:
                continue
            override = ds_params.get(key, default_params.get(key, None))
            if override is not None:
                cfg.model[key] = caster(override)


# W&B logger


def _create_wandb_logger(cfg: DictConfig, seed: int | None) -> WandbLogger:
    run_name = f"{cfg.model.name}_{cfg.backbone}_{cfg.dataset}"
    if seed is not None:
        run_name += f"_seed{seed}"

    tags = []
    if seed is not None:
        tags.append("seed")
    run_tag = cfg.get("run_tag", None)
    if run_tag is not None:
        tags.append(str(run_tag))

    return WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=run_name,
        id=wandb.util.generate_id(),
        log_model=False,
        save_dir=os.getcwd(),
        tags=tags or None,
        config={
            "model": cfg.model.name,
            "backbone": cfg.backbone,
            "dataset": cfg.dataset,
            "lr": cfg.model.vit_optimizer.lr if hasattr(cfg.model, "vit_optimizer") else cfg.model.optimizer.lr,
            "batch_size": cfg.training.batch_size,
            "accumulate_grad_batches": cfg.training.accumulate_grad_batches,
            "max_epochs": cfg.training.max_epochs,
            "seed": seed,
        },
    )


# GPU assignment for local parallel runs


def _assign_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return
    try:
        job_num = HydraConfig.get().job.num
    except ValueError:
        return
    gpu_id = job_num % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(f"Job #{job_num} pinned to GPU {gpu_id}/{num_gpus}")


# Main


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Disable spt's run-registry / cache_dir on the SLURM worker. Module-level
    # placement is unreliable here because hydra-submitit-launcher unpickles
    # `main` directly and the spt singleton is created during plugin discovery
    # before benchmarks.run is re-imported. Setting this inside main() runs
    # before `spt.Manager(...)` is constructed below, which is when
    # `_resolve_run_dir` reads cache_dir. With cache_dir=None, manager.py:469
    # short-circuits and Lightning's ModelCheckpoint at cfg.checkpoint.dir is
    # the only path that gets written.
    spt.get_config().cache_dir = None
    log.info(f"spt cache_dir disabled (was: {spt.get_config().cache_dir!r})")

    if cfg.get("distribute_gpus", False):
        _assign_gpu()

    seed = cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    _resolve_params(cfg)

    log.info(
        f"{cfg.model.name} | {cfg.backbone} | {cfg.dataset} | "
        f"bs={cfg.training.batch_size} | accum={cfg.training.accumulate_grad_batches} | "
        f"epochs={cfg.training.max_epochs}"
    )

    # Data
    ds_config = get_config(cfg.dataset)
    if ds_config.modality != "image":
        raise ValueError(
            f"Dataset '{cfg.dataset}' is registered as modality={ds_config.modality!r}, "
            "but the current benchmark runner only has image model/backbone support."
        )
    train_transform, val_transform, collate_fn = get_transforms(cfg.model.name, ds_config, cfg.model)
    data, ds_config = create_dataset(
        cfg.dataset,
        train_transform,
        val_transform,
        collate_fn,
        cfg.training,
        data_dir=cfg.get("data_dir"),
    )

    # Scheduler needs total steps
    accum = cfg.training.accumulate_grad_batches
    total_steps = math.ceil(len(data.train) / accum) * cfg.training.max_epochs
    with open_dict(cfg):
        cfg.model._total_steps = total_steps

    # Model
    module, embed_dim = build_module(cfg, ds_config)

    # Callbacks
    callbacks = create_eval_callbacks(module, ds_config, embed_dim)
    ckpt_cfg = cfg.checkpoint
    run_ckpt_dir = os.path.join(ckpt_cfg.dir, f"{cfg.model.name}_{cfg.backbone}_{cfg.dataset}")
    ckpt_kwargs = {
        "dirpath": run_ckpt_dir,
        "filename": "{epoch}-{step}",
        "every_n_epochs": ckpt_cfg.every_n_epochs,
        "save_last": ckpt_cfg.save_last,
        "monitor": ckpt_cfg.monitor,
        "mode": ckpt_cfg.mode,
        "save_top_k": ckpt_cfg.save_top_k,
        "save_weights_only": True,
    }
    callbacks.append(ModelCheckpoint(**ckpt_kwargs))

    # Auto-resume: if a previous SLURM walltime-out / requeue / manual resubmit
    # left a last.ckpt at the same (model, backbone, dataset) path, hand it to
    # spt.Manager so trainer.fit() picks up where it stopped.
    resume_ckpt = os.path.join(run_ckpt_dir, "last.ckpt")
    if not os.path.isfile(resume_ckpt):
        resume_ckpt = None

    # Logger
    smoke_test = cfg.get("smoke_test", False)
    logger = True
    if cfg.wandb.enabled and not smoke_test:
        logger = _create_wandb_logger(cfg, seed)

    # Trainer
    has_val = data.val is not None
    output_dir = HydraConfig.get().runtime.output_dir or os.getcwd()

    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else cfg.training.max_epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=accum,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=3 if smoke_test else 1.0,
        limit_val_batches=3 if smoke_test else (1.0 if has_val else 0),
        accelerator="auto",
        default_root_dir=output_dir,
    )

    try:
        if resume_ckpt is not None:
            log.info(f"Resuming from checkpoint: {resume_ckpt}")
        manager = spt.Manager(trainer=trainer, module=module, data=data, ckpt_path=resume_ckpt)
        manager()
    finally:
        if cfg.wandb.enabled and not smoke_test:
            wandb.finish()


# dataset=all expansion (must run before Hydra parses argv)


def _expand_dataset_all():
    import sys

    from benchmarks.dataset import get_image_dataset_names

    for i, arg in enumerate(sys.argv):
        if arg.startswith("dataset=") and arg.split("=", 1)[1].lower() == "all":
            sys.argv[i] = f"dataset={','.join(get_image_dataset_names(include_results_only=True))}"
            if "--multirun" not in sys.argv and "-m" not in sys.argv:
                sys.argv.append("--multirun")
            break


_expand_dataset_all()

if __name__ == "__main__":
    main()
