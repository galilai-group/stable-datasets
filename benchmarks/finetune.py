"""Finetune entry point (ViT checkpoints only).

For one pretrained ViT checkpoint: load its backbone, attach a FRESH linear head,
subsample the train set (10%, uniform label distribution), and finetune the whole
model (backbone + head) end-to-end with cross-entropy — using MAE's ImageNet
finetune recipe adapted to our low-data setting:

  * effective batch = min(reference_batch=256, subset); LR = base_lr(3e-4) * eff/256
    (MAE 1e-3@1024 scaled to 256, rounded up; ViT-S @224 fits 256 so no grad-accum)
  * 100 epochs, AdamW (wd 0.05, betas 0.9/0.999), cosine decay, 5 warmup epochs
  * RandAugment(2, 9), label smoothing 0.1, drop-path 0.1
  * layer-wise LR decay and mixup/cutmix deliberately omitted

The training schematic is the supervised model (backbone -> linear -> CE); we
reuse its `forward`. See conf/finetune.yaml for the knobs.

    python -m benchmarks.finetune model=simclr backbone=vit_small_patch16_224 dataset=cifar10
"""

from __future__ import annotations

import glob
import logging
import math
import os

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import torchvision.transforms.v2 as tv2
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from stable_pretraining.data import transforms as T
from torch import nn

import wandb
from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import (
    collate_single,
    create_backbone,
    create_eval_callbacks,
    get_embedding_dim,
    resolve_backbone_family,
)
from benchmarks.models.supervised import forward as supervised_forward


log = logging.getLogger(__name__)


# Transforms — MAE-style finetune aug: RRC + flip + RandAugment + normalize.


def finetune_transform(ds_config, num_ops: int, magnitude: int) -> T.Compose:
    h, w = ds_config.image_size
    ops = [
        T.RGB(),
        T.RandomResizedCrop((h, w), scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        # RandAugment operates on uint8 here (before ToImage normalizes to float).
        T.WrapTorchTransform(tv2.RandAugment(num_ops=num_ops, magnitude=magnitude)),
        T.ToImage(mean=ds_config.mean, std=ds_config.std),
    ]
    return T.Compose(*ops)


def val_finetune_transform(ds_config) -> T.Compose:
    h, w = ds_config.image_size
    return T.Compose(T.RGB(), T.Resize((h, w)), T.ToImage(mean=ds_config.mean, std=ds_config.std))


# Epochs (num_epochs_pct scales the method's pretrain epochs; else num_epochs)


def _pretrain_epochs(cfg: DictConfig) -> int:
    params = cfg.model.get("params", {})
    ds_params = params.get(cfg.dataset, {})
    default_params = params.get("default", {})
    return int(ds_params.get("max_epochs", default_params.get("max_epochs", 100)))


def _resolve_finetune_epochs(cfg: DictConfig) -> tuple[int, int]:
    pretrain = _pretrain_epochs(cfg)
    pct = cfg.finetune.get("num_epochs_pct", None)
    if pct is not None:
        return max(1, math.ceil(float(pct) / 100.0 * pretrain)), pretrain
    return int(cfg.finetune.num_epochs), pretrain


# Pretrained checkpoint (backbone only, fresh head)


def _pretrained_ckpt_path(cfg: DictConfig) -> tuple[str, str]:
    # sds_neurips_review layout: {checkpoint_root}/{model}/{dataset}/last.ckpt.
    # One curated checkpoint per (method, dataset); the source seed is recorded in
    # that tree's manifest.csv, not encoded in the path.
    ckpt_dir = os.path.join(cfg.finetune.checkpoint_root, cfg.model.name, cfg.dataset)
    if cfg.finetune.get("which", "last") == "best":
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt")))
        ckpt = cands[-1] if cands else os.path.join(ckpt_dir, "last.ckpt")
    else:
        ckpt = os.path.join(ckpt_dir, "last.ckpt")
    return f"{cfg.model.name}_{cfg.dataset}", ckpt


def _load_backbone_weights(backbone: nn.Module, ckpt_path: str) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No pretrained checkpoint at {ckpt_path}. Pretrain that "
            f"(model, backbone, dataset) first, or set finetune.which/load_seed."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    bb_state = {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
    if not bb_state:
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'backbone.*' weights")
    # EMA methods (DINO) wrap the encoder in a TeacherStudentWrapper, so the
    # checkpoint stores backbone.teacher.* / backbone.student.* (+ EMA scalars)
    # rather than a plain backbone.*. Unwrap the TEACHER encoder — it is what
    # DINO's online probe evaluates, so it matches the reported baseline. Without
    # this the plain-backbone load silently misses every tensor (random backbone).
    if any(k.startswith(("teacher.", "student.")) for k in bb_state):
        prefix = "teacher." if any(k.startswith("teacher.") for k in bb_state) else "student."
        bb_state = {k[len(prefix) :]: v for k, v in bb_state.items() if k.startswith(prefix)}
        log.info(f"Unwrapped '{prefix}' encoder from TeacherStudentWrapper checkpoint")
    # MAE stores its encoder as a MaskedEncoder: the ViT lives under `backbone.vit.*`
    # (alongside a separate MaskedEncoder-level `patch_embed` used for masking). Unwrap
    # the `vit.` encoder — those are exactly the plain-ViT weights the finetune wants —
    # mirroring the DINO teacher/student unwrap above. No-op for every other method.
    if any(k.startswith("vit.") for k in bb_state):
        bb_state = {k[len("vit.") :]: v for k, v in bb_state.items() if k.startswith("vit.")}
        log.info("Unwrapped 'vit.' encoder from MAE MaskedEncoder checkpoint")
    missing, unexpected = backbone.load_state_dict(bb_state, strict=False)
    log.info(
        f"Loaded {len(bb_state)} backbone tensors from {ckpt_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    # Fail loud: any missing backbone param means part of the encoder is random,
    # which silently invalidates the finetune (this is exactly how the DINO runs
    # trained from scratch). A correct load matches the architecture exactly.
    if missing:
        raise RuntimeError(
            f"Backbone load from {ckpt_path} left {len(missing)} params uninitialized "
            f"(e.g. {list(missing)[:3]}); refusing to finetune a partially-random backbone."
        )


# W&B logger — finetune runs named + tagged distinctly


def _create_wandb_logger(cfg, ft_epochs, pretrain_epochs, eff_batch, lr, n_params, seed):
    family = resolve_backbone_family(cfg.backbone)
    run_name = f"finetune_{cfg.model.name}_{cfg.backbone}_{cfg.dataset}"
    load_seed = cfg.finetune.get("load_seed", None)
    if load_seed is not None:  # which pretrained seed this finetune was initialized from
        run_name += f"_seed{load_seed}"
    return WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=run_name,
        id=wandb.util.generate_id(),
        log_model=False,
        save_dir=os.getcwd(),
        tags=["finetune", family],
        config={
            "experiment": "finetune",
            "model": cfg.model.name,
            "backbone": cfg.backbone,
            "backbone_family": family,
            "n_params": n_params,
            "dataset": cfg.dataset,
            "load_seed": load_seed,
            "lr": lr,
            "effective_batch": eff_batch,
            "weight_decay": cfg.finetune.weight_decay,
            "finetune_epochs": ft_epochs,
            "pretrain_epochs": pretrain_epochs,
            "warmup_epochs": cfg.finetune.warmup_epochs,
            "subsample_pct": cfg.finetune.subsample_pct,
            "label_balance": cfg.finetune.label_balance,
            "label_smoothing": cfg.finetune.label_smoothing,
            "drop_path": cfg.finetune.drop_path,
            "randaug": f"{cfg.finetune.randaug_num_ops},{cfg.finetune.randaug_magnitude}",
        },
    )


@hydra.main(version_base=None, config_path="conf", config_name="finetune")
def main(cfg: DictConfig) -> None:
    spt.get_config().cache_dir = None

    if resolve_backbone_family(cfg.backbone) != "vit":
        raise ValueError(
            f"finetune is configured for ViT checkpoints only; got backbone={cfg.backbone!r}."
        )

    seed = cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    ds_config = get_config(cfg.dataset)
    if ds_config.modality != "image":
        raise ValueError(f"finetune supports image datasets only; got {ds_config.modality!r}")

    ft = cfg.finetune
    train_t = finetune_transform(ds_config, int(ft.randaug_num_ops), int(ft.randaug_magnitude))
    val_t = val_finetune_transform(ds_config)

    # Effective batch = min(reference, subset); ViT-S @224 fits it, so DataLoader
    # batch == effective (no accum). drop_last=False so tiny subsets still batch.
    ref = int(ft.reference_batch)
    # Peek subset size to size the batch (10% uniform of train).
    import benchmarks.dataset as _ds

    full_train = _peek_train_size(cfg.dataset)
    subset_size = math.ceil(ft.subsample_pct / 100.0 * full_train)
    eff_batch = max(1, min(ref, subset_size))
    lr = float(ft.base_lr) * eff_batch / ref

    training_cfg = dict(
        batch_size=eff_batch,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.get("prefetch_factor", 2),
    )
    from omegaconf import OmegaConf

    data, ds_config = create_dataset(
        cfg.dataset,
        train_t,
        val_t,
        collate_single,
        OmegaConf.create(training_cfg),
        data_dir=cfg.get("data_dir"),
        train_subsample={"pct": ft.subsample_pct, "balance": ft.label_balance, "seed": ft.subsample_seed},
        train_drop_last=False,
    )

    ft_epochs, pretrain_epochs = _resolve_finetune_epochs(cfg)
    steps_per_epoch = max(1, len(data.train))
    total_steps = steps_per_epoch * ft_epochs
    warmup_steps = max(1, int(ft.warmup_epochs) * steps_per_epoch)

    log.info(
        f"FINETUNE {cfg.model.name} | {cfg.backbone} | {cfg.dataset} | "
        f"eff_batch={eff_batch} lr={lr:.2e} wd={ft.weight_decay} | epochs={ft_epochs} "
        f"warmup={ft.warmup_epochs}ep | steps/ep={steps_per_epoch} total={total_steps} | "
        f"subset={subset_size} ({ft.subsample_pct}% {ft.label_balance})"
    )

    # Model: pretrained backbone (+ drop-path) + FRESH head; CE with label smoothing.
    backbone = create_backbone(cfg.backbone, ds_config, drop_path_rate=float(ft.drop_path))
    embed_dim = get_embedding_dim(backbone)
    run_dir_name, ckpt_path = _pretrained_ckpt_path(cfg)
    _load_backbone_weights(backbone, ckpt_path)

    optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": float(ft.weight_decay),
            "betas": tuple(ft.betas),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": total_steps,
            "peak_step": warmup_steps,
        },
        "interval": "step",
    }
    module = spt.Module(
        backbone=backbone,
        classifier=nn.Linear(embed_dim, ds_config.num_classes),
        supervised_loss=nn.CrossEntropyLoss(label_smoothing=float(ft.label_smoothing)),
        forward=supervised_forward,
        optim=optim,
    )
    n_params = sum(p.numel() for p in module.backbone.parameters())

    callbacks = create_eval_callbacks(module, ds_config, embed_dim)
    ft_ckpt_dir = os.path.join(cfg.checkpoint.dir, "finetune", run_dir_name)
    callbacks.append(
        ModelCheckpoint(
            dirpath=ft_ckpt_dir,
            filename="{epoch}-{step}",
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            save_last=cfg.checkpoint.save_last,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_weights_only=False,
        )
    )

    smoke_test = cfg.get("smoke_test", False)
    logger = True
    if cfg.wandb.enabled and not smoke_test:
        logger = _create_wandb_logger(cfg, ft_epochs, pretrain_epochs, eff_batch, lr, n_params, seed)

    has_val = data.val is not None
    try:
        output_dir = HydraConfig.get().runtime.output_dir or os.getcwd()
    except ValueError:
        output_dir = os.getcwd()

    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else ft_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=3 if smoke_test else 1.0,
        limit_val_batches=3 if smoke_test else (1.0 if has_val else 0),
        accelerator="auto",
        default_root_dir=output_dir,
    )

    try:
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
    finally:
        if cfg.wandb.enabled and not smoke_test:
            wandb.finish()


def _peek_train_size(dataset: str) -> int:
    """Full train-split row count from the processed-cache metadata (no data load)."""
    import glob as _glob
    import json as _json

    root = os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/.stable-datasets"))
    for meta in _glob.glob(f"{root}/processed/*{dataset}*_train_*/_metadata.json") + _glob.glob(
        f"{root}/processed/*{dataset}*train*/_metadata.json"
    ):
        try:
            return int(_json.load(open(meta))["num_rows"])
        except Exception:
            continue
    raise FileNotFoundError(f"could not find processed train metadata for dataset {dataset!r} under {root}")


if __name__ == "__main__":
    main()
