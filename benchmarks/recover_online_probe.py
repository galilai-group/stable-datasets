"""Recover ONLINE eval metrics (linear probe, kNN, RankMe, LiDAR) by a FROZEN
re-eval of a saved checkpoint — W&B-independent.

Motivation: the online numbers that get compared against the offline finetuned-backbone
probe live in W&B, and some W&B runs are corrupt (pre-fix `wandb_resume.json` cross-run
contamination — different datasets logged into one run, e.g. run 7rbvv9tl). The
CHECKPOINTS on disk are not corrupt, so we recompute the online numbers directly.

Why a pure forward pass reproduces them: the checkpoint carries the frozen backbone,
the trained linear-probe head (`callbacks_modules.linear_probe.*`, preserved by commit
15d8f9a) AND the final-epoch kNN/RankMe/LiDAR feature bank (`ordered_queue_*`). The
queue only appends during TRAIN (`on_train_batch_end`) and validation takes a read-only
snapshot (`on_validation_epoch_start`), so a validate pass reproduces every metric
exactly without polluting the bank. Validated bit-exact against clean W&B runs.

Idempotent: reads the per-cell `max_epochs` from the (hydra-composed) config and only
computes for checkpoints that reached the final epoch. Cells still training write an
`incomplete` marker (overwritten by a later re-run once they finish).

Output: one file per cell in RECOVER_DIR, `{method}__{dataset}__seed{N|null}.csv`, with
columns (Sami's spec): dataset, method, seed(nullable), origin(online_rerun|wandb),
linear_top1, linear_top5, knn_top1, knn_top5, rankme, lidar, plus provenance
(rankme_condition_number, rankme_entropy, lidar_entropy, epoch, max_epochs, num_classes,
backbone, checkpoint_or_run). Merge with benchmarks/merge_recovered_online_probe.py.

    python -m benchmarks.recover_online_probe model=barlow_twins dataset=cifar10 \
        backbone=vit_small_patch16_224 [+seed=1]
"""

from __future__ import annotations

import csv
import glob
import logging
import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig, open_dict

from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import build_module, create_eval_callbacks, get_transforms
from benchmarks.run import _resolve_params  # merges per-dataset max_epochs/batch_size into cfg.training

log = logging.getLogger(__name__)

DEFAULT_DIR = os.path.join(
    os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")),
    "online_probe_recovered",
)
# Column order = Sami's spec, then provenance/extras.
CSV_HEADER = [
    "dataset", "method", "seed", "origin",
    "linear_top1", "linear_top5", "knn_top1", "knn_top5", "rankme", "lidar",
    "rankme_condition_number", "rankme_entropy", "lidar_entropy",
    "epoch", "max_epochs", "num_classes", "backbone", "checkpoint_or_run",
]


def _final_checkpoint(ckpt_dir: str) -> str | None:
    for pattern in ("last.resume-clean.ckpt", "last-v1.ckpt", "last.ckpt"):
        hits = glob.glob(os.path.join(ckpt_dir, pattern))
        if hits:
            return max(hits, key=os.path.getmtime)
    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))
    return max(epoch_ckpts, key=os.path.getmtime) if epoch_ckpts else None


def _write_cell(out_dir: str, method: str, dataset: str, seed, row: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{method}__{dataset}__seed{seed if seed is not None else 'null'}.csv"
    with open(os.path.join(out_dir, stem), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    # Fixed RNG seed so re-runs are IDEMPOTENT: makes kNN tie-breaking and LiDAR's
    # surrogate-class sampling deterministic (linear/rankme are already exact). This
    # is the recovery's own seed, unrelated to the checkpoint's training `seed` col.
    pl.seed_everything(0, workers=True)
    # Merge per-dataset params (max_epochs, batch_size, accum, lr) from cfg.model.params
    # into cfg.training — EXACTLY as run.py does. Without this cfg.training.max_epochs
    # stays the config default (50) and the completion guard compares against the wrong
    # value, so a still-training cell at epoch>=49 would be mis-passed as complete.
    _resolve_params(cfg)
    seed = cfg.get("seed", None)
    out_dir = os.environ.get("RECOVER_DIR", DEFAULT_DIR)  # env, not hydra override (struct-mode)
    method, dataset, backbone = cfg.model.name, cfg.dataset, cfg.backbone
    base = {"dataset": dataset, "method": method,
            "seed": seed if seed is not None else "", "backbone": backbone}

    run_dir = f"{method}_{backbone}_{dataset}" + (f"_seed{seed}" if seed is not None else "")
    ckpt_dir = os.path.join(cfg.checkpoint.dir, run_dir)
    ckpt = _final_checkpoint(ckpt_dir)
    if ckpt is None:
        log.warning(f"No checkpoint under {ckpt_dir}; marking missing_checkpoint (needs wandb).")
        _write_cell(out_dir, method, dataset, seed, {**base, "origin": "missing_checkpoint"})
        return

    # Idempotency guard: only recover checkpoints that finished all epochs.
    # cfg.training.max_epochs is the per-cell max (model yaml sets it per dataset);
    # Lightning epochs are 0-indexed so the final epoch is max_epochs - 1.
    max_epochs = int(cfg.training.max_epochs)
    ckpt_epoch = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch")
    if ckpt_epoch is None or ckpt_epoch < max_epochs - 1:
        log.warning(f"{method}/{dataset}: ckpt epoch {ckpt_epoch} < final {max_epochs - 1}; "
                    "still training — writing 'incomplete' (re-run when done).")
        _write_cell(out_dir, method, dataset, seed,
                    {**base, "origin": "incomplete", "epoch": ckpt_epoch, "max_epochs": max_epochs})
        return

    ds_config = get_config(dataset)
    if ds_config.modality != "image":
        raise ValueError(f"Dataset '{dataset}' modality={ds_config.modality!r} not supported.")
    train_transform, val_transform, collate_fn = get_transforms(method, ds_config, cfg.model)
    data, ds_config = create_dataset(dataset, train_transform, val_transform, collate_fn,
                                     cfg.training, data_dir=cfg.get("data_dir"))
    if data.val is None:
        raise ValueError(f"Dataset '{dataset}' has no val split; cannot recover.")

    with open_dict(cfg):
        cfg.model._total_steps = 1
    module, embed_dim = build_module(cfg, ds_config)

    # FULL eval callbacks (linear_probe + knn_probe + rankme + lidar) exactly as
    # training used them, minus LearningRateMonitor (needs an optimizer). The
    # checkpoint restores the backbone, the probe head, and the ordered_queue bank,
    # so validate() reproduces every online metric.
    # The probe head must MATCH the checkpoint being reloaded. module.strict_loading
    # is False below, so a mismatch does not raise -- it silently leaves the probe at
    # random init and reports garbage accuracy. Default "common" matches checkpoints
    # written by the current code (BatchNorm1d+Linear). For a pre-2026-09 checkpoint,
    # whose state_dict has flat linear_probe.{weight,bias} keys, run with
    # SDS_PROBE_PROTOCOL=legacy_linear.
    _probe_protocol = os.environ.get("SDS_PROBE_PROTOCOL", "common")
    callbacks = [cb for cb in create_eval_callbacks(module, ds_config, embed_dim, probe_protocol=_probe_protocol)
                 if not isinstance(cb, LearningRateMonitor)]
    module.strict_loading = False

    trainer = pl.Trainer(
        accelerator="auto", devices=1, logger=False, callbacks=callbacks,
        num_sanity_val_steps=0, enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.validate(module, datamodule=data, ckpt_path=ckpt)

    m = trainer.callback_metrics

    def g(*names) -> float:
        for n in names:
            v = m.get(n)
            if v is not None:
                return round(float(v.item() if hasattr(v, "item") else v), 6)
        return float("nan")

    row = {
        **base, "origin": "online_rerun",
        "linear_top1": g("eval/linear_probe_top1_epoch", "eval/linear_probe_top1"),
        "linear_top5": g("eval/linear_probe_top5_epoch", "eval/linear_probe_top5"),
        "knn_top1": g("eval/knn_probe_top1_epoch", "eval/knn_probe_top1"),
        "knn_top5": g("eval/knn_probe_top5_epoch", "eval/knn_probe_top5"),
        "rankme": g("rankme", "rankme_epoch"),
        "lidar": g("lidar", "lidar_epoch"),
        "rankme_condition_number": g("rankme/condition_number"),
        "rankme_entropy": g("rankme/entropy"),
        "lidar_entropy": g("lidar/entropy"),
        "epoch": ckpt_epoch, "max_epochs": max_epochs,
        "num_classes": ds_config.num_classes, "checkpoint_or_run": ckpt,
    }
    log.info(f"RECOVERED {method}/{dataset}{'' if seed is None else f'/seed{seed}'}: "
             f"linear_top1={row['linear_top1']} knn_top1={row['knn_top1']} "
             f"rankme={row['rankme']} lidar={row['lidar']} (@ epoch {ckpt_epoch})")
    _write_cell(out_dir, method, dataset, seed, row)


if __name__ == "__main__":
    main()
