"""Offline linear probe for off-the-shelf SSL backbones.

Reuses the in-domain benchmark plumbing (``spt.Module`` + ``spt.Manager``
+ ``create_eval_callbacks``) by supplying a frozen-encoder forward
function.  ``OnlineProbe`` then trains its own linear head on top of the
detached features — identical contract to the in-domain runs (same
``eval/linear_probe_top1_epoch`` / ``eval/knn_probe_top1`` keys), so the
results are directly comparable.

Logs are emitted to W&B with ``model="offline_probe"`` and
``backbone=<spec.name>`` so they live in the same project as the
in-domain runs without colliding with the ``vit_small_patch16_224``
runs that ``render_latex.py`` already collects. A persistent record is
also appended to ``benchmarks/results/transfer_results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from stable_pretraining.data import transforms
from torch.utils.data import DataLoader, Dataset

import wandb
from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import collate_single, create_eval_callbacks
from benchmarks.transfer.checkpoints import BACKBONES, BackboneSpec, load_backbone


log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "transfer_results.csv"

# Features land on scratch to stay off the 125 GB home quota.
_DEFAULT_FEATURE_CACHE = Path.home() / "scratch" / ".stable-datasets" / "probe_features"

# spt's CSVLogger can receive None as a metric key (e.g. from OnlineProbe on
# certain datasets), which then crashes _rewrite_with_new_header. Patch it out
# at import time so it's transparent to all callers.
try:
    from lightning.fabric.loggers.csv_logs import _ExperimentWriter

    _orig_log_metrics = _ExperimentWriter.log_metrics

    def _log_metrics_no_none(self, metrics_dict, step=None):
        _orig_log_metrics(self, {k: v for k, v in metrics_dict.items() if k is not None}, step)

    _ExperimentWriter.log_metrics = _log_metrics_no_none
except Exception:
    pass


# Transforms — keyed by the *backbone* spec, not by the dataset, since
# off-the-shelf backbones expect ImageNet-style 224x224 RGB normalized
# with their own training stats.


def _resize_for_eval(spec: BackboneSpec) -> transforms.Compose:
    h, w = spec.image_size
    short_edge = int(round(h * 256 / 224))
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize(short_edge),
        transforms.CenterCrop((h, w)),
        transforms.ToImage(mean=list(spec.mean), std=list(spec.std)),
    )


def _train_transform(spec: BackboneSpec) -> transforms.Compose:
    h, w = spec.image_size
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((h, w), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(mean=list(spec.mean), std=list(spec.std)),
    )


# Forward — frozen-encoder feature extraction.


def _frozen_forward(self, batch, stage):
    """Run the frozen backbone and surface features for OnlineProbe.

    The backbone is forced to eval() (BN/dropout deterministic) and run
    under ``no_grad``. We do not produce a loss: ``OnlineProbe`` adds its
    own CE loss to ``outputs["loss"]`` via ``wrap_forward``, which ends
    up being the only term the optimizer steps on. The dummy main
    optimizer (lr=0 over frozen params) is a no-op.
    """
    self.backbone.eval()
    with torch.no_grad():
        emb = self.backbone(batch["image"])
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    out = {"embedding": emb}
    if "label" in batch:
        out["label"] = batch["label"]
    return out


# Feature pre-extraction (cached offline probe path)


class _EmbeddingDataset(Dataset):
    """In-memory dataset of (embedding, label) pairs returned as dicts."""

    def __init__(self, embs: torch.Tensor, labels: torch.Tensor) -> None:
        self.embs = embs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embs)

    def __getitem__(self, i):
        return {"embedding": self.embs[i], "label": self.labels[i]}


class _FeatureDataModule(pl.LightningDataModule):
    """DataModule backed by pre-extracted embeddings instead of raw images."""

    def __init__(
        self,
        train_embs: torch.Tensor,
        train_lbls: torch.Tensor,
        val_embs: torch.Tensor,
        val_lbls: torch.Tensor,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._train_ds = _EmbeddingDataset(train_embs, train_lbls)
        self._val_ds = _EmbeddingDataset(val_embs, val_lbls)
        self._bs = batch_size

    def train_dataloader(self) -> DataLoader:
        # num_workers=0: tensors are in RAM, no I/O bottleneck, forking overhead not worth it.
        return DataLoader(
            self._train_ds, batch_size=self._bs, shuffle=True,
            drop_last=True, num_workers=0, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self._bs, shuffle=False,
            num_workers=0, pin_memory=True,
        )


def _cached_forward(self, batch, stage):
    """Pass-through when features are pre-computed; batch already has 'embedding' and 'label'."""
    return batch


def _extract_to_disk(
    backbone: torch.nn.Module,
    backbone_name: str,
    dataset_name: str,
    spec: BackboneSpec,
    data_dir: str | None,
    cache_root: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_embs, train_lbls, val_embs, val_lbls), loading from cache if present.

    Always uses the eval transform (no augmentation) for both splits so the
    cache is deterministic and can be shared across seeds.
    """
    cache_dir = cache_root / backbone_name / dataset_name
    train_cache = cache_dir / "train.pt"
    val_cache = cache_dir / "val.pt"

    if train_cache.exists() and val_cache.exists():
        log.info("Loading cached features from %s", cache_dir)
        tr = torch.load(train_cache, weights_only=True)
        va = torch.load(val_cache, weights_only=True)
        return tr["embeddings"], tr["labels"], va["embeddings"], va["labels"]

    cache_dir.mkdir(parents=True, exist_ok=True)

    eval_tf = _resize_for_eval(spec)
    extract_cfg = type("Cfg", (), {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": 4,
    })()
    extract_dm, _ = create_dataset(
        dataset_name, eval_tf, eval_tf, collate_single, extract_cfg, data_dir=data_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device).eval()

    def _run(loader, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        embs, lbls = [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                emb = backbone(img)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                embs.append(emb.cpu())
                if "label" in batch:
                    lbl = batch["label"]
                    lbls.append(lbl.cpu() if torch.is_tensor(lbl) else torch.tensor(lbl))
        embs_t = torch.cat(embs)
        lbls_t = torch.cat(lbls) if lbls else torch.empty(0, dtype=torch.long)
        cache_file = cache_dir / f"{split}.pt"
        # Write to a temp file then rename so concurrent tasks never read a partial file.
        tmp = cache_file.with_suffix(".tmp")
        torch.save({"embeddings": embs_t, "labels": lbls_t}, tmp)
        os.replace(tmp, cache_file)  # atomic on same filesystem
        log.info("Saved %s features → %s (%.1f MB)", split, cache_file, cache_file.stat().st_size / 1e6)
        return embs_t, lbls_t

    log.info("Extracting %s / %s features on %s …", backbone_name, dataset_name, device)
    tr_embs, tr_lbls = _run(extract_dm.train_dataloader(), "train")
    va_embs, va_lbls = _run(extract_dm.val_dataloader(), "val")

    backbone.cpu()
    return tr_embs, tr_lbls, va_embs, va_lbls


# Result tracking


@dataclass
class ProbeResult:
    backbone: str
    dataset: str
    top1: float
    top5: float
    knn_top1: float | None
    max_epochs: int
    batch_size: int
    timestamp: str


def _append_result(row: ProbeResult) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(row))
    log.info(f"Appended result to {RESULTS_CSV}")


def _read_metric(trainer: pl.Trainer, key: str) -> float | None:
    val = trainer.callback_metrics.get(key)
    if val is None:
        return None
    return float(val.detach().cpu().item()) if torch.is_tensor(val) else float(val)


# Trainer


def train_offline_probe(
    backbone_name: str,
    dataset_name: str,
    *,
    max_epochs: int = 90,
    batch_size: int = 256,
    num_workers: int = 8,
    precision: str = "bf16-mixed",
    data_dir: str | None = None,
    feature_cache_dir: str | None = None,
    wandb_enabled: bool = True,
    wandb_entity: str = "samibg",
    wandb_project: str = "finalized-stable-datasets",
    seed: int | None = None,
    smoke_test: bool = False,
) -> ProbeResult:
    """Run an offline linear probe and return the final-epoch metrics.

    Side effects: appends to ``benchmarks/results/transfer_results.csv``
    and (if ``wandb_enabled``) creates a W&B run with
    ``model="offline_probe"`` and ``backbone=<backbone_name>``.
    """
    # Disable spt's auto-loaded checkpoint callbacks (WandbCheckpoint,
    # SklearnCheckpoint, HuggingFaceCheckpointCallback, etc.). The offline
    # probe has a frozen encoder and needs no resume capability, so writing
    # ~700 MB backbone checkpoints per run is pure waste.
    spt.get_config().cache_dir = None

    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    backbone, spec = load_backbone(backbone_name)
    for p in backbone.parameters():
        p.requires_grad_(False)

    ds_config = get_config(dataset_name)
    if ds_config.modality != "image":
        raise ValueError(
            f"Dataset {dataset_name!r} has modality={ds_config.modality!r}; "
            "offline probe only supports images."
        )

    cache_root = Path(feature_cache_dir) if feature_cache_dir else _DEFAULT_FEATURE_CACHE
    tr_embs, tr_lbls, va_embs, va_lbls = _extract_to_disk(
        backbone, backbone_name, dataset_name, spec,
        data_dir, cache_root, batch_size, num_workers,
    )
    data = _FeatureDataModule(tr_embs, tr_lbls, va_embs, va_lbls, batch_size)

    # spt.Module needs an optimizer config even though every backbone
    # param is frozen. SGD lr=0 over backbone.parameters() is a true
    # no-op: grads stay None for frozen params, .step() touches nothing.
    # The OnlineProbe callback owns its own optimizer for the linear head.
    module = spt.Module(
        backbone=backbone,
        forward=_cached_forward,
        optim={
            "optimizer": {"type": "SGD", "lr": 0.0, "weight_decay": 0.0},
            "scheduler": {"type": "ConstantLR"},
            "interval": "step",
        },
        hparams={
            "model": "offline_probe",
            "backbone": backbone_name,
            "dataset": dataset_name,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        },
    )

    callbacks = create_eval_callbacks(module, ds_config, spec.embed_dim)

    logger: bool | WandbLogger = True
    if wandb_enabled and not smoke_test:
        run_name = f"offline_probe_{backbone_name}_{dataset_name}"
        if seed is not None:
            run_name += f"_seed{seed}"
        logger = WandbLogger(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            id=wandb.util.generate_id(),
            log_model=False,
            save_dir=os.getcwd(),
            tags=["offline_probe", "transfer"]
            + ([f"seed:{seed}"] if seed is not None else []),
            config={
                "model": "offline_probe",
                "backbone": backbone_name,
                "backbone_display": spec.display_name,
                "dataset": dataset_name,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "embed_dim": spec.embed_dim,
                "seed": seed,
            },
        )

    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else max_epochs,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=3 if smoke_test else 1.0,
        limit_val_batches=3 if smoke_test else 1.0,
        accelerator="auto",
    )

    try:
        manager = spt.Manager(trainer=trainer, module=module, data=data)
        manager()
        top1 = _read_metric(trainer, "eval/linear_probe_top1_epoch") or 0.0
        top5 = _read_metric(trainer, "eval/linear_probe_top5_epoch") or 0.0
        knn = _read_metric(trainer, "eval/knn_probe_top1")
    finally:
        if wandb_enabled and not smoke_test:
            wandb.finish()

    result = ProbeResult(
        backbone=backbone_name,
        dataset=dataset_name,
        top1=top1,
        top5=top5,
        knn_top1=knn,
        max_epochs=max_epochs,
        batch_size=batch_size,
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
    )
    if not smoke_test:
        _append_result(result)
    return result


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    # bf16-mixed (not 16-mixed) on purpose: 16-mixed uses GradScaler, which
    # asserts at least one optimizer param produced an inf check. The dummy
    # main optimizer over the frozen backbone never sees a grad, so GradScaler
    # crashes. bf16 has no scaler.
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--data-dir", default="/oscar/home/sboughan/scratch/.stable-datasets")
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Directory for pre-extracted backbone features (default: ~/scratch/.stable-datasets/probe_features).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="samibg")
    parser.add_argument("--wandb-project", default="finalized-stable-datasets")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    result = train_offline_probe(
        backbone_name=args.backbone,
        dataset_name=args.dataset,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        data_dir=args.data_dir,
        feature_cache_dir=args.feature_cache_dir,
        wandb_enabled=not args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    print(
        f"\n{result.backbone} | {result.dataset} | "
        f"top1={result.top1*100:.2f} top5={result.top5*100:.2f} "
        f"knn1={(result.knn_top1 or 0)*100:.2f}"
    )


if __name__ == "__main__":
    main()
