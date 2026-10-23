"""Train small supervised baselines for the digit datasets in this repository.

This example follows the repository's ``stable-pretraining`` pattern with an
``spt.Module``, ``spt.data.DataModule``, ``spt.Manager``, and Lightning trainer.
It has a dedicated entry point because ``benchmarks/run.py`` currently rejects
non-image datasets, while FSDD needs a length-aware audio model and collator.
Install the ``benchmarks`` extra before running the experiment.

Examples::

    python examples/validate_digits.py --dataset sign-language-digits
    python examples/validate_digits.py --dataset fsdd --epochs 15
    python examples/validate_digits.py --dataset both --smoke-test

Each dataset gets its own output directory containing split manifests, the best
validation checkpoint, ``metrics.csv``, and a complete ``metrics.json`` record.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


SIGN_LANGUAGE_DIGITS_REVISION = "4b09433ef9d5c60602e8805bb54b8aa2428527f8"
FSDD_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring a trainer framework."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, np.ndarray]:
    """Return deterministic class-stratified train/validation/test indices.

    Counts are allocated independently per class with the largest-remainder
    method, so every input row appears exactly once even when class sizes are
    not divisible by ten.
    """
    labels_array = np.asarray(labels)
    ratios_array = np.asarray(ratios, dtype=np.float64)
    split_names = ("train", "validation", "test")
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if ratios_array.shape != (3,) or np.any(ratios_array <= 0):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(ratios_array.sum(), 1.0):
        raise ValueError("ratios must sum to 1")

    rng = np.random.default_rng(seed)
    split_parts: dict[str, list[np.ndarray]] = {name: [] for name in split_names}
    for label in sorted(np.unique(labels_array).tolist()):
        class_indices = np.flatnonzero(labels_array == label)
        rng.shuffle(class_indices)
        exact_counts = ratios_array * len(class_indices)
        counts = np.floor(exact_counts).astype(np.int64)
        remainder = len(class_indices) - int(counts.sum())
        # Stable tie-breaking keeps the result reproducible across NumPy versions.
        remainder_order = sorted(range(3), key=lambda i: (-(exact_counts[i] - counts[i]), i))
        for split_index in remainder_order[:remainder]:
            counts[split_index] += 1

        start = 0
        for name, count in zip(split_names, counts.tolist()):
            split_parts[name].append(class_indices[start : start + count])
            start += count

    result = {}
    for name in split_names:
        indices = np.concatenate(split_parts[name]).astype(np.int64, copy=False)
        rng.shuffle(indices)
        result[name] = indices
    return result


def fsdd_split_indices(
    official_train_utterance_ids: Sequence[int] | np.ndarray,
    official_test_utterance_ids: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the experiment split without moving examples across official splits."""
    train_ids = np.asarray(official_train_utterance_ids, dtype=np.int64)
    test_ids = np.asarray(official_test_utterance_ids, dtype=np.int64)
    if train_ids.ndim != 1 or test_ids.ndim != 1:
        raise ValueError("utterance ids must be one-dimensional")
    if train_ids.size == 0 or test_ids.size == 0:
        raise ValueError("official train and test splits must both be non-empty")
    if np.any((train_ids < 5) | (train_ids > 49)):
        raise ValueError("FSDD official train must contain only utterance ids 5 through 49")
    if np.any((test_ids < 0) | (test_ids > 4)):
        raise ValueError("FSDD official test must contain only utterance ids 0 through 4")

    result = {
        "train": np.flatnonzero((train_ids >= 10) & (train_ids <= 49)).astype(np.int64),
        "validation": np.flatnonzero((train_ids >= 5) & (train_ids <= 9)).astype(np.int64),
        "test": np.arange(test_ids.size, dtype=np.int64),
    }
    if any(indices.size == 0 for indices in result.values()):
        raise ValueError("the requested FSDD train/validation/test partition produced an empty split")
    return result


def conv1d_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Compute valid sequence lengths after a Conv1d layer."""
    return (
        torch.div(
            lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
            stride,
            rounding_mode="floor",
        )
        + 1
    )


def audio_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable ``[T, 1]`` waveforms into ``[B, 1, T_max]``."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    waveforms: list[torch.Tensor] = []
    labels: list[int] = []
    for sample in samples:
        waveform = torch.as_tensor(sample["series"], dtype=torch.float32)
        if waveform.ndim != 2 or waveform.shape[1] != 1:
            raise ValueError(f"expected each waveform to have shape [T, 1], got {tuple(waveform.shape)}")
        if waveform.shape[0] == 0:
            raise ValueError("waveforms must be non-empty")
        if not torch.isfinite(waveform).all():
            raise ValueError("waveforms must contain only finite values")
        waveforms.append(waveform)
        labels.append(int(sample["label"]))

    lengths = torch.tensor([waveform.shape[0] for waveform in waveforms], dtype=torch.long)
    padded = torch.zeros((len(waveforms), 1, int(lengths.max())), dtype=torch.float32)
    for row, waveform in enumerate(waveforms):
        padded[row, 0, : waveform.shape[0]] = waveform[:, 0]
    return {
        "series": padded,
        "lengths": lengths,
        "label": torch.tensor(labels, dtype=torch.long),
    }


class TinyImageCNN(nn.Module):
    """Small 2D CNN suitable for 100x100 RGB digit-sign images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(image).flatten(1))


class TinyAudioCNN(nn.Module):
    """Small 1D CNN with valid-length propagation and masked pooling."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(1, 16, kernel_size=9, stride=4, padding=4),
                nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, series: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if series.ndim != 3 or series.shape[1] != 1:
            raise ValueError(f"expected audio input [B, 1, T], got {tuple(series.shape)}")
        if lengths.ndim != 1 or lengths.shape[0] != series.shape[0]:
            raise ValueError("lengths must contain one value per waveform")

        features = series
        valid_lengths = lengths
        for convolution in self.convolutions:
            features = F.relu(convolution(features))
            valid_lengths = conv1d_output_lengths(
                valid_lengths,
                kernel_size=convolution.kernel_size[0],
                stride=convolution.stride[0],
                padding=convolution.padding[0],
                dilation=convolution.dilation[0],
            )
            # A biased convolution maps right-padding zeros to non-zero values.
            # Clear them at every layer so they cannot leak back into valid
            # boundary positions in the next convolution.
            positions = torch.arange(features.shape[-1], device=features.device)[None, :]
            layer_mask = positions < valid_lengths[:, None]
            features = features * layer_mask[:, None, :]
        valid_lengths = valid_lengths.clamp(min=1, max=features.shape[-1])
        positions = torch.arange(features.shape[-1], device=features.device)[None, :]
        mask = positions < valid_lengths[:, None]
        masked_sum = (features * mask[:, None, :]).sum(dim=-1)
        masked_mean = masked_sum / valid_lengths[:, None].to(features.dtype)
        masked_max = features.masked_fill(~mask[:, None, :], -torch.inf).amax(dim=-1)
        return self.classifier(torch.cat((masked_mean, masked_max), dim=1))


def _image_to_tensor(image: Any, size: int) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)
        if tensor.ndim != 3:
            raise ValueError(f"expected a 2D or 3D image, got {tuple(tensor.shape)}")
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        elif tensor.shape[0] not in (1, 3):
            raise ValueError(f"could not infer image channel dimension from {tuple(tensor.shape)}")
        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return ((tensor - 0.5) / 0.5).contiguous()


class _ImageAdapter(Dataset):
    def __init__(self, dataset: Any, image_size: int):
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        return {
            "image": _image_to_tensor(sample["image"], self.image_size),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }


class _AudioAdapter(Dataset):
    def __init__(self, dataset: Any, amplitude_scale: float = 32768.0):
        self.dataset = dataset.with_format("numpy") if hasattr(dataset, "with_format") else dataset
        self.amplitude_scale = amplitude_scale

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        series = np.asarray(sample["series"], dtype=np.float32) / self.amplitude_scale
        return {"series": series, "label": int(sample["label"])}


def _metadata_columns(dataset: Any, names: Sequence[str]) -> dict[str, list[Any]]:
    """Read metadata without decoding binary payload columns when possible."""
    try:
        if getattr(dataset, "_indices", None) is None:
            table = dataset.table
            if table.num_rows == len(dataset):
                return {name: table[name].to_pylist() for name in names}
    except (AttributeError, KeyError, TypeError):
        pass
    rows = [dataset[index] for index in range(len(dataset))]
    return {name: [row[name] for row in rows] for name in names}


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["epoch", "split", "loss", "accuracy", "samples", "elapsed_seconds", "selected"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str | None]:
    import stable_datasets

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": _package_version("pillow"),
        "torch": torch.__version__,
        "stable_datasets": getattr(stable_datasets, "__version__", _package_version("stable-datasets")),
        "stable_pretraining": _package_version("stable-pretraining"),
        "lightning": _package_version("lightning"),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


@dataclass
class TrainConfig:
    dataset: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    max_train_batches: int | None
    max_eval_batches: int | None
    smoke_test: bool


def _require_training_stack():
    try:
        import lightning as pl
        import stable_pretraining as spt
    except ImportError as error:
        raise RuntimeError(
            "validate_digits requires Lightning and stable-pretraining; "
            "install this repository with `pip install -e '.[benchmarks]'`"
        ) from error
    return pl, spt


def supervised_digit_forward(self, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    """Forward function bound to ``spt.Module`` for both modalities."""
    labels = batch["label"]
    if "image" in batch:
        logits = self.network(batch["image"])
    else:
        logits = self.network(batch["series"], batch["lengths"])
    loss = self.supervised_loss(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    split_name = {"fit": "train", "validate": "validation", "test": "test"}.get(stage, stage)
    batch_size = labels.shape[0]
    self.log(
        f"{split_name}/loss",
        loss,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=batch_size,
    )
    self.log(
        f"{split_name}/accuracy",
        accuracy,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=batch_size,
        prog_bar=split_name == "validation",
    )
    return {"logits": logits, "label": labels, "loss": loss}


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name not in metrics:
        raise RuntimeError(f"Lightning did not expose expected epoch metric {name!r}; got {sorted(metrics)}")
    value = metrics[name]
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _make_history_callback(pl, started: float):
    class HistoryCallback(pl.Callback):
        def __init__(self):
            self.history: list[dict[str, Any]] = []
            self.test_samples = 0
            self._train_samples = 0
            self._validation_samples = 0
            self._pending_validation: dict[str, float | int] | None = None

        def on_train_epoch_start(self, trainer, pl_module):
            self._train_samples = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._train_samples += int(batch["label"].shape[0])

        def on_validation_epoch_start(self, trainer, pl_module):
            self._validation_samples = 0

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self._validation_samples += int(batch["label"].shape[0])

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            self._pending_validation = {
                "loss": _metric_value(metrics, "validation/loss"),
                "accuracy": _metric_value(metrics, "validation/accuracy"),
                "samples": self._validation_samples,
            }

        def on_train_epoch_end(self, trainer, pl_module):
            if self._pending_validation is None:
                raise RuntimeError("training epoch ended before validation metrics were captured")
            metrics = trainer.callback_metrics
            self.history.append(
                {
                    "epoch": trainer.current_epoch + 1,
                    "elapsed_seconds": time.perf_counter() - started,
                    "train": {
                        "loss": _metric_value(metrics, "train/loss"),
                        "accuracy": _metric_value(metrics, "train/accuracy"),
                        "samples": self._train_samples,
                    },
                    "validation": self._pending_validation,
                }
            )
            self._pending_validation = None

        def on_test_start(self, trainer, pl_module):
            self.test_samples = 0

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.test_samples += int(batch["label"].shape[0])

    return HistoryCallback()


def train_and_evaluate(
    *,
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    output_dir: Path,
    train_config: TrainConfig,
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train through stable-pretraining and evaluate the best validation checkpoint."""
    pl, spt = _require_training_stack()
    from lightning.pytorch.callbacks import ModelCheckpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    # Configure spt before constructing Trainer: Lightning discovers spt's
    # default callbacks during Trainer.__init__.
    spt.get_config().cache_dir = None
    spt.set(
        requeue_checkpoint=False,
        progress_bar="none",
        default_loggers={"registry": False},
        default_callbacks={
            "progress_bar": False,
            "registry": False,
            "logging": False,
            "env_dump": False,
            "trainer_info": False,
            "sklearn_checkpoint": False,
            "wandb_checkpoint": False,
            "trackio_checkpoint": False,
            "swanlab_checkpoint": False,
            "module_summary": False,
            "slurm_info": False,
            "unused_params": False,
            "hf_checkpoint": False,
        },
    )
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    history_callback = _make_history_callback(pl, start_time)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-validation",
        monitor="validation/accuracy",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    train_batches_per_epoch = len(loaders["train"])
    if train_config.max_train_batches is not None:
        train_batches_per_epoch = min(train_batches_per_epoch, train_config.max_train_batches)
    scheduler_config = {
        "type": "CosineAnnealingLR",
        "T_max": max(1, train_batches_per_epoch * train_config.epochs),
    }
    lightning_module = spt.Module(
        network=model,
        supervised_loss=nn.CrossEntropyLoss(),
        forward=supervised_digit_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": train_config.learning_rate,
                "weight_decay": train_config.weight_decay,
            },
            "scheduler": scheduler_config,
            "interval": "step",
        },
        hparams={
            "dataset": train_config.dataset,
            "seed": train_config.seed,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
        },
    )
    data = spt.data.DataModule(
        train=loaders["train"],
        val=loaders["validation"],
        test=loaders["test"],
    )
    accelerator = "gpu" if train_config.device == "cuda" else train_config.device
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=train_config.epochs,
        callbacks=[history_callback, checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
        default_root_dir=output_dir,
        log_every_n_steps=1,
        limit_train_batches=train_config.max_train_batches or 1.0,
        limit_val_batches=train_config.max_eval_batches or 1.0,
        limit_test_batches=train_config.max_eval_batches or 1.0,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=lightning_module,
        data=data,
        seed=train_config.seed,
        weights_only=False,
    )
    manager()
    if not checkpoint_callback.best_model_path:
        raise RuntimeError("training finished without producing a best-validation checkpoint")
    test_outputs = trainer.test(
        lightning_module,
        datamodule=data,
        ckpt_path=checkpoint_callback.best_model_path,
        verbose=False,
    )
    if len(test_outputs) != 1:
        raise RuntimeError(f"expected one test result, got {len(test_outputs)}")
    test_metrics = {
        "loss": _metric_value(test_outputs[0], "test/loss"),
        "accuracy": _metric_value(test_outputs[0], "test/accuracy"),
        "samples": history_callback.test_samples,
    }
    history = history_callback.history
    if len(history) != train_config.epochs:
        raise RuntimeError(f"expected {train_config.epochs} history rows, got {len(history)}")
    best_record = max(history, key=lambda row: row["validation"]["accuracy"])
    best_epoch = int(best_record["epoch"])
    best_validation = best_record["validation"]
    runtime_seconds = time.perf_counter() - start_time
    metric_rows: list[dict[str, Any]] = []
    for record in history:
        for split_name in ("train", "validation"):
            metric_rows.append(
                {
                    "epoch": record["epoch"],
                    "split": split_name,
                    **record[split_name],
                    "elapsed_seconds": round(record["elapsed_seconds"], 6),
                    "selected": split_name == "validation" and record["epoch"] == best_epoch,
                }
            )
    metric_rows.append(
        {
            "epoch": best_epoch,
            "split": "test",
            **test_metrics,
            "elapsed_seconds": round(runtime_seconds, 6),
            "selected": True,
        }
    )

    results = {
        "config": {"training": asdict(train_config), "experiment": dict(experiment_config)},
        "selection": {
            "metric": "validation/accuracy",
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        "test": test_metrics,
        "history": history,
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": runtime_seconds,
            "device": train_config.device,
            "platform": platform.platform(),
            "command": [Path(sys.executable).name, *sys.argv],
        },
        "versions": _versions(),
        "engine": {
            "module": "stable_pretraining.Module",
            "data": "stable_pretraining.data.DataModule",
            "runner": "stable_pretraining.Manager",
            "trainer": "lightning.Trainer",
            "optimizer": "AdamW",
            "scheduler": scheduler_config,
        },
        "model": {
            "class": type(model).__name__,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "artifacts": {
            "checkpoint": str(Path(checkpoint_callback.best_model_path).resolve().relative_to(output_dir.resolve())),
            "metrics_csv": "metrics.csv",
        },
    }
    _write_metrics_csv(output_dir / "metrics.csv", metric_rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"{train_config.dataset} selected epoch {best_epoch} through stable-pretraining; "
        f"test loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}"
    )
    return results


def _builder_cache_kwargs(cache_dir: Path | None) -> dict[str, str]:
    if cache_dir is None:
        return {}
    return {
        "processed_cache_dir": str(cache_dir / "processed"),
        "download_dir": str(cache_dir / "downloads"),
    }


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    collate_fn=None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def run_sign_language_digits(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from stable_datasets.images import SignLanguageDigits

    dataset = SignLanguageDigits(split="all", **_builder_cache_kwargs(args.cache_dir))
    metadata = _metadata_columns(dataset, ("filename", "label"))
    labels = np.asarray(metadata["label"], dtype=np.int64)
    split_indices = stratified_split_indices(labels, seed=args.seed)
    output_dir = args.output_dir / "sign_language_digits"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_datasets = {}
    split_counts = {}
    class_counts = {}
    for split_name, indices in split_indices.items():
        split_counts[split_name] = int(indices.size)
        class_counts[split_name] = np.bincount(labels[indices], minlength=10).astype(int).tolist()
        manifest_rows = [
            {
                "dataset_index": int(index),
                "filename": metadata["filename"][index],
                "label": int(labels[index]),
            }
            for index in sorted(indices.tolist(), key=lambda i: metadata["filename"][i])
        ]
        _write_manifest(output_dir / f"manifest_{split_name}.csv", manifest_rows)
        split_datasets[split_name] = _ImageAdapter(dataset.select(indices), args.image_size)

    loaders = {
        name: _make_loader(
            adapted,
            batch_size=args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
            num_workers=args.num_workers,
        )
        for name, adapted in split_datasets.items()
    }
    config = TrainConfig(
        dataset="sign-language-digits",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=str(device),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        smoke_test=args.smoke_test,
    )
    experiment_config = {
        "source_revision": SIGN_LANGUAGE_DIGITS_REVISION,
        "source_split": "all",
        "split_definition": "per-class largest-remainder allocation with ratios 0.8/0.1/0.1",
        "split_counts": split_counts,
        "class_counts": class_counts,
        "image_preprocessing": {
            "convert": "RGB",
            "resize": [args.image_size, args.image_size],
            "value_range": "uint8 [0,255] -> float32 [-1,1]",
            "augmentation": None,
        },
        "manifests": {name: f"manifest_{name}.csv" for name in split_indices},
    }
    return train_and_evaluate(
        model=TinyImageCNN(),
        loaders=loaders,
        output_dir=output_dir,
        train_config=config,
        experiment_config=experiment_config,
    )


def run_fsdd(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from stable_datasets.timeseries import FSDD

    cache_kwargs = _builder_cache_kwargs(args.cache_dir)
    official_train = FSDD(split="train", **cache_kwargs)
    official_test = FSDD(split="test", **cache_kwargs)
    fields = ("filename", "label", "speaker_id", "utterance_id", "sample_rate")
    train_metadata = _metadata_columns(official_train, fields)
    test_metadata = _metadata_columns(official_test, fields)
    if set(train_metadata["sample_rate"]) != {8000} or set(test_metadata["sample_rate"]) != {8000}:
        raise ValueError("FSDD experiment expects all decoded WAV headers to report 8000 Hz")
    split_indices = fsdd_split_indices(train_metadata["utterance_id"], test_metadata["utterance_id"])
    source_datasets = {"train": official_train, "validation": official_train, "test": official_test}
    source_metadata = {"train": train_metadata, "validation": train_metadata, "test": test_metadata}
    output_dir = args.output_dir / "fsdd"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_datasets = {}
    split_counts = {}
    class_counts = {}
    for split_name, indices in split_indices.items():
        metadata = source_metadata[split_name]
        labels = np.asarray(metadata["label"], dtype=np.int64)
        split_counts[split_name] = int(indices.size)
        class_counts[split_name] = np.bincount(labels[indices], minlength=10).astype(int).tolist()
        source_split = "test" if split_name == "test" else "train"
        manifest_rows = [
            {
                "source_split": source_split,
                "source_index": int(index),
                "filename": metadata["filename"][index],
                "label": int(metadata["label"][index]),
                "speaker_id": metadata["speaker_id"][index],
                "utterance_id": int(metadata["utterance_id"][index]),
                "sample_rate": int(metadata["sample_rate"][index]),
            }
            for index in sorted(indices.tolist(), key=lambda i: metadata["filename"][i])
        ]
        _write_manifest(output_dir / f"manifest_{split_name}.csv", manifest_rows)
        selected = source_datasets[split_name].select(indices)
        split_datasets[split_name] = _AudioAdapter(selected)

    loaders = {
        name: _make_loader(
            adapted,
            batch_size=args.batch_size,
            shuffle=name == "train",
            seed=args.seed,
            num_workers=args.num_workers,
            collate_fn=audio_collate,
        )
        for name, adapted in split_datasets.items()
    }
    config = TrainConfig(
        dataset="fsdd",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=str(device),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        smoke_test=args.smoke_test,
    )
    experiment_config = {
        "source_revision": FSDD_REVISION,
        "official_splits": {"train": "utterance index 5-49", "test": "utterance index 0-4"},
        "experiment_split": {
            "train": "official train, utterance index 10-49",
            "validation": "official train, utterance index 5-9",
            "test": "official test, utterance index 0-4",
            "speaker_independent": False,
        },
        "split_counts": split_counts,
        "class_counts": class_counts,
        "audio_preprocessing": {
            "input_layout": "[T, 1]",
            "batch_layout": "[B, 1, T_max]",
            "sample_rate_hz": 8000,
            "amplitude": "builder float32 PCM counts divided by 32768.0 before the model",
            "pooling": "masked mean and max after propagated convolution lengths",
            "cropping": None,
        },
        "manifests": {name: f"manifest_{name}.csv" for name in split_indices},
    }
    return train_and_evaluate(
        model=TinyAudioCNN(),
        loaders=loaders,
        output_dir=output_dir,
        train_config=config,
        experiment_config=experiment_config,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("sign-language-digits", "fsdd", "both"),
        default="both",
        help="dataset baseline to run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-runs/digit-validation"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one epoch with at most three batches per split; not a reportable experiment",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size, and image-size must be positive; num-workers must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        parser.error("max-train-batches must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        parser.error("max-eval-batches must be positive")
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 3
        args.max_eval_batches = 3
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_training_stack()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}; stable-pretraining={_package_version('stable-pretraining') or 'not installed'}")
    if args.dataset in ("sign-language-digits", "both"):
        run_sign_language_digits(args, device)
    if args.dataset in ("fsdd", "both"):
        # Resetting makes model initialization independent of which run came first.
        seed_everything(args.seed)
        run_fsdd(args, device)


if __name__ == "__main__":
    main()
