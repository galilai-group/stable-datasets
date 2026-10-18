"""MNIST-C robustness evaluation for benchmark checkpoints trained on clean MNIST.

Follows the MNIST-C protocol (Mu & Gilmer, 2019): train on clean data, test on
each corruption. The training half is the regular benchmark run with
``dataset=mnistc``, which pretrains on the uncorrupted ``identity`` config and
fits the online linear probe alongside it. This module reloads that run's
backbone and linear probe from its checkpoint, frozen, and scores them on the
clean test set and on the test set of each of the 15 corruptions.

Accuracy uses the same metric as the benchmark's ``eval/linear_probe_top1``
(macro-averaged ``MulticlassAccuracy``) and the same validation transform, so
the clean row reproduces the run's final linear-probe accuracy. Results are
printed and appended to ``benchmarks/results/mnist_c_robustness.csv``.

Usage:
    python -m benchmarks.robustness.mnist_c \\
        --checkpoint <ckpt_dir>/simclr_vit_small_patch16_224_mnistc/last.ckpt
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from benchmarks.dataset import DatasetConfig, _with_data_dirs, get_config
from benchmarks.models import collate_single, create_backbone, get_embedding_dim, val_transform
from stable_datasets.images import MNISTC


log = logging.getLogger(__name__)

DATASET_NAME = "mnistc"
CLEAN_CONFIG = "identity"
CORRUPTIONS: tuple[str, ...] = tuple(c.name for c in MNISTC.BUILDER_CONFIGS if c.name != CLEAN_CONFIG)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "mnist_c_robustness.csv"

_BACKBONE_PREFIX = "backbone."
_PROBE_PREFIX = "callbacks_modules.linear_probe."


# Checkpoint loading


def _sub_state_dict(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_backbone_and_probe(
    checkpoint: str | Path,
    ds_config: DatasetConfig,
    backbone_name: str = "vit_small_patch16_224",
) -> tuple[nn.Module, nn.Linear]:
    """Rebuild the backbone and online linear probe saved by ``benchmarks.run``.

    Both are returned in eval mode with gradients disabled. Raises if the
    checkpoint has no linear probe (e.g. a run on an unlabeled dataset).
    """
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]

    backbone = create_backbone(backbone_name, ds_config)
    backbone.load_state_dict(_sub_state_dict(state_dict, _BACKBONE_PREFIX))

    probe_state = _sub_state_dict(state_dict, _PROBE_PREFIX)
    if not probe_state:
        raise KeyError(f"No linear probe ('{_PROBE_PREFIX}*') found in checkpoint {checkpoint}.")
    probe = nn.Linear(get_embedding_dim(backbone), ds_config.num_classes)
    probe.load_state_dict(probe_state)

    for module in (backbone, probe):
        module.eval().requires_grad_(False)
    return backbone, probe


# Evaluation


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def probe_accuracy(
    backbone: nn.Module,
    probe: nn.Linear,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, int]:
    """Return (macro top-1 accuracy, number of samples) of ``probe(backbone(x))`` over ``loader``."""
    metric = MulticlassAccuracy(num_classes).to(device)
    num_samples = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        metric.update(probe(backbone(images)), labels)
        num_samples += len(labels)
    return float(metric.compute()), num_samples


@dataclass
class RobustnessResult:
    checkpoint: str
    corruption: str
    top1: float
    clean_top1: float
    relative_drop: float
    num_samples: int
    timestamp: str


def evaluate_mnist_c(
    checkpoint: str | Path,
    *,
    backbone_name: str = "vit_small_patch16_224",
    corruptions: tuple[str, ...] = CORRUPTIONS,
    data_dir: str | None = None,
    batch_size: int = 256,
    num_workers: int = 8,
    device: torch.device | None = None,
    smoke_test: bool = False,
) -> list[RobustnessResult]:
    """Score a clean-trained checkpoint on the clean MNIST-C test set and on each corruption.

    The first result is the clean (``identity``) reference; every row carries
    ``clean_top1`` and the relative accuracy drop ``1 - top1 / clean_top1``.
    """
    unknown = sorted(set(corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruptions {unknown}. Available: {list(CORRUPTIONS)}")

    ds_config = get_config(DATASET_NAME)
    if ds_config.builder_kwargs.get("config_name") != CLEAN_CONFIG:
        raise ValueError(f"'{DATASET_NAME}' must train on the '{CLEAN_CONFIG}' config for this protocol.")

    device = device or _default_device()
    backbone, probe = load_backbone_and_probe(checkpoint, ds_config, backbone_name)
    backbone.to(device)
    probe.to(device)

    # Normalize every split with the clean training statistics the model was trained on.
    transform = val_transform(ds_config)
    data_kwargs = _with_data_dirs(ds_config, data_dir)
    max_batches = 3 if smoke_test else None
    timestamp = datetime.now().isoformat(timespec="seconds")

    def _evaluate(config_name: str) -> tuple[float, int]:
        ds = MNISTC(split="test", **{**data_kwargs, "config_name": config_name}).with_transform(transform)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_single,
        )
        top1, n = probe_accuracy(backbone, probe, loader, ds_config.num_classes, device, max_batches)
        log.info(f"{config_name:>15}: top1={top1:.4f} (n={n})")
        return top1, n

    clean_top1, clean_n = _evaluate(CLEAN_CONFIG)
    results = [RobustnessResult(str(checkpoint), CLEAN_CONFIG, clean_top1, clean_top1, 0.0, clean_n, timestamp)]
    for corruption in corruptions:
        top1, n = _evaluate(corruption)
        relative_drop = 1.0 - top1 / clean_top1 if clean_top1 > 0 else float("nan")
        results.append(RobustnessResult(str(checkpoint), corruption, top1, clean_top1, relative_drop, n, timestamp))
    return results


# Reporting


def format_table(results: list[RobustnessResult]) -> str:
    """Render results as a Markdown table with a mean-over-corruptions row."""
    lines = ["| Corruption | Top-1 | Rel. drop |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.corruption} | {r.top1:.4f} | {r.relative_drop:.2%} |")
    corrupted = [r for r in results if r.corruption != CLEAN_CONFIG]
    if corrupted:
        mean_top1 = sum(r.top1 for r in corrupted) / len(corrupted)
        mean_drop = sum(r.relative_drop for r in corrupted) / len(corrupted)
        lines.append(f"| **mean (corrupted)** | {mean_top1:.4f} | {mean_drop:.2%} |")
    return "\n".join(lines)


def append_results(results: list[RobustnessResult], path: Path = RESULTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    log.info(f"Appended {len(results)} rows to {path}")


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="benchmarks.run checkpoint trained with dataset=mnistc")
    parser.add_argument("--backbone", default="vit_small_patch16_224")
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS))
    parser.add_argument("--data-dir", default=os.environ.get("STABLE_DATASETS_ROOT"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=RESULTS_CSV)
    parser.add_argument("--smoke-test", action="store_true", help="evaluate only 3 batches per split")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    results = evaluate_mnist_c(
        args.checkpoint,
        backbone_name=args.backbone,
        corruptions=tuple(args.corruptions),
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )
    print(format_table(results))
    append_results(results, args.output)


if __name__ == "__main__":
    main()
