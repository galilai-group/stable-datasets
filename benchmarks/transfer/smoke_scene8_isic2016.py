"""Bounded CPU optimization check with the pinned stable-pretraining SimCLR API.

This checks data/augmentation/forward/loss/backward/update execution. Eight
small-batch steps are not evidence of converged representation learning.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from benchmarks.transfer.scene8_isic2016 import SourceSamples, environment, file_hash, read_protocols, write_json
from stable_datasets.images import ISIC2016, Scene8


REVISION = "ab836bf699a2be712dfcf980a5eb70d36391e876"


def run_smoke(raw_dir, cache_dir, protocol_dir, output, steps=8):
    """Run a fresh tiny encoder on fitting images only; targets are unused."""
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/project1/matplotlib").resolve()))
    from stable_pretraining.methods import SimCLR

    distribution = importlib.metadata.distribution("stable-pretraining")
    direct = json.loads(distribution.read_text("direct_url.json"))
    if direct.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("The installed stable-pretraining revision is not the declared pin.")
    records, manifests = read_protocols(protocol_dir)
    scene = Scene8(split="train", archive_path=Path(raw_dir) / "scene8.zip", processed_cache_dir=cache_dir)
    isic = ISIC2016(data_dir=raw_dir, processed_cache_dir=cache_dir)
    sources = {
        "scene8": SourceSamples([scene], records["scene8"]),
        "isic2016": SourceSamples([isic["train"], isic["test"]], records["isic2016"]),
    }
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(64, scale=(0.2, 1), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    report = {
        "purpose": "Known-method execution smoke only; not converged SSL training.",
        "dependency": {
            "version": distribution.version,
            "commit": REVISION,
            "method": "stable_pretraining.methods.SimCLR",
            "method_file_sha256": file_hash(inspect.getfile(SimCLR)),
        },
        "environment": environment(),
        "runner_sha256": file_hash(__file__),
        "seed": 20261018,
        "batch_size": 8,
        "steps": steps,
        "target_labels_used": False,
        "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
        "augmentation": repr(augmentation),
        "datasets": {},
    }
    for name, source in sources.items():
        random.seed(20261018)
        np.random.seed(20261018)
        torch.manual_seed(20261018)
        encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        model = SimCLR(encoder_name=encoder, projector_dims=(128, 32), temperature=0.5, pretrained=False).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        before = {key: value.detach().clone() for key, value in model.named_parameters()}
        protocol = "scene8-pixel-groups-v1" if name == "scene8" else "isic2016-train-filtered-pixel-groups-v2"
        parent = manifests[(protocol, 42)]
        fit_ids = parent["roles"]["fit"]
        pool_ids = [fit_ids[i] for i in np.linspace(0, len(fit_ids) - 1, 32, dtype=int)]
        lookup = {r.image_id: i for i, r in enumerate(source.records)}
        trace = []
        started = time.perf_counter()
        for step in range(steps):
            ids = [pool_ids[(step * 8 + index) % len(pool_ids)] for index in range(8)]
            first, second = [], []
            for image_id in ids:
                image = source[lookup[image_id]]["image"].convert("RGB")
                first.append(augmentation(image))
                second.append(augmentation(image))
            optimizer.zero_grad(set_to_none=True)
            result = model(torch.stack(first), torch.stack(second))
            loss = result.loss
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Nonfinite or nonscalar contrastive loss.")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("Missing or nonfinite model gradients.")
            norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients)))
            if norm <= 0:
                raise ValueError("All contrastive gradients are zero.")
            optimizer.step()
            trace.append({"step": step + 1, "image_ids": ids, "loss": float(loss.detach()), "gradient_norm": norm})
        changed = {key: float((value.detach() - before[key]).norm()) for key, value in model.named_parameters()}
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()) or not any(
            value > 0 for value in changed.values()
        ):
            raise ValueError("Invalid or unchanged optimized parameters.")
        if not any(value > 0 for key, value in changed.items() if key.startswith("backbone.")):
            raise ValueError("Encoder parameters did not update.")
        report["datasets"][name] = {
            "protocol": protocol,
            "partition_seed": 42,
            "pool_ids": pool_ids,
            "trace": trace,
            "parameter_change_norms": changed,
            "seconds": time.perf_counter() - started,
            "all_losses_and_gradients_finite": True,
            "encoder_parameters_changed": True,
        }
    write_json(output, report)
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(".cache/project1/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/project1/processed"))
    parser.add_argument("--protocol-dir", type=Path, default=Path(".cache/project1/verified-v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/source/datasets/experiments/scene8_isic2016/training_smoke.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = run_smoke(args.raw_dir, args.cache_dir, args.protocol_dir, args.output)
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in result.items()
                    if key in ("seconds", "encoder_parameters_changed", "all_losses_and_gradients_finite")
                }
                for name, result in report["datasets"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
