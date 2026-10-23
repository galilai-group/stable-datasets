"""Small LeJEPA experiment with a frozen linear probe and matched random baseline.

Run from the repository root; see docs/object-datasets-validation.md.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from stable_datasets.images import ETH80, FreiburgGroceries


SPT_REVISION = "ab836bf699a2be712dfcf980a5eb70d36391e876"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def partition_freiburg(train, validation_fraction, split_seed):
    """Deduplicate by source path before stratifying the official training set."""
    first = {}
    labels = train.table.column("label").to_pylist()
    for i, image_id in enumerate(train.table.column("image_id").to_pylist()):
        first.setdefault(image_id, i)
    indices = [first[key] for key in sorted(first)]
    fit, val = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=[labels[i] for i in indices],
    )
    return train.select(sorted(fit)), train.select(sorted(val))


def load_splits(name, cache_root, config):
    kwargs = {"download_dir": cache_root / "downloads", "processed_cache_dir": cache_root / "processed"}
    if name == "eth80":
        return ETH80(**kwargs)
    official = FreiburgGroceries(config_name=config["freiburg_config"], **kwargs)
    train, validation = partition_freiburg(official["train"], config["validation_fraction"], config["split_seed"])
    return {"train": train, "validation": validation, "test": official["test"]}


def split_manifest(splits):
    manifest = {}
    for name, ds in splits.items():
        manifest[name] = [
            {
                "image_id": r["image_id"],
                "label": r["label"],
                **({"object_id": r["object_id"]} if "object_id" in r else {}),
            }
            for r in ds.remove_columns("image").with_format("raw")
        ]
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not {r["image_id"] for r in manifest[a]} & {r["image_id"] for r in manifest[b]}
        if "object_id" in manifest[a][0]:
            assert not {r["object_id"] for r in manifest[a]} & {r["object_id"] for r in manifest[b]}
    return manifest


class Views:
    """Independent crops of one source image; SSL never receives labels."""

    def __init__(self, config):
        def augment(size, scale):
            return T.Compose(
                [
                    T.RandomResizedCrop(size, scale=scale),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )

        self.global_transform = augment(config["image_size"], (0.3, 1.0))
        self.local_transform = augment(config["local_size"], (0.05, 0.3))

    def __call__(self, row):
        return {
            "global_views": [self.global_transform(row["image"]) for _ in range(2)],
            "local_views": [self.local_transform(row["image"]) for _ in range(2)],
        }


class EvaluationTransform:
    def __init__(self, size):
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __call__(self, row):
        return {"image": self.transform(row["image"]), "label": row["label"]}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(ds, config, seed, training=False):
    return DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=training,
        drop_last=training,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"] > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def train_ssl(splits, config, device, out_dir):
    from stable_pretraining.methods.lejepa import LeJEPA

    projector = nn.Sequential(nn.Linear(192, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))
    model = LeJEPA(
        encoder_name="vit_tiny_patch16_224",
        projector=projector,
        n_slices=config["n_slices"],
        n_points=17,
        lamb=0.02,
        pretrained=False,
        drop_path_rate=0.0,
    ).to(device)
    initial_encoder = copy.deepcopy(model.backbone).cpu()
    torch.save(initial_encoder.state_dict(), out_dir / "random-encoder.pt")
    loader = make_loader(splits["train"].with_transform(Views(config)), config, config["seed"], True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)
    epochs = config["epochs"]
    steps_per_epoch = min(len(loader), config["max_batches"]) if config["max_batches"] else len(loader)
    total_steps = epochs * steps_per_epoch
    warmup = min(config["warmup_epochs"] * steps_per_epoch, max(1, total_steps // 2))
    step = 0
    history = []
    start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        totals = np.zeros(3)
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index >= steps_per_epoch:
                break
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total_steps - warmup - 1)
                factor = 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * factor
            output = model(
                global_views=[v.to(device) for v in batch["global_views"]],
                local_views=[v.to(device) for v in batch["local_views"]],
            )
            if not torch.isfinite(output.loss):
                raise RuntimeError("Non-finite LeJEPA loss")
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Non-finite LeJEPA gradients")
            optimizer.step()
            totals += [float(x.detach()) for x in (output.loss, output.inv_loss, output.sigreg_loss)]
            step += 1
        record = dict(zip(("loss", "invariance", "sigreg"), (totals / steps_per_epoch).tolist()))
        record.update(epoch=epoch + 1, steps=step, seconds=time.perf_counter() - epoch_start)
        history.append(record)
        print(json.dumps(record), flush=True)
        write_json(out_dir / "training.json", history)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "loader_rng": loader.generator.get_state(),
            },
            out_dir / "checkpoint.pt",
        )
    return model.backbone, initial_encoder, time.perf_counter() - start


@torch.no_grad()
def embeddings(encoder, ds, config, device):
    encoder.eval().requires_grad_(False)
    loader = make_loader(ds.with_transform(EvaluationTransform(config["image_size"])), config, config["seed"])
    xs, ys = [], []
    for batch in loader:
        xs.append(encoder(batch["image"].to(device)).cpu())
        ys.append(batch["label"].cpu())
    return torch.cat(xs), torch.cat(ys)


def fit_probe(train, validation, num_classes, config):
    """Select weight decay and iteration using validation only, with frozen features."""
    x, y = train
    vx, vy = validation
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / std, (vx - mean) / std
    best_score, best_state, chosen, trace = -1, None, None, []
    for decay in config["probe_weight_decays"]:
        torch.manual_seed(config["seed"] + 1000)
        head = nn.Linear(x.shape[1], num_classes)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=decay)
        for epoch in range(config["probe_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                score = (head(vx).argmax(1) == vy).float().mean().item()
            trace.append({"weight_decay": decay, "epoch": epoch + 1, "validation_accuracy": score})
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(head.state_dict())
                chosen = {"weight_decay": decay, "epoch": epoch + 1, "validation_accuracy": score}
    head.load_state_dict(best_state)
    head.eval()
    return head, mean, std, chosen, trace


def evaluate(encoder, splits, config, device, out_dir, name):
    encoder = encoder.to(device).eval().requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    features = {split: embeddings(encoder, splits[split], config, device) for split in ("train", "validation")}
    classes = splits["train"].features["label"].num_classes
    head, mean, std, chosen, trace = fit_probe(features["train"], features["validation"], classes, config)
    write_json(out_dir / f"{name}-probe-validation.json", trace)
    torch.save(
        {"head": head.state_dict(), "mean": mean, "std": std, "selection": chosen}, out_dir / f"{name}-probe.pt"
    )
    # Test features and labels are accessed only after all probe choices are fixed.
    x, y = embeddings(encoder, splits["test"], config, device)
    with torch.no_grad():
        pred = head((x - mean) / std).argmax(1)
    assert all(torch.equal(before[key], value.cpu()) for key, value in encoder.state_dict().items())
    with (out_dir / f"{name}-test-predictions.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "label", "prediction"])
        for image_id, target, prediction in zip(
            splits["test"].table.column("image_id").to_pylist(), y.tolist(), pred.tolist()
        ):
            writer.writerow([image_id, target, prediction])
    return {
        **chosen,
        "test_accuracy": accuracy_score(y, pred),
        "test_balanced_accuracy": balanced_accuracy_score(y, pred),
        "correct": int((y == pred).sum()),
        "test_count": len(y),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(classes))).tolist(),
    }


def environment():
    packages = [
        "torch",
        "torchvision",
        "stable-pretraining",
        "stable-datasets",
        "timm",
        "numpy",
        "pyarrow",
        "scikit-learn",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "stable_pretraining_reference_commit": SPT_REVISION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eth80", "freiburg"], required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/object_datasets_lejepa.json")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int, default=0, help="Nonzero means smoke test, not a full experiment.")
    parser.add_argument(
        "--skip-probe", action="store_true", help="For timing-only smoke runs; does not evaluate test data."
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(
        dataset=args.dataset,
        seed=args.seed,
        device=args.device,
        max_batches=args.max_batches,
        skip_probe=args.skip_probe,
    )
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["epochs"] < 1 or args.max_batches < 0:
        parser.error("epochs must be positive and max-batches nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "environment.json", environment())
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "requirements-lock.txt").write_text(freeze)
    splits = load_splits(args.dataset, args.cache_root, config)
    write_json(args.output / "splits.json", split_manifest(splits))
    trained, initial, elapsed = train_ssl(splits, config, args.device, args.output)
    result = {
        "ssl_seconds": elapsed,
        "split_counts": {name: len(ds) for name, ds in splits.items()},
        "smoke_test": bool(args.max_batches),
        "seed": args.seed,
        "dataset": args.dataset,
    }
    if not args.skip_probe:
        result["random"] = evaluate(initial, splits, config, args.device, args.output, "random")
        result["lejepa"] = evaluate(trained, splits, config, args.device, args.output, "lejepa")
    write_json(args.output / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
