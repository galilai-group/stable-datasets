"""Train a linear probe on ImageNet-Sketch with frozen ImageNet-1K features.

Features come from a pretrained timm model (timm's eval transform, no augmentation). The probe is
trained on a per-class train split and evaluated on the held-out val split, next to the model's
own ImageNet-1K head on the same images. Some sketches appear more than once: the authors filled
small classes with flipped and rotated copies, and some drawings were scraped twice (e.g. with and
without a watermark). Near-duplicates are grouped and kept on the same side of the split.

Usage:
    python examples/probe_imagenet_sketch.py --model resnet50.tv_in1k
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stable_datasets.images import ImageNetSketch


THUMB_SIZE = 32


class ProbeDataset(Dataset):
    """Rows as ``(transform(image), label, grayscale thumbnail)``; module-level so workers can pickle it."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = row["image"].convert("RGB")
        thumb = np.asarray(image.convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR))
        return self.transform(image), row["label"], torch.from_numpy(thumb.copy())


@torch.inference_mode()
def extract(model, loader):
    """Pooled features, ImageNet-1K head predictions, labels and thumbnails for every row."""
    features, head_top1, labels, thumbs = [], [], [], []
    for images, batch_labels, batch_thumbs in tqdm(loader, desc="features"):
        feats = model.forward_head(model.forward_features(images), pre_logits=True)
        features.append(feats.numpy())
        head_top1.append(model.get_classifier()(feats).argmax(dim=1).numpy())
        labels.append(batch_labels.numpy())
        thumbs.append(batch_thumbs.numpy())
    return {
        "features": np.concatenate(features),
        "head_top1": np.concatenate(head_top1),
        "labels": np.concatenate(labels),
        "thumbs": np.concatenate(thumbs),
    }


def _dihedral(x):
    """The 8 flips/rotations of a batch of square images, shape (8, n, h, w)."""
    rotations = [np.rot90(x, k, axes=(1, 2)) for k in range(4)]
    return np.stack(rotations + [r[:, :, ::-1] for r in rotations])


def duplicate_groups(thumbs, labels, threshold):
    """Group id per row; rows of one class whose thumbnails match up to a flip/rotation share an id."""
    x = thumbs.astype(np.float32)
    x -= x.mean(axis=(1, 2), keepdims=True)
    x /= np.linalg.norm(x.reshape(len(x), -1), axis=1)[:, None, None] + 1e-8
    parent = np.arange(len(x))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        base = x[idx].reshape(len(idx), -1)
        variants = _dihedral(x[idx]).reshape(8, len(idx), -1)
        similarity = np.max(variants @ base.T, axis=0)
        np.fill_diagonal(similarity, -1)
        for i, j in zip(*np.nonzero(similarity > threshold)):
            parent[find(idx[i])] = find(idx[j])
    return np.array([find(i) for i in range(len(x))])


def split_by_group(labels, groups, val_fraction, seed):
    """Boolean val mask: per class, whole groups go to val while they fit in ``val_fraction`` of the class.

    A group larger than the remaining quota stays in train, so every class keeps training rows; a
    class made of one repeated drawing has no val rows.
    """
    rng = np.random.default_rng(seed)
    val = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        class_groups = np.unique(groups[idx])
        rng.shuffle(class_groups)
        quota = round(val_fraction * len(idx))
        for group in class_groups:
            members = idx[groups[idx] == group]
            if val[idx].sum() + len(members) <= quota:
                val[members] = True
    return val


def train_probe(x, y, weight_decay, num_classes, max_iter):
    probe = torch.nn.Linear(x.shape[1], num_classes)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(x), y) + weight_decay * probe.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


@torch.inference_mode()
def accuracy(probe, x, y, k=1):
    topk = probe(x).topk(k, dim=1).indices
    return 100 * (topk == y[:, None]).any(dim=1).float().mean().item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probe on ImageNet-Sketch with frozen timm features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="resnet50.tv_in1k", help="timm model name")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of each class held out for val")
    parser.add_argument("--duplicate-threshold", type=float, default=0.8, help="thumbnail cosine for a copy")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--max-iter", type=int, default=300, help="L-BFGS iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--features", type=Path, help="cache file for extracted features (.npz)")
    parser.add_argument("--out", type=Path, default=Path("imagenet_sketch_probe.json"), help="JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import timm
        import timm.data
    except ImportError:
        sys.exit("This script needs timm: pip install timm")

    start = time.perf_counter()
    torch.manual_seed(args.seed)
    if args.features is not None and args.features.is_file():
        data = dict(np.load(args.features))
    else:
        model = timm.create_model(args.model, pretrained=True).eval()
        transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
        ds = ImageNetSketch(split="test")
        loader = DataLoader(ProbeDataset(ds, transform), batch_size=args.batch_size, num_workers=args.workers)
        data = extract(model, loader)
        if args.features is not None:
            np.savez(args.features, **data)

    labels, head_top1 = data["labels"], data["head_top1"]
    groups = duplicate_groups(data["thumbs"], labels, args.duplicate_threshold)
    val = split_by_group(labels, groups, args.val_fraction, args.seed)
    copies = len(labels) - len(np.unique(groups))
    largest_group = int(np.bincount(groups).max())
    print(f"{copies} rows are near-duplicates of another row (largest group: {largest_group})")
    val_classes = len(np.unique(labels[val]))
    print(f"{val.sum()} val rows in {val_classes} classes, {(~val).sum()} train rows")

    features = torch.from_numpy(data["features"])
    mean, std = features[~val].mean(dim=0), features[~val].std(dim=0) + 1e-6
    features = (features - mean) / std
    y = torch.from_numpy(labels).long()
    x_train, y_train, x_val, y_val = features[~val], y[~val], features[val], y[val]

    # Pick the weight decay on a group-aware split of the train rows, then refit on all train rows.
    inner_val = split_by_group(labels[~val], groups[~val], args.val_fraction, args.seed + 1)
    search = {}
    for wd in args.weight_decays:
        probe = train_probe(x_train[~inner_val], y_train[~inner_val], wd, 1000, args.max_iter)
        search[wd] = accuracy(probe, x_train[inner_val], y_train[inner_val])
        print(f"weight decay {wd:g}: inner val top-1 {search[wd]:.2f}", flush=True)
    best_wd = max(search, key=search.get)
    probe = train_probe(x_train, y_train, best_wd, 1000, args.max_iter)

    results = {
        "imagenet_head": {
            "train_top1": 100 * float((head_top1[~val] == labels[~val]).mean()),
            "val_top1": 100 * float((head_top1[val] == labels[val]).mean()),
        },
        "linear_probe": {
            "train_top1": accuracy(probe, x_train, y_train),
            "val_top1": accuracy(probe, x_val, y_val),
            "val_top5": accuracy(probe, x_val, y_val, k=5),
            "weight_decay": best_wd,
        },
    }
    print("\n| Classifier | Train top-1 | Val top-1 |\n|---|---:|---:|")
    for name, r in results.items():
        print(f"| {name} | {r['train_top1']:.2f} | {r['val_top1']:.2f} |")

    report = {
        "model": args.model,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": {"train": int((~val).sum()), "val": int(val.sum()), "copies": int(copies)},
        "largest_duplicate_group": largest_group,
        "val_classes": val_classes,
        "weight_decay_search": {str(k): v for k, v in search.items()},
        "results": results,
        "versions": {module.__name__: module.__version__ for module in (torch, timm, np)},
        "wall_seconds": time.perf_counter() - start,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
