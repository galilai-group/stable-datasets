"""Reproduce timm's published ImageNet-V2 and ImageNet-Sketch accuracies with the stable-datasets builders.

Each model is evaluated with timm's eval transform (the same preprocessing as timm's validate.py),
so a top-1 within --tolerance of timm's number means the images and labels match the originals.
timm has no table for the threshold-0.7 and top-images variants; only our numbers are reported.

Usage:
    python examples/validate_imagenet_v2_sketch.py --models resnet50.tv_in1k resnet18.tv_in1k

References:
    https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv
    https://github.com/huggingface/pytorch-image-models/blob/main/results/results-sketch.csv
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import stable_datasets
from stable_datasets.images import ImageNetSketch, ImageNetV2


# name -> (builder, config_name, expected rows)
DATASETS = {
    "imagenet_v2/matched-frequency": (ImageNetV2, "matched-frequency", 10_000),
    "imagenet_v2/threshold-0.7": (ImageNetV2, "threshold-0.7", 10_000),
    "imagenet_v2/top-images": (ImageNetV2, "top-images", 10_000),
    "imagenet_sketch": (ImageNetSketch, None, 50_889),
}

# timm (top-1, top-5) in percent, from the CSVs linked above.
REFERENCE = {
    "imagenet_v2/matched-frequency": {
        "resnet50.tv_in1k": (63.190, 84.640),
        "resnet18.tv_in1k": (57.350, 80.080),
        "resnet34.tv_in1k": (61.210, 82.670),
        "deit_small_patch16_224.fb_in1k": (68.530, 88.170),
        "efficientnet_b0.ra_in1k": (66.150, 86.120),
        "resnet50.tv2_in1k": (69.510, 88.630),
    },
    "imagenet_sketch": {
        "resnet50.tv_in1k": (24.064, 41.329),
        "resnet18.tv_in1k": (20.238, 37.258),
        "resnet34.tv_in1k": (23.494, 41.380),
        "deit_small_patch16_224.fb_in1k": (29.435, 48.268),
        "efficientnet_b0.ra_in1k": (25.017, 42.791),
        "resnet50.tv2_in1k": (29.869, 48.046),
    },
}


class TimmEvalDataset(Dataset):
    """Wrap builder rows as ``(transform(rgb_image), label)``; module-level so DataLoader workers can pickle it."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(row["image"].convert("RGB")), row["label"]


@torch.inference_mode()
def evaluate(model, loader, device, num_classes=1000, desc=None):
    """Return integer top-1/top-5 hit counts and the per-class label histogram."""
    top1 = top5 = count = 0
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for images, labels in tqdm(loader, desc=desc, leave=False):
        logits = model(images.to(device, non_blocking=True))
        hits = logits.topk(5, dim=1).indices.cpu() == labels[:, None]
        top1 += int(hits[:, 0].sum())
        top5 += int(hits.any(dim=1).sum())
        count += len(labels)
        class_counts += np.bincount(labels.numpy(), minlength=num_classes)
    return {"count": count, "top1_correct": top1, "top5_correct": top5, "class_counts": class_counts}


def markdown_table(results):
    lines = ["| Dataset | Model | Rows | Top-1 | Top-5 | timm Top-1 | Δ (pp) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        ref = "-" if r["timm_top1"] is None else f"{r['timm_top1']:.3f}"
        delta = "-" if r["delta_top1"] is None else f"{r['delta_top1']:+.3f}"
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['count']:,} | {r['top1']:.3f} | {r['top5']:.3f} | {ref} | {delta} |"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce timm's published ImageNet-V2 / ImageNet-Sketch accuracies on stable-datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=["resnet50.tv_in1k", "resnet18.tv_in1k"], help="timm model names"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
        metavar="NAME",
        help="any of: " + ", ".join(DATASETS),
    )
    parser.add_argument("--batch-size", type=int, default=64, help="images per batch")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader workers")
    parser.add_argument("--limit", type=int, help="evaluate N evenly spaced rows and skip the pass/fail checks")
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, help="torch device")
    parser.add_argument("--tolerance", type=float, default=0.1, help="max |top-1 - timm top-1| in percentage points")
    parser.add_argument("--out", type=Path, default=Path("imagenet_v2_sketch_validation.json"), help="JSON report")
    parser.add_argument("--download-dir", help="raw download dir; None means the library default")
    parser.add_argument("--cache-dir", help="processed cache dir; None means the library default")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import timm
        import timm.data
    except ImportError:
        sys.exit("This script needs timm: pip install timm")

    start = time.perf_counter()
    device = torch.device(args.device)
    print(f"device: {device}", flush=True)
    models, data_configs = {}, {}
    for name in args.models:
        model = timm.create_model(name, pretrained=True).eval().to(device)
        data_configs[name] = timm.data.resolve_data_config({}, model=model)
        models[name] = (model, timm.data.create_transform(**data_configs[name]))

    results, dataset_stats, failures = [], {}, []
    for key in args.datasets:
        builder_cls, config_name, expected_rows = DATASETS[key]
        print(f"\n{key}", flush=True)
        ds = builder_cls(
            split="test", config_name=config_name, download_dir=args.download_dir, processed_cache_dir=args.cache_dir
        )
        rows = ds
        if args.limit is not None and args.limit < len(ds):
            # Evenly spaced rows so a small smoke test still spans many classes.
            rows = ds.select(np.linspace(0, len(ds) - 1, args.limit).astype(np.int64))

        class_names = ds.features["label"].names
        stats = dataset_stats[key] = {
            "rows": len(ds),
            "num_classes": len(class_names),
            "label_order_matches_timm": class_names == timm.data.ImageNetInfo().label_names(),
        }
        if args.limit is None and len(ds) != expected_rows:
            failures.append(f"{key}: {len(ds)} rows, expected {expected_rows}")
        if not stats["label_order_matches_timm"]:
            failures.append(f"{key}: class order differs from timm's ImageNet-1k synsets")

        for name, (model, transform) in models.items():
            loader = DataLoader(
                TimmEvalDataset(rows, transform),
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            counts = evaluate(model, loader, device, num_classes=len(class_names), desc=f"{key} {name}")
            present = counts.pop("class_counts")
            present = present[present > 0]
            stats.update(
                evaluated_rows=counts["count"],
                distinct_labels=len(present),
                min_per_class=int(present.min()),
                max_per_class=int(present.max()),
            )
            top1 = 100 * counts["top1_correct"] / counts["count"]
            top5 = 100 * counts["top5_correct"] / counts["count"]
            timm_top1, timm_top5 = REFERENCE.get(key, {}).get(name, (None, None))
            delta = None if timm_top1 is None else top1 - timm_top1
            results.append(
                {
                    "dataset": key,
                    "model": name,
                    **counts,
                    "top1": top1,
                    "top5": top5,
                    "timm_top1": timm_top1,
                    "timm_top5": timm_top5,
                    "delta_top1": delta,
                }
            )
            print(f"{name}: top-1 {top1:.3f}, top-5 {top5:.3f}", flush=True)
            if args.limit is None and delta is not None and abs(delta) > args.tolerance:
                failures.append(f"{key} / {name}: top-1 off by {delta:+.3f} pp (tolerance {args.tolerance})")

    print("\n" + markdown_table(results))
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            **{module.__name__: module.__version__ for module in (stable_datasets, torch, timm, PIL, np)},
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_configs": data_configs,
        "results": results,
        "datasets": dataset_stats,
        "failures": failures,
        "wall_seconds": time.perf_counter() - start,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.out}")
    if failures:
        print("FAILED:\n" + "\n".join(failures))
        return 1
    if args.limit is not None:
        print("--limit set: accuracy and row-count checks skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
