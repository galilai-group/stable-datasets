#!/usr/bin/env python
"""Download all classification datasets for SSL benchmarks.

Usage:
    python download_datasets.py
    python download_datasets.py --datasets cifar10,stl10,svhn
    python download_datasets.py --data-dir /mnt/data/sami/.stable-datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import stable_datasets as sds

# Classification datasets (with labels)
CLASSIFICATION_DATASETS = [
    "arabiccharacters",
    "arabicdigits",
    "country211",
    "cub200",
    "dtd",
    "emnist",
    "facepointing",
    "fashionmnist",
    "flowers102",
    "food101",
    "hasyv2",
    "kmnist",
    "linnaeus5",
    "medmnist",
    "notmnist",
    "rockpaperscissor",
    "stl10",
    "svhn",
]

# Dataset-specific kwargs
DATASET_KWARGS = {
    "emnist": {"config_name": "balanced"},
    "medmnist": {"config_name": "pneumoniamnist"},
}


def download_dataset(name: str, data_dir: str | None = None) -> None:
    """Download a single dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"{'='*60}")

    dataset_classes = {
        name.lower(): cls
        for name, cls in vars(sds.images).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }

    if name not in dataset_classes:
        print(f"❌ Unknown dataset: {name}")
        return

    dataset_cls = dataset_classes[name]
    extra_kwargs = DATASET_KWARGS.get(name, {})

    # Use custom data directory if provided
    if data_dir is not None:
        root = Path(data_dir)
        extra_kwargs["download_dir"] = str(root / "downloads")
        extra_kwargs["processed_cache_dir"] = str(root / "processed")

    try:
        # Download train split
        print(f"  → Loading train split...")
        train_ds = dataset_cls(split="train", **extra_kwargs)
        print(f"  ✓ Train: {len(train_ds)} examples")

        # Download test/validation split
        for split_name in ("test", "validation"):
            try:
                print(f"  → Loading {split_name} split...")
                val_ds = dataset_cls(split=split_name, **extra_kwargs)
                print(f"  ✓ {split_name.capitalize()}: {len(val_ds)} examples")
                break
            except (ValueError, KeyError):
                continue

        print(f"✓ Successfully downloaded: {name}")

    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download SSL benchmark datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to download (default: all classification datasets)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/data/sami/.stable-datasets",
        help="Root directory for downloads and cache (default: /mnt/data/sami/.stable-datasets)",
    )
    args = parser.parse_args()

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = CLASSIFICATION_DATASETS

    print(f"Downloading {len(datasets)} datasets...")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Data directory: {args.data_dir}")

    for dataset_name in datasets:
        download_dataset(dataset_name, data_dir=args.data_dir)

    print(f"\n{'='*60}")
    print(f"✓ Download complete! Downloaded {len(datasets)} datasets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
