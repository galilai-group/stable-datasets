#!/usr/bin/env python
"""Download all classification datasets for SSL benchmarks.

Usage:
    python download_datasets.py
    python download_datasets.py --datasets cifar10,stl10,svhn
"""

from __future__ import annotations

import argparse

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


def download_dataset(name: str) -> None:
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
    args = parser.parse_args()

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = CLASSIFICATION_DATASETS

    print(f"Downloading {len(datasets)} datasets...")
    print(f"Datasets: {', '.join(datasets)}")

    for dataset_name in datasets:
        download_dataset(dataset_name)

    print(f"\n{'='*60}")
    print(f"✓ Download complete! Downloaded {len(datasets)} datasets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
