#!/usr/bin/env python
"""Pre-cache SSL benchmark datasets as 224x224 RGB numpy arrays.

Converts each dataset split from HuggingFace Arrow format into flat numpy
memmap files (uint8 images + int64 labels) plus a small JSON metadata sidecar.
This eliminates the per-sample PIL decode + resize overhead that dominates
dataloader time when GPUs are fast (e.g. H200s with ViT-Small).

Run once before launching sweeps:

    python cache_datasets.py
    python cache_datasets.py --datasets cifar10,stl10,flowers102
    python cache_datasets.py --data-dir /mnt/data/sami/.stable-datasets \
                             --cache-dir /mnt/data/sami/stable-datasets/.cached_224

The cache layout is:

    {cache_dir}/{dataset_name}/
        train_images.npy   # (N, 224, 224, 3) uint8 memmap
        train_labels.npy   # (N,) int64
        val_images.npy     # (M, 224, 224, 3) uint8 memmap
        val_labels.npy     # (M,) int64
        metadata.json      # num_classes, original_channels, has_native_val, etc.

For datasets without a native val/test split, a 90/10 train split (seed=42)
is used — matching the existing benchmark code exactly.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Reproduces the constants from dataset.py / main.py
DATASET_KWARGS = {
    "emnist": {"config_name": "balanced"},
    "medmnist": {"config_name": "pneumoniamnist"},
}

SKIP_DATASETS = {
    "cars196", "cifar10c", "cifar100c", "clevrer",
    "dsprites", "dspritescolor", "dspritesnoise", "dspritesscream",
    "cars3d", "shapes3d", "smallnorb",
    "facepointing", "celeba",
    "imagenet",  # too large, skip by default
}

TARGET_SIZE = (224, 224)


def _get_dataset_classes():
    import stable_datasets as sds

    return {
        name.lower(): cls
        for name, cls in vars(sds.images).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }


def _load_val_split(dataset_cls, **kwargs):
    for split_name in ("test", "validation"):
        try:
            return dataset_cls(split=split_name, **kwargs), split_name
        except (ValueError, KeyError):
            continue
    return None, None


def _extract_channels(sample) -> int:
    """Get original channel count from a single sample."""
    img = sample["image"]
    if hasattr(img, "mode"):
        return 1 if img.mode in ("L", "1", "P") else 3
    arr = np.array(img)
    return 1 if arr.ndim == 2 else arr.shape[2]


def _process_split(hf_dataset, out_dir: Path, split_name: str) -> int:
    """Convert an HF dataset split to numpy memmap files.

    Returns the number of samples processed.
    """
    n = len(hf_dataset)
    images_path = out_dir / f"{split_name}_images.npy"
    labels_path = out_dir / f"{split_name}_labels.npy"

    # Pre-allocate memmap files
    images = np.lib.format.open_memmap(
        str(images_path), mode="w+", dtype=np.uint8,
        shape=(n, TARGET_SIZE[0], TARGET_SIZE[1], 3),
    )
    labels = np.zeros(n, dtype=np.int64)

    for i in tqdm(range(n), desc=f"  {split_name}", unit="img"):
        sample = hf_dataset[i]
        img = sample["image"]

        # Convert to PIL if needed, then to RGB 224x224
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)

        images[i] = np.array(img)
        labels[i] = sample.get("label", -1)

    # Flush memmap to disk
    del images

    np.save(str(labels_path), labels)
    return n


def cache_single_dataset(
    name: str,
    data_dir: str | None,
    cache_dir: str,
) -> None:
    """Cache a single dataset to disk."""
    dataset_classes = _get_dataset_classes()
    if name not in dataset_classes:
        log.error(f"Unknown dataset: {name}")
        return

    out_dir = Path(cache_dir) / name
    metadata_path = out_dir / "metadata.json"

    if metadata_path.exists():
        log.info(f"[{name}] Already cached, skipping. Delete {out_dir} to re-cache.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_cls = dataset_classes[name]
    extra_kwargs = DATASET_KWARGS.get(name, {})
    if data_dir is not None:
        root = Path(data_dir)
        extra_kwargs["download_dir"] = str(root / "downloads")
        extra_kwargs["processed_cache_dir"] = str(root / "processed")

    t0 = time.time()

    # Load train
    log.info(f"[{name}] Loading train split...")
    train_hf = dataset_cls(split="train", **extra_kwargs)

    # Detect original channels from first sample
    original_channels = _extract_channels(train_hf[0])

    # Load val (or create synthetic val split)
    log.info(f"[{name}] Loading validation split...")
    val_hf, val_split_name = _load_val_split(dataset_cls, **extra_kwargs)

    has_native_val = val_hf is not None
    if not has_native_val:
        log.warning(f"[{name}] No native val split; holding out 10% of train (seed=42).")
        splits = train_hf.train_test_split(test_size=0.1, seed=42)
        train_hf = splits["train"]
        val_hf = splits["test"]

    # Extract num_classes
    num_classes = 0
    label_feature = train_hf.features.get("label")
    if label_feature is not None and hasattr(label_feature, "num_classes"):
        num_classes = label_feature.num_classes

    # Process splits
    log.info(f"[{name}] Caching train ({len(train_hf)} images)...")
    n_train = _process_split(train_hf, out_dir, "train")

    log.info(f"[{name}] Caching val ({len(val_hf)} images)...")
    n_val = _process_split(val_hf, out_dir, "val")

    # Save metadata
    metadata = {
        "name": name,
        "num_classes": num_classes,
        "original_channels": original_channels,
        "image_size": list(TARGET_SIZE),
        "has_native_val": has_native_val,
        "n_train": n_train,
        "n_val": n_val,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    total = n_train + n_val
    size_gb = (total * TARGET_SIZE[0] * TARGET_SIZE[1] * 3) / (1024**3)
    log.info(
        f"[{name}] Done: {total} images, {size_gb:.1f} GB, {elapsed:.0f}s "
        f"({total / elapsed:.0f} img/s)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache SSL benchmark datasets as 224x224 numpy memmaps"
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (default: all classification datasets)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/mnt/data/sami/.stable-datasets",
        help="Root directory for HF downloads/cache",
    )
    parser.add_argument(
        "--cache-dir", type=str,
        default="/mnt/data/sami/stable-datasets/.cached_224",
        help="Output directory for cached numpy files",
    )
    parser.add_argument(
        "--include-imagenet", action="store_true",
        help="Include ImageNet (excluded by default due to size)",
    )
    args = parser.parse_args()

    dataset_classes = _get_dataset_classes()
    skip = set(SKIP_DATASETS)
    if args.include_imagenet:
        skip.discard("imagenet")

    if args.datasets:
        names = [d.strip().lower() for d in args.datasets.split(",")]
    else:
        names = sorted(n for n in dataset_classes if n not in skip)

    log.info(f"Caching {len(names)} datasets to {args.cache_dir}")
    log.info(f"Datasets: {', '.join(names)}")

    for name in names:
        try:
            cache_single_dataset(name, args.data_dir, args.cache_dir)
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}", exc_info=True)

    log.info("All done!")


if __name__ == "__main__":
    main()