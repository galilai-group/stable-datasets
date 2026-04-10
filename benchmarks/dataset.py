"""Dataset loading for SSL benchmarks.

Loads image classification datasets from stable_datasets and wraps them
as a ``spt.data.DataModule``.  Transform and collation logic lives in
the individual model files (``models/*.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import stable_pretraining as spt
import torch

import stable_datasets as sds


log = logging.getLogger(__name__)


# Dataset metadata


@dataclass
class DatasetConfig:
    """Static metadata for a benchmark dataset."""

    name: str
    display_name: str
    num_classes: int
    channels: int  # 1 for grayscale, 3 for RGB
    mean: list[float]
    std: list[float]
    image_size: tuple[int, int] = (224, 224)  # always 224x224 for ViT benchmarks


_IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _rgb(name, display_name, num_classes, **stats):
    return DatasetConfig(name, display_name, num_classes, channels=3, **{**_IMAGENET_STATS, **stats})


def _gray(name, display_name, num_classes, mean, std):
    return DatasetConfig(name, display_name, num_classes, channels=1, mean=mean, std=std)


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # RGB datasets (3-channel, ImageNet stats unless overridden)
    "arabiccharacters": _rgb("arabiccharacters", "Arabic Characters", 28),
    "arabicdigits": _rgb("arabicdigits", "Arabic Digits", 10),
    "beans": _rgb("beans", "Beans", 3),
    "cifar10": _rgb("cifar10", "CIFAR-10", 10, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]),
    "cifar100": _rgb("cifar100", "CIFAR-100", 100, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    "country211": _rgb("country211", "Country-211", 211),
    "cub200": _rgb("cub200", "CUB-200", 200),
    "dtd": _rgb("dtd", "DTD", 47),
    "fgvcaircraft": _rgb("fgvcaircraft", "FGVC Aircraft", 100),
    "flowers102": _rgb("flowers102", "Flowers-102", 102, mean=[0.4344, 0.3830, 0.2954], std=[0.2937, 0.2458, 0.2726]),
    "food101": _rgb("food101", "Food-101", 101),
    "imagenet": _rgb("imagenet", "ImageNet", 1000),
    "imagenette": _rgb("imagenette", "Imagenette", 10),
    "rockpaperscissor": _rgb("rockpaperscissor", "Rock-Paper-Scissors", 3),
    "stl10": _rgb("stl10", "STL-10", 10, mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    "svhn": _rgb("svhn", "SVHN", 10, mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    "tinyimagenet": _rgb("tinyimagenet", "Tiny ImageNet", 200),
    # Grayscale datasets (1-channel)
    "emnist": _gray("emnist", "EMNIST", 47, mean=[0.1751], std=[0.3332]),
    "fashionmnist": _gray("fashionmnist", "FashionMNIST", 10, mean=[0.2860], std=[0.3530]),
    "hasyv2": _gray("hasyv2", "HASYv2", 369, mean=[0.4526], std=[0.2194]),
    "kmnist": _gray("kmnist", "KMNIST", 10, mean=[0.1918], std=[0.3483]),
    "medmnist": _gray("medmnist", "MedMNIST", 2, mean=[0.4823], std=[0.2363]),
    "notmnist": _gray("notmnist", "NotMNIST", 10, mean=[0.4178], std=[0.4532]),
}

# Datasets that need extra kwargs when loading
DATASET_KWARGS = {
    "emnist": {"config_name": "balanced"},
    "medmnist": {"config_name": "pneumoniamnist"},
}


# Dataset loading


def _get_dataset_classes() -> dict[str, type]:
    return {
        name.lower(): cls
        for name, cls in vars(sds.images).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }


def _load_validation_split(dataset_cls, **kwargs):
    try:
        return dataset_cls(split="test", **kwargs)
    except (ValueError, KeyError):
        pass
    try:
        return dataset_cls(split="validation", **kwargs)
    except (ValueError, KeyError):
        return None


def get_config(name: str) -> DatasetConfig:
    """Look up static config for a dataset by name.

    Falls back to runtime extraction if the dataset isn't in the static table.
    """
    name_lower = name.lower()
    if name_lower in DATASET_CONFIGS:
        return DATASET_CONFIGS[name_lower]

    raise ValueError(f"Unknown dataset: '{name}'. Available: {', '.join(sorted(DATASET_CONFIGS.keys()))}")


def create_dataset(
    name: str,
    train_transform,
    val_transform,
    collate_fn,
    training_cfg,
    data_dir: str | None = None,
) -> tuple[spt.data.DataModule, DatasetConfig]:
    """Load a dataset and wrap it as a DataModule.

    Transforms and collation are provided by the caller (typically from
    the model's ``create_transforms`` function).

    Args:
        name: Dataset name (case-insensitive).
        train_transform: Transform applied to each training sample.
        val_transform: Transform applied to each validation sample.
        collate_fn: Collation function for the DataLoader.
        training_cfg: OmegaConf node with batch_size and num_workers.
        data_dir: Root directory for HF downloads/cache.

    Returns:
        Tuple of (DataModule, DatasetConfig).
    """
    name_lower = name.lower()
    ds_config = get_config(name_lower)

    dataset_classes = _get_dataset_classes()
    if name_lower not in dataset_classes:
        available = ", ".join(sorted(dataset_classes.keys()))
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    dataset_cls = dataset_classes[name_lower]
    extra_kwargs = DATASET_KWARGS.get(name_lower, {})
    if data_dir is not None:
        root = Path(data_dir)
        extra_kwargs["download_dir"] = str(root / "downloads")
        extra_kwargs["processed_cache_dir"] = str(root / "processed")

    log.info(f"Loading train split for '{name_lower}'...")
    train_hf = dataset_cls(split="train", **extra_kwargs)
    log.info(f"Loading validation split for '{name_lower}'...")
    val_hf = _load_validation_split(dataset_cls, **extra_kwargs)

    if val_hf is None:
        log.warning(f"No test/validation split for '{name_lower}'; holding out 10%% of train.")
        splits = train_hf.train_test_split(test_size=0.1, seed=42)
        train_hf = splits["train"]
        val_hf = splits["test"]

    batch_size = training_cfg.batch_size
    num_workers = training_cfg.num_workers
    prefetch_factor = getattr(training_cfg, "prefetch_factor", 2) if num_workers > 0 else None
    mp_ctx = "fork" if num_workers > 0 else None

    # Apply transforms directly via StableDataset.with_transform
    train_ds = train_hf.with_transform(train_transform)
    val_ds = val_hf.with_transform(val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        multiprocessing_context=mp_ctx,
        prefetch_factor=prefetch_factor,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        multiprocessing_context=mp_ctx,
        prefetch_factor=prefetch_factor,
    )

    return spt.data.DataModule(train=train_loader, val=val_loader), ds_config
