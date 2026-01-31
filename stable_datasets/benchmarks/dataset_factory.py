"""Dataset factory for SSL baselines.

This module provides utilities to create datasets from stable-datasets
for SSL training. stable-datasets returns HuggingFace datasets directly,
so we use HuggingFace's native transform functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
import torchcodec

import numpy as np
import stable_datasets as sds
import sys ; sys.path.append("/root/stable-datasets/stable-pretraining") # so u can use mae code wthout new release (gitmodule)

import stable_pretraining as spt
import torch
from stable_pretraining.data import transforms


@dataclass
class DatasetConfig:
    """Configuration for a dataset, extracted from the HF dataset."""

    name: str
    num_classes: int
    image_size: tuple[int, int]
    channels: int
    mean: list[float]
    std: list[float]
    low_resolution: bool
    num_frames: int 


NORMALIZATION_STATS: dict[str, dict] = {
    "cifar10": spt.data.static.CIFAR10,
    "cifar100": spt.data.static.CIFAR100,
    "cifar10c": spt.data.static.CIFAR10,
    "cifar100c": spt.data.static.CIFAR100,
    "fashionmnist": spt.data.static.FashionMNIST,
    "svhn": spt.data.static.SVHN,
    "stl10": spt.data.static.STL10,
    "flowers102": spt.data.static.OxfordFlowers,
}

DEFAULT_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_dataset_classes() -> dict[str, type]:
    """Return all available dataset classes from stable_datasets.images."""
    classes = {
        name.lower(): cls
        for name, cls in vars(sds.images).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }
    return classes


def _extract_image_dimensions(image) -> tuple[int, int, int]:
    """Extract height, width, and channels from an image (PIL or array)."""
    if hasattr(image, "size"):
        w, h = image.size
        image_np = np.array(image)
        channels = 1 if image_np.ndim == 2 else image_np.shape[2]
    else:
        image_np = np.array(image)
        if image_np.ndim == 2:
            h, w = image_np.shape
            channels = 1
        else:
            h, w, channels = image_np.shape
    return h, w, channels


def _get_normalization_stats(name: str, channels: int) -> tuple[list[float], list[float]]:
    """Get normalization stats, adjusting for grayscale if needed."""
    stats = NORMALIZATION_STATS.get(name, DEFAULT_STATS)
    mean = stats["mean"]
    std = stats["std"]

    if channels == 1 and len(mean) == 3:
        mean = [sum(mean) / 3]
        std = [sum(std) / 3]

    return mean, std


def extract_config_from_dataset(name: str, dataset) -> DatasetConfig:
    """Extract configuration from a loaded HuggingFace dataset."""
    name_lower = name.lower()

    num_classes = 0
    label_feature = dataset.features.get("label")
    if label_feature is not None and hasattr(label_feature, "num_classes"):
        num_classes = label_feature.num_classes

    sample = dataset[0]
    if 'image' in sample:
        image = sample['image']
        h, w, channels = _extract_image_dimensions(image)
        num_frames = 1

    if 'video' in sample:
        sample = sample['video']
        if isinstance(sample, torchcodec.decoders.VideoDecoder):
            num_frames = sample._num_frames
            frame = sample.get_frame_at(0)
            shape = frame.data.shape
            channels, h, w = shape
        else: 
            num_frames, h, w, channels = sample.data.shape

    mean, std = _get_normalization_stats(name_lower, channels)

    return DatasetConfig(
        name=name_lower,
        num_classes=num_classes,
        image_size=(h, w),
        channels=channels,
        mean=mean,
        std=std,
        low_resolution=max(h, w) <= 64,
        num_frames=num_frames,
    )


def _build_view_transform(config: DatasetConfig):
    """Build view transform for SSL with resolution-appropriate augmentations."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}

    if config.low_resolution:
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((h, w), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**stats),
        )

    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((h, w), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.ToImage(**stats),
    )


def _build_val_transform(config: DatasetConfig):
    """Build validation transform (simple resize and normalize)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((h, w)),
        transforms.ToImage(**stats),
    )


def create_ssl_transforms(config: DatasetConfig, num_views: int = 2):
    """Create SSL training and validation transforms."""
    view_transform = _build_view_transform(config)
    train_transform = transforms.MultiViewTransform([view_transform] * num_views)
    val_transform = _build_val_transform(config)
    return train_transform, val_transform


def create_mae_transforms(config: DatasetConfig):
    """Create MAE-specific transforms (single view with minimal augmentation)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((h, w), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**stats),
    )
    val_transform = _build_val_transform(config)
    return train_transform, val_transform


def _load_validation_split(dataset_cls, **kwargs):
    """Load validation split, trying 'test' first, then 'validation'."""
    for split_name in ("test", "validation"):
        try:
            return dataset_cls(split=split_name, **kwargs)
        except (ValueError, KeyError):
            continue
    return None


def _make_transform_fn(transform):
    """Create a transform function for HuggingFace set_transform."""
    def apply_transform(batch):
        return transform({"image": batch["image"], "label": batch["label"]})
    return apply_transform


def _create_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = False, drop_last: bool = False):
    """Create a DataLoader with standard settings."""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def create_dataset(
    name: str,
    batch_size: int = 256,
    num_workers: int = 8,
    transform_type: str = "ssl",
    **dataset_kwargs,
) -> tuple[spt.data.DataModule, DatasetConfig]:
    """Create a dataset and data module for SSL training.

    Args:
        name: Dataset name (case-insensitive)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        transform_type: Type of transforms ("ssl" for SimCLR-style, "mae" for MAE-style)
        **dataset_kwargs: Additional kwargs passed to the dataset class

    Returns:
        Tuple of (DataModule, DatasetConfig)
    """
    name_lower = name.lower()
    dataset_classes = get_dataset_classes()

    if name_lower not in dataset_classes:
        available = ", ".join(sorted(dataset_classes.keys()))
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    dataset_cls = dataset_classes[name_lower]

    train_hf = dataset_cls(split="train", **dataset_kwargs)
    val_hf = _load_validation_split(dataset_cls, **dataset_kwargs)

    config = extract_config_from_dataset(name_lower, train_hf)

    if transform_type == "mae":
        train_transform, val_transform = create_mae_transforms(config)
    else:
        train_transform, val_transform = create_ssl_transforms(config)

    train_hf.set_transform(_make_transform_fn(train_transform))
    train_dataloader = _create_dataloader(train_hf, batch_size, num_workers, shuffle=True, drop_last=True)

    val_dataloader = None
    if val_hf is not None:
        val_hf.set_transform(_make_transform_fn(val_transform))
        val_dataloader = _create_dataloader(val_hf, batch_size, num_workers)

    data_module = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
    return data_module, config
