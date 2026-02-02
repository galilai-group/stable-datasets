"""Dataset factory for SSL baselines.

This module provides utilities to create datasets from stable-datasets
for SSL training. Uses a PyTorch Dataset wrapper to avoid HuggingFace's
set_transform quirks with nested list structures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stable_datasets as sds
import sys
sys.path.append("/root/stable-datasets/stable-pretraining")

import stable_pretraining as spt
import torch
import torchcodec
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
    data_key: str = "image"  # "image" or "video"


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


# =============================================================================
# PyTorch Dataset Wrapper (bypasses HuggingFace set_transform quirks)
# =============================================================================

class TransformDataset(torch.utils.data.Dataset):
    """Simple PyTorch Dataset wrapper that applies transforms per-sample.
    
    This bypasses HuggingFace's set_transform which mangles nested list structures.
    Normalizes output to always use "image" key for compatibility with forward functions.
    """
    
    def __init__(self, hf_dataset, transform, data_key: str = "image"):
        self.dataset = hf_dataset
        self.transform = transform
        self.data_key = data_key
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Build sample dict with data_key and label
        transform_input = {self.data_key: sample[self.data_key]}
        if "label" in sample:
            transform_input["label"] = sample["label"]
        # Apply transform (may be MultiViewTransform or single-view)
        output = self.transform(transform_input)
        # Normalize key to "image" for compatibility with forward functions
        return self._normalize_keys(output)
    
    def _normalize_keys(self, output):
        """Normalize data_key to 'image' for forward function compatibility."""
        if self.data_key == "image":
            return output
        
        if "views" in output:
            # Multi-view: normalize each view
            for view in output["views"]:
                if self.data_key in view:
                    view["image"] = view.pop(self.data_key)
        elif self.data_key in output:
            # Single-view: rename key
            output["image"] = output.pop(self.data_key)
        
        return output


def _collate_views(batch):
    """Collate function for multi-view and single-view batches.
    
    Handles:
    - MultiViewTransform output: {"views": [view0_dict, view1_dict]}
    - Single-view output: {"image": tensor, "label": int}
    """
    first = batch[0]
    
    if "views" in first:
        # Multi-view: each sample has {"views": [{"image": tensor, "label": int}, ...]}
        num_views = len(first["views"])
        views_list = []
        for v in range(num_views):
            # Stack images from view v across all samples
            view_images = torch.stack([sample["views"][v]["image"] for sample in batch])
            view_dict = {"image": view_images}
            # Stack labels if present
            if "label" in first["views"][v]:
                view_labels = torch.tensor([sample["views"][v]["label"] for sample in batch])
                view_dict["label"] = view_labels
            views_list.append(view_dict)
        return {"views": views_list}
    
    else:
        # Single-view: each sample has {"image": tensor, "label": int}
        result = {"image": torch.stack([sample["image"] for sample in batch])}
        if "label" in first:
            result["label"] = torch.tensor([sample["label"] for sample in batch])
        return result


# =============================================================================
# Dataset Config Extraction
# =============================================================================

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
    data_key = "image"
    num_frames = 1
    
    if 'image' in sample:
        image = sample['image']
        h, w, channels = _extract_image_dimensions(image)
        data_key = "image"

    if 'video' in sample:
        data_key = "video"
        video = sample['video']
        if isinstance(video, torchcodec.decoders.VideoDecoder):
            num_frames = video._num_frames
            frame = video.get_frame_at(0)
            shape = frame.data.shape
            channels, h, w = shape
        else: 
            num_frames, h, w, channels = video.data.shape

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
        data_key=data_key,
    )


# =============================================================================
# Transform Builders
# =============================================================================

def _build_view_transform(config: DatasetConfig):
    """Build view transform for SSL with resolution-appropriate augmentations."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key

    if config.low_resolution:
        return transforms.Compose(
            transforms.RGB(source=source, target=target),
            transforms.RandomResizedCrop((h, w), scale=(0.2, 1.0), source=source, target=target),
            transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8, source=source, target=target),
            transforms.RandomGrayscale(p=0.2, source=source, target=target),
            transforms.ToImage(**stats, source=source, target=target),
        )

    return transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.RandomResizedCrop((h, w), scale=(0.08, 1.0), source=source, target=target),
        transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8, source=source, target=target),
        transforms.RandomGrayscale(p=0.2, source=source, target=target),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5, source=source, target=target),
        transforms.ToImage(**stats, source=source, target=target),
    )


def _build_val_transform(config: DatasetConfig):
    """Build validation transform (simple resize and normalize)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key
    return transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.Resize((h, w), source=source, target=target),
        transforms.ToImage(**stats, source=source, target=target),
    )


def create_simclr_transforms(config: DatasetConfig, num_views: int = 2):
    """Create SimCLR transforms (multi-view contrastive learning)."""
    view_transform = _build_view_transform(config)
    train_transform = transforms.MultiViewTransform([view_transform] * num_views)
    val_transform = _build_val_transform(config)
    return train_transform, val_transform


def create_mae_transforms(config: DatasetConfig):
    """Create MAE transforms (single view with minimal augmentation for reconstruction)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key

    train_transform = transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.RandomResizedCrop((h, w), scale=(0.2, 1.0), source=source, target=target),
        transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
        transforms.ToImage(**stats, source=source, target=target),
    )
    val_transform = _build_val_transform(config)
    return train_transform, val_transform


# =============================================================================
# Model-to-Transform Mapping
# =============================================================================

MODEL_TRANSFORMS = {
    "simclr": create_simclr_transforms,  # Multi-view contrastive transforms
    "mae": create_mae_transforms,         # Single-view reconstruction transforms
}


def get_transforms_for_model(model_name: str, config: DatasetConfig, **kwargs):
    """Get transforms appropriate for a given model.
    
    Args:
        model_name: Name of the model ("simclr", "mae", etc.)
        config: Dataset configuration
        **kwargs: Additional kwargs passed to the transform factory
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_TRANSFORMS:
        available = ", ".join(MODEL_TRANSFORMS.keys())
        raise ValueError(f"Unknown model for transforms: {model_name}. Available: {available}")
    
    return MODEL_TRANSFORMS[model_name_lower](config, **kwargs)


# =============================================================================
# Dataset Creation
# =============================================================================

def _load_validation_split(dataset_cls, **kwargs):
    """Load validation split, trying 'test' first, then 'validation'."""
    for split_name in ("test", "validation"):
        try:
            return dataset_cls(split=split_name, **kwargs)
        except (ValueError, KeyError):
            continue
    return None


def create_dataset(
    name: str,
    model_name: str = "simclr",
    batch_size: int = 256,
    num_workers: int = 4,
    **dataset_kwargs,
) -> tuple[spt.data.DataModule, DatasetConfig]:
    """Create a dataset and data module for SSL training.

    Args:
        name: Dataset name (case-insensitive)
        model_name: SSL model name - determines transform type ("simclr", "mae", etc.)
        batch_size: Batch size for training
        num_workers: Number of data loading workers
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

    # Load raw HuggingFace datasets
    train_hf = dataset_cls(split="train", **dataset_kwargs)
    val_hf = _load_validation_split(dataset_cls, **dataset_kwargs)

    # Extract config from dataset
    config = extract_config_from_dataset(name_lower, train_hf)

    # Create model-specific transforms
    train_transform, val_transform = get_transforms_for_model(model_name, config)

    # Wrap in PyTorch Dataset (bypasses HuggingFace set_transform quirks)
    train_dataset = TransformDataset(train_hf, train_transform, data_key=config.data_key)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_views,
    )

    val_dataloader = None
    if val_hf is not None:
        val_dataset = TransformDataset(val_hf, val_transform, data_key=config.data_key)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_views,
        )

    data_module = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
    return data_module, config


# TODO: SimCLR (Anurag, )