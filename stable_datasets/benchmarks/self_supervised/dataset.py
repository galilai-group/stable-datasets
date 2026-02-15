"""Dataset loading and config-driven transforms for SSL benchmarks.

Loads datasets from stable_datasets, extracts runtime config (image size, num_classes,
normalization stats), and creates transforms driven by the model's transform config
(multiview, single, or multicrop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stable_datasets as sds
import stable_pretraining as spt
import torch
from stable_pretraining.data import transforms

# Lazy import torchcodec only when needed for video datasets
try:
    import torchcodec
    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False
    torchcodec = None


# =============================================================================
# Dataset Config
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration extracted at runtime from a loaded HF dataset."""

    name: str
    num_classes: int
    image_size: tuple[int, int]
    channels: int
    mean: list[float]
    std: list[float]
    low_resolution: bool
    num_frames: int
    data_key: str = "image"


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

# Datasets that need extra kwargs when loading
DATASET_KWARGS = {
    "emnist": {"config_name": "balanced"},
    "medmnist": {"config_name": "pneumoniamnist"},
}


# =============================================================================
# PyTorch Dataset Wrapper (bypasses HuggingFace set_transform quirks)
# =============================================================================


class TransformDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper that applies transforms per-sample.

    Bypasses HuggingFace's set_transform which mangles nested list structures
    from MultiViewTransform. Normalizes output to always use "image" key.
    """

    def __init__(self, hf_dataset, transform, data_key: str = "image"):
        self.dataset = hf_dataset
        self.transform = transform
        self.data_key = data_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        transform_input = {self.data_key: sample[self.data_key]}
        if "label" in sample:
            transform_input["label"] = sample["label"]
        output = self.transform(transform_input)
        return self._normalize_keys(output)

    def _normalize_keys(self, output):
        if self.data_key == "image":
            return output
        if "views" in output:
            for view in output["views"]:
                if self.data_key in view:
                    view["image"] = view.pop(self.data_key)
        elif self.data_key in output:
            output["image"] = output.pop(self.data_key)
        return output


def _collate_views(batch):
    """Collate function for multi-view and single-view batches.

    Handles three formats:
    - Single-view: {"image": tensor, "label": int}
    - List-based multi-view: {"views": [view0_dict, view1_dict]}
    - Dict-based multi-view (DINO): {"global_1": {...}, "local_1": {...}}
    """
    first = batch[0]

    if "image" in first:
        result = {"image": torch.stack([s["image"] for s in batch])}
        if "label" in first:
            result["label"] = torch.tensor([s["label"] for s in batch])
        return result

    if "views" in first:
        num_views = len(first["views"])
        views_list = []
        for v in range(num_views):
            view_dict = {"image": torch.stack([s["views"][v]["image"] for s in batch])}
            if "label" in first["views"][v]:
                view_dict["label"] = torch.tensor(
                    [s["views"][v]["label"] for s in batch]
                )
            views_list.append(view_dict)
        return {"views": views_list}

    # Dict-based multi-view (DINO format)
    collated = {}
    for view_name in first.keys():
        if isinstance(first[view_name], dict) and "image" in first[view_name]:
            view_dict = {
                "image": torch.stack([s[view_name]["image"] for s in batch])
            }
            if "label" in first[view_name]:
                view_dict["label"] = torch.tensor(
                    [s[view_name]["label"] for s in batch]
                )
            collated[view_name] = view_dict
    return collated


# =============================================================================
# Config Extraction
# =============================================================================


def _extract_image_dimensions(image) -> tuple[int, int, int]:
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
    stats = NORMALIZATION_STATS.get(name, DEFAULT_STATS)
    mean, std = stats["mean"], stats["std"]
    if channels == 1 and len(mean) == 3:
        mean = [sum(mean) / 3]
        std = [sum(std) / 3]
    return mean, std


def extract_config(name: str, dataset) -> DatasetConfig:
    """Extract configuration from a loaded HuggingFace dataset."""
    name_lower = name.lower()

    num_classes = 0
    label_feature = dataset.features.get("label")
    if label_feature is not None and hasattr(label_feature, "num_classes"):
        num_classes = label_feature.num_classes

    sample = dataset[0]
    data_key = "image"
    num_frames = 1

    if "image" in sample:
        h, w, channels = _extract_image_dimensions(sample["image"])

    if "video" in sample:
        data_key = "video"
        video = sample["video"]
        if TORCHCODEC_AVAILABLE and isinstance(video, torchcodec.decoders.VideoDecoder):
            num_frames = video._num_frames
            frame = video.get_frame_at(0)
            channels, h, w = frame.data.shape
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
# Transform Builders (parameterized by type, not by model name)
# =============================================================================


def _build_view_transform(config: DatasetConfig):
    """Build a single augmented view transform."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key

    if config.low_resolution:
        return transforms.Compose(
            transforms.RGB(source=source, target=target),
            transforms.RandomResizedCrop(
                (h, w), scale=(0.2, 1.0), source=source, target=target
            ),
            transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8,
                source=source, target=target,
            ),
            transforms.RandomGrayscale(p=0.2, source=source, target=target),
            transforms.ToImage(**stats, source=source, target=target),
        )

    return transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.RandomResizedCrop(
            (h, w), scale=(0.08, 1.0), source=source, target=target
        ),
        transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8,
            source=source, target=target,
        ),
        transforms.RandomGrayscale(p=0.2, source=source, target=target),
        transforms.GaussianBlur(
            kernel_size=23, sigma=(0.1, 2.0), p=0.5, source=source, target=target
        ),
        transforms.ToImage(**stats, source=source, target=target),
    )


def _build_val_transform(config: DatasetConfig):
    """Build validation transform (resize + normalize)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key
    return transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.Resize((h, w), source=source, target=target),
        transforms.ToImage(**stats, source=source, target=target),
    )


def _create_multiview_transforms(config: DatasetConfig, num_views: int = 2):
    """Multi-view contrastive transforms (SimCLR, LeJEPA, NNCLR, BarlowTwins)."""
    view_transform = _build_view_transform(config)
    train_transform = transforms.MultiViewTransform([view_transform] * num_views)
    return train_transform, _build_val_transform(config)


def _create_single_transforms(config: DatasetConfig):
    """Single-view transforms with minimal augmentation (MAE)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key

    train_transform = transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.RandomResizedCrop(
            (h, w), scale=(0.2, 1.0), source=source, target=target
        ),
        transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
        transforms.ToImage(**stats, source=source, target=target),
    )
    return train_transform, _build_val_transform(config)


def _create_multicrop_transforms(
    config: DatasetConfig, num_global: int = 2, num_local: int = 6
):
    """Multi-crop transforms with global + local views (DINO)."""
    h, w = config.image_size
    stats = {"mean": config.mean, "std": config.std}
    source = target = config.data_key

    if config.low_resolution:
        num_local = 0

    global_crop_scale = (0.5, 1.0) if config.low_resolution else (0.4, 1.0)
    global_transform = transforms.Compose(
        transforms.RGB(source=source, target=target),
        transforms.RandomResizedCrop(
            (h, w), scale=global_crop_scale, source=source, target=target
        ),
        transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8,
            source=source, target=target,
        ),
        transforms.RandomGrayscale(p=0.2, source=source, target=target),
        transforms.GaussianBlur(
            kernel_size=23, sigma=(0.1, 2.0), p=1.0, source=source, target=target
        ),
        transforms.ToImage(**stats, source=source, target=target),
    )

    transform_dict = {}
    for i in range(num_global):
        transform_dict[f"global_{i + 1}"] = global_transform

    if num_local > 0:
        local_size = (max(h // 2, 32), max(w // 2, 32))
        local_transform = transforms.Compose(
            transforms.RGB(source=source, target=target),
            transforms.RandomResizedCrop(
                local_size, scale=(0.05, 0.4), source=source, target=target
            ),
            transforms.RandomHorizontalFlip(p=0.5, source=source, target=target),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8,
                source=source, target=target,
            ),
            transforms.RandomGrayscale(p=0.2, source=source, target=target),
            transforms.GaussianBlur(
                kernel_size=23, sigma=(0.1, 2.0), p=0.5, source=source, target=target
            ),
            transforms.ToImage(**stats, source=source, target=target),
        )
        for i in range(num_local):
            transform_dict[f"local_{i + 1}"] = local_transform

    train_transform = transforms.MultiViewTransform(transform_dict)
    return train_transform, _build_val_transform(config)


def create_transforms(ds_config: DatasetConfig, transform_cfg):
    """Create transforms from a model's transform config.

    Args:
        ds_config: Runtime dataset configuration.
        transform_cfg: OmegaConf node with keys: type, and type-specific params
            (num_views for multiview, num_global/num_local for multicrop).
    """
    t_type = transform_cfg.type

    if t_type == "multiview":
        return _create_multiview_transforms(ds_config, transform_cfg.get("num_views", 2))
    if t_type == "single":
        return _create_single_transforms(ds_config)
    if t_type == "multicrop":
        return _create_multicrop_transforms(
            ds_config, transform_cfg.get("num_global", 2), transform_cfg.get("num_local", 6)
        )

    raise ValueError(f"Unknown transform type: {t_type}. Use 'multiview', 'single', or 'multicrop'.")


# =============================================================================
# Dataset Creation
# =============================================================================


def _get_dataset_classes() -> dict[str, type]:
    return {
        name.lower(): cls
        for name, cls in vars(sds.images).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }


def _load_validation_split(dataset_cls, **kwargs):
    for split_name in ("test", "validation"):
        try:
            return dataset_cls(split=split_name, **kwargs)
        except (ValueError, KeyError):
            continue
    return None


def create_dataset(
    name: str,
    transform_cfg,
    training_cfg,
) -> tuple[spt.data.DataModule, DatasetConfig]:
    """Create a dataset with config-driven transforms.

    Args:
        name: Dataset name (case-insensitive).
        transform_cfg: Model's transform config (type + params).
        training_cfg: Training config with batch_size and num_workers.

    Returns:
        Tuple of (DataModule, DatasetConfig).
    """
    name_lower = name.lower()
    dataset_classes = _get_dataset_classes()

    if name_lower not in dataset_classes:
        available = ", ".join(sorted(dataset_classes.keys()))
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    dataset_cls = dataset_classes[name_lower]
    extra_kwargs = DATASET_KWARGS.get(name_lower, {})

    train_hf = dataset_cls(split="train", **extra_kwargs)
    val_hf = _load_validation_split(dataset_cls, **extra_kwargs)

    ds_config = extract_config(name_lower, train_hf)
    train_transform, val_transform = create_transforms(ds_config, transform_cfg)

    batch_size = training_cfg.batch_size
    num_workers = training_cfg.num_workers

    train_dataset = TransformDataset(train_hf, train_transform, data_key=ds_config.data_key)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_views,
    )

    val_loader = None
    if val_hf is not None:
        val_dataset = TransformDataset(val_hf, val_transform, data_key=ds_config.data_key)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_views,
        )

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    return data_module, ds_config
