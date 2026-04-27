"""Dataset loading for SSL benchmarks.

Loads image classification datasets from stable_datasets and wraps them
as a ``spt.data.DataModule``.  Transform and collation logic lives in
the individual model files (``models/*.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import stable_pretraining as spt
import numpy as np
import torch

import stable_datasets as sds
from stable_datasets.schema import ClassLabel


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
    modality: str = "image"
    builder_module: str = "images"
    builder_name: str | None = None
    builder_kwargs: dict[str, Any] = field(default_factory=dict)
    input_key: str = "image"
    label_key: str = "label"
    num_channels: int | None = None
    sequence_length: int | None = None
    include_in_results: bool = True


_IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _image(
    name,
    display_name,
    num_classes,
    *,
    channels,
    mean,
    std,
    builder_name=None,
    builder_kwargs=None,
    include_in_results=True,
):
    return DatasetConfig(
        name=name,
        display_name=display_name,
        num_classes=num_classes,
        channels=channels,
        mean=mean,
        std=std,
        builder_name=builder_name,
        builder_kwargs=dict(builder_kwargs or {}),
        num_channels=channels,
        include_in_results=include_in_results,
    )


def _rgb(name, display_name, num_classes, **kwargs):
    stats = {**_IMAGENET_STATS, **{k: kwargs.pop(k) for k in list(kwargs) if k in {"mean", "std"}}}
    return _image(name, display_name, num_classes, channels=3, **stats, **kwargs)


def _gray(name, display_name, num_classes, mean, std, **kwargs):
    return _image(name, display_name, num_classes, channels=1, mean=mean, std=std, **kwargs)


def _timeseries(
    name,
    display_name,
    num_classes,
    *,
    builder_name,
    num_channels,
    mean=None,
    std=None,
    sequence_length=None,
):
    return DatasetConfig(
        name=name,
        display_name=display_name,
        num_classes=num_classes,
        channels=num_channels,
        mean=list(mean or [0.0] * num_channels),
        std=list(std or [1.0] * num_channels),
        modality="timeseries",
        builder_module="timeseries",
        builder_name=builder_name,
        input_key="series",
        label_key="label",
        num_channels=num_channels,
        sequence_length=sequence_length,
    )


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # RGB datasets (3-channel, ImageNet stats unless overridden)
    "beans": _rgb("beans", "Beans", 3),
    "awa2": _rgb("awa2", "AWA2", 50),
    "cifar10": _rgb("cifar10", "CIFAR-10", 10, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]),
    "cifar100": _rgb("cifar100", "CIFAR-100", 100, mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    "country211": _rgb("country211", "Country-211", 211),
    "cub200": _rgb("cub200", "CUB-200", 200),
    "dtd": _rgb("dtd", "DTD", 47),
    "fgvcaircraft": _rgb("fgvcaircraft", "FGVC Aircraft", 100),
    "fgvcaircraft_family": _rgb(
        "fgvcaircraft_family",
        "FGVC Aircraft (Family)",
        70,
        builder_name="FGVCAircraft",
        builder_kwargs={"config_name": "family"},
    ),
    "fgvcaircraft_manufacturer": _rgb(
        "fgvcaircraft_manufacturer",
        "FGVC Aircraft (Manufacturer)",
        30,
        builder_name="FGVCAircraft",
        builder_kwargs={"config_name": "manufacturer"},
    ),
    "flowers102": _rgb("flowers102", "Flowers-102", 102, mean=[0.4344, 0.3830, 0.2954], std=[0.2937, 0.2458, 0.2726]),
    "food101": _rgb("food101", "Food-101", 101),
    "imagenet": _rgb("imagenet", "ImageNet", 1000, builder_name="ImageNet1K", include_in_results=False),
    "imagenette": _rgb("imagenette", "Imagenette", 10),
    "rockpaperscissor": _rgb("rockpaperscissor", "Rock-Paper-Scissors", 3),
    "stl10": _rgb("stl10", "STL-10", 10, mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
    "svhn": _rgb("svhn", "SVHN", 10, mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    "tinyimagenet": _rgb("tinyimagenet", "Tiny ImageNet", 200, builder_name="TinyImagenet", include_in_results=False),
    "cars196": _rgb(
        "cars196",
        "Cars196",
        196,
        mean=[0.4707, 0.4602, 0.4550],
        std=[0.2920, 0.2918, 0.2972],
    ),
    "galaxy10": _rgb(
        "galaxy10",
        "Galaxy10",
        10,
        builder_name="Galaxy10Decal",
        mean=[0.1851, 0.1431, 0.1086],
        std=[0.2166, 0.1820, 0.1571],
    ),
    "linnaeus5": _rgb(
        "linnaeus5",
        "Linnaeus 5",
        5,
        mean=[0.4459, 0.4528, 0.3350],
        std=[0.2731, 0.2659, 0.2797],
    ),
    "imagenet100": _rgb("imagenet100", "ImageNet-100", 100, builder_name="ImageNet100"),
    # Grayscale datasets (1-channel) — includes character/line-art datasets
    # whose images are visually binary (ink-on-paper) regardless of source channel count.
    "arabiccharacters": _gray("arabiccharacters", "Arabic Characters", 28, mean=[0.1030], std=[0.3040]),
    "arabicdigits": _gray("arabicdigits", "Arabic Digits", 10, mean=[0.1072], std=[0.2832]),
    "emnist": _gray("emnist", "EMNIST", 47, mean=[0.1751], std=[0.3332]),
    "emnist_byclass": _gray(
        "emnist_byclass",
        "EMNIST ByClass",
        62,
        mean=[0.1751],
        std=[0.3332],
        builder_name="EMNIST",
        builder_kwargs={"config_name": "byclass"},
    ),
    "emnist_bymerge": _gray(
        "emnist_bymerge",
        "EMNIST ByMerge",
        47,
        mean=[0.1751],
        std=[0.3332],
        builder_name="EMNIST",
        builder_kwargs={"config_name": "bymerge"},
    ),
    "emnist_letters": _gray(
        "emnist_letters",
        "EMNIST Letters",
        26,
        mean=[0.1751],
        std=[0.3332],
        builder_name="EMNIST",
        builder_kwargs={"config_name": "letters"},
    ),
    "emnist_digits": _gray(
        "emnist_digits",
        "EMNIST Digits",
        10,
        mean=[0.1751],
        std=[0.3332],
        builder_name="EMNIST",
        builder_kwargs={"config_name": "digits"},
    ),
    "emnist_mnist": _gray(
        "emnist_mnist",
        "EMNIST MNIST",
        10,
        mean=[0.1751],
        std=[0.3332],
        builder_name="EMNIST",
        builder_kwargs={"config_name": "mnist"},
    ),
    "fashionmnist": _gray("fashionmnist", "FashionMNIST", 10, mean=[0.2860], std=[0.3530]),
    "kmnist": _gray("kmnist", "KMNIST", 10, mean=[0.1918], std=[0.3483]),
    "medmnist": _gray("medmnist", "MedMNIST", 2, mean=[0.4823], std=[0.2363]),
    "pathmnist": _rgb(
        "pathmnist",
        "PathMNIST",
        9,
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "pathmnist"},
    ),
    "dermamnist": _rgb(
        "dermamnist",
        "DermaMNIST",
        7,
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "dermamnist"},
    ),
    "octmnist": _gray(
        "octmnist",
        "OCTMNIST",
        4,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "octmnist"},
    ),
    "retinamnist": _rgb(
        "retinamnist",
        "RetinaMNIST",
        5,
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "retinamnist"},
    ),
    "breastmnist": _gray(
        "breastmnist",
        "BreastMNIST",
        2,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "breastmnist"},
    ),
    "bloodmnist": _rgb(
        "bloodmnist",
        "BloodMNIST",
        8,
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "bloodmnist"},
    ),
    "tissuemnist": _gray(
        "tissuemnist",
        "TissueMNIST",
        8,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "tissuemnist"},
    ),
    "organamnist": _gray(
        "organamnist",
        "OrganAMNIST",
        11,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "organamnist"},
    ),
    "organcmnist": _gray(
        "organcmnist",
        "OrganCMNIST",
        11,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "organcmnist"},
    ),
    "organsmnist": _gray(
        "organsmnist",
        "OrganSMNIST",
        11,
        mean=[0.4823],
        std=[0.2363],
        builder_name="MedMNIST",
        builder_kwargs={"config_name": "organsmnist"},
    ),
    "notmnist": _gray("notmnist", "NotMNIST", 10, mean=[0.4178], std=[0.4532]),
    "hasyv2": _gray(
        "hasyv2",
        "HASYv2",
        369,
        mean=[0.9275],
        std=[0.2389],
        builder_kwargs={"config_name": "fold-1"},
    ),
    # Timeseries builders are registered for metadata/loading but are not part
    # of dataset=all until a timeseries backbone/model path exists.
    "catsdogs": _timeseries("catsdogs", "CatsDogs", 2, builder_name="CatsDogs", num_channels=1),
    "japanesevowels": _timeseries(
        "japanesevowels",
        "Japanese Vowels",
        9,
        builder_name="JapaneseVowels",
        num_channels=12,
        sequence_length=29,
    ),
    "mosquitosound": _timeseries("mosquitosound", "Mosquito Sound", 2, builder_name="MosquitoSound", num_channels=1),
    "phoneme": _timeseries("phoneme", "Phoneme", 39, builder_name="Phoneme", num_channels=1, sequence_length=512),
    "urbansound": _timeseries("urbansound", "UrbanSound", 10, builder_name="UrbanSound", num_channels=1),
}

DATASET_CONFIGS["emnist"].builder_kwargs = {"config_name": "balanced"}
DATASET_CONFIGS["medmnist"].builder_kwargs = {"config_name": "pneumoniamnist"}

IMAGE_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    name: cfg for name, cfg in DATASET_CONFIGS.items() if cfg.modality == "image"
}
TIMESERIES_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    name: cfg for name, cfg in DATASET_CONFIGS.items() if cfg.modality == "timeseries"
}
INCLUDED_IMAGE_DATASETS: set[str] = {
    name for name, cfg in IMAGE_DATASET_CONFIGS.items() if cfg.include_in_results
}
INCLUDED_TIMESERIES_DATASETS: set[str] = {
    name for name, cfg in TIMESERIES_DATASET_CONFIGS.items() if cfg.include_in_results
}


# Dataset loading


def _get_dataset_classes(module_name: str) -> dict[str, type]:
    module = getattr(sds, module_name)
    return {
        name.lower(): cls
        for name, cls in vars(module).items()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
    }


def _get_dataset_class(ds_config: DatasetConfig) -> type:
    dataset_classes = _get_dataset_classes(ds_config.builder_module)
    builder_key = (ds_config.builder_name or ds_config.name).lower()
    if builder_key not in dataset_classes:
        available = ", ".join(sorted(dataset_classes.keys()))
        raise ValueError(f"Unknown dataset builder: {builder_key}. Available: {available}")
    return dataset_classes[builder_key]


def _is_labeled_split(ds, ds_config: DatasetConfig, max_checks: int = 32) -> bool:
    label_feature = ds.features.get(ds_config.label_key)
    if not isinstance(label_feature, ClassLabel):
        return False

    for i in range(min(max_checks, len(ds))):
        label = ds[i].get(ds_config.label_key)
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            return False
        if label_int < 0 or label_int >= ds_config.num_classes:
            return False
    return True


def _load_validation_split(dataset_cls, ds_config: DatasetConfig, **kwargs):
    for split in ("validation", "valid"):
        try:
            return dataset_cls(split=split, **kwargs)
        except (ValueError, KeyError):
            pass

    try:
        test_ds = dataset_cls(split="test", **kwargs)
    except (ValueError, KeyError):
        return None

    if _is_labeled_split(test_ds, ds_config):
        return test_ds
    log.warning(f"Ignoring unlabeled test split for '{ds_config.name}'.")
    return None


def get_config(name: str) -> DatasetConfig:
    """Look up static config for a dataset by name.

    Falls back to runtime extraction if the dataset isn't in the static table.
    """
    name_lower = name.lower()
    if name_lower in DATASET_CONFIGS:
        return DATASET_CONFIGS[name_lower]

    raise ValueError(f"Unknown dataset: '{name}'. Available: {', '.join(sorted(DATASET_CONFIGS.keys()))}")


def get_image_dataset_names(*, include_results_only: bool = False) -> list[str]:
    """Return registered image dataset ids, excluding timeseries configs."""
    names = INCLUDED_IMAGE_DATASETS if include_results_only else set(IMAGE_DATASET_CONFIGS)
    return sorted(names)


def _with_data_dirs(ds_config: DatasetConfig, data_dir: str | None) -> dict[str, Any]:
    extra_kwargs = dict(ds_config.builder_kwargs)
    if data_dir is not None:
        root = Path(data_dir)
        extra_kwargs["download_dir"] = str(root / "downloads")
        extra_kwargs["processed_cache_dir"] = str(root / "processed")
    return extra_kwargs


def _timeseries_transform(ds_config: DatasetConfig):
    mean = np.asarray(ds_config.mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(ds_config.std, dtype=np.float32).reshape(1, -1)

    def transform(sample):
        series = np.asarray(sample[ds_config.input_key], dtype=np.float32)
        if series.ndim == 1:
            series = series[:, None]
        if series.ndim != 2:
            raise ValueError(f"Expected 1D/2D timeseries, got shape {series.shape}")

        if ds_config.sequence_length is not None:
            target_len = ds_config.sequence_length
            clipped = series[:target_len]
            mask = np.zeros(target_len, dtype=bool)
            mask[: len(clipped)] = True
            if len(clipped) < target_len:
                pad = np.zeros((target_len - len(clipped), clipped.shape[1]), dtype=np.float32)
                series = np.concatenate([clipped, pad], axis=0)
            else:
                series = clipped
        else:
            mask = np.ones(series.shape[0], dtype=bool)

        series = (series - mean) / np.maximum(std, 1e-6)
        return {
            "series": torch.from_numpy(series.T.copy()),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(int(sample[ds_config.label_key]), dtype=torch.long),
        }

    return transform


def collate_timeseries(batch):
    """Collate variable-length timeseries as [B, C, T] with a boolean mask."""
    max_len = max(sample["series"].shape[-1] for sample in batch)
    channels = batch[0]["series"].shape[0]
    series = torch.zeros(len(batch), channels, max_len, dtype=batch[0]["series"].dtype)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, sample in enumerate(batch):
        length = sample["series"].shape[-1]
        series[i, :, :length] = sample["series"]
        mask[i, :length] = sample.get("mask", torch.ones(length, dtype=torch.bool))[:length]
        labels[i] = sample["label"]
    return {"series": series, "mask": mask, "label": labels}


def create_dataset(
    name: str,
    train_transform,
    val_transform,
    collate_fn,
    training_cfg,
    data_dir: str | None = None,
) -> tuple[spt.data.DataModule, DatasetConfig]:
    """Load a dataset and wrap it as a DataModule.

    ``collate_fn`` is used for the train loader. The val loader always
    uses ``collate_single`` because validation samples are single-view
    regardless of the training multi-view configuration.

    Args:
        name: Dataset name (case-insensitive).
        train_transform: Transform applied to each training sample.
        val_transform: Transform applied to each validation sample.
        collate_fn: Collation function for the *train* DataLoader.
        training_cfg: OmegaConf node with batch_size and num_workers.
        data_dir: Root directory for HF downloads/cache.

    Returns:
        Tuple of (DataModule, DatasetConfig).
    """
    name_lower = name.lower()
    ds_config = get_config(name_lower)
    dataset_cls = _get_dataset_class(ds_config)
    extra_kwargs = _with_data_dirs(ds_config, data_dir)

    log.info(f"Loading train split for '{name_lower}'...")
    train_hf = dataset_cls(split="train", **extra_kwargs)
    log.info(f"Loading validation split for '{name_lower}'...")
    val_hf = _load_validation_split(dataset_cls, ds_config, **extra_kwargs)

    if val_hf is None:
        log.warning(f"No test/validation split for '{name_lower}'; holding out 10%% of train.")
        splits = train_hf.train_test_split(test_size=0.1, seed=42)
        train_hf = splits["train"]
        val_hf = splits["test"]

    batch_size = training_cfg.batch_size
    num_workers = training_cfg.num_workers
    prefetch_factor = getattr(training_cfg, "prefetch_factor", 2) if num_workers > 0 else None
    mp_ctx = "fork" if num_workers > 0 else None

    if ds_config.modality == "timeseries":
        train_transform = train_transform or _timeseries_transform(ds_config)
        val_transform = val_transform or _timeseries_transform(ds_config)
        collate_fn = collate_fn or collate_timeseries
        val_collate_fn = collate_timeseries
    else:
        from benchmarks.models import collate_single

        val_collate_fn = collate_single

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
        collate_fn=val_collate_fn,
        multiprocessing_context=mp_ctx,
        prefetch_factor=prefetch_factor,
    )

    return spt.data.DataModule(train=train_loader, val=val_loader), ds_config
