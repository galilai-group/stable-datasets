"""
Additions to stable_datasets/benchmarks/self_supervised/dataset.py
for cached dataset support.

INTEGRATION INSTRUCTIONS:
1. Add the new imports, constant, and CachedDataset class after the existing
   TransformDataset class.
2. Replace the existing `create_dataset` function with the new version below.

Everything else (transforms, collate, extract_config) stays unchanged.
"""

# ===========================================================================
# NEW: Add these imports near the top of dataset.py
# ===========================================================================
# import json
# from pathlib import Path
# from PIL import Image as PILImage

# ===========================================================================
# NEW: Add this constant after DATASET_KWARGS
# ===========================================================================

CACHE_DIR_DEFAULT = "/mnt/data/sami/stable-datasets/.cached_224"


# ===========================================================================
# NEW: Add this class after TransformDataset
# ===========================================================================


class CachedDataset(torch.utils.data.Dataset):
    """Dataset backed by pre-cached 224x224 uint8 numpy memmaps.

    Eliminates HF Arrow deserialization, PIL decode, and resize overhead.
    Returns samples as {"image": PIL.Image, "label": int} so existing
    transform chains work without any modification.
    """

    def __init__(self, images_path: str, labels_path: str, transform, data_key: str = "image"):
        # Memory-mapped read: no RAM cost, just pages in on access
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.transform = transform
        self.data_key = data_key

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # .copy() is required: memmap returns a read-only view, and PIL/torch
        # transforms may need to write. On a 224x224x3 array this is ~150KB,
        # taking < 30 microseconds — negligible vs the old PIL decode path.
        img = PILImage.fromarray(self.images[idx].copy())
        label = int(self.labels[idx])

        transform_input = {self.data_key: img}
        transform_input["label"] = label
        output = self.transform(transform_input)
        return self._normalize_keys(output)

    def _normalize_keys(self, output):
        """Same normalization as TransformDataset."""
        if self.data_key == "image":
            return output
        if "views" in output:
            for view in output["views"]:
                if self.data_key in view:
                    view["image"] = view.pop(self.data_key)
        elif self.data_key in output:
            output["image"] = output.pop(self.data_key)
        return output


# ===========================================================================
# NEW: Helper to load DatasetConfig from cached metadata
# ===========================================================================


def _load_cached_config(name: str, cache_dir: str) -> DatasetConfig | None:
    """Try to load DatasetConfig from cached metadata.json.

    Returns None if cache doesn't exist for this dataset.
    """
    metadata_path = Path(cache_dir) / name / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        meta = json.load(f)

    channels = meta["original_channels"]
    mean, std = _get_normalization_stats(name, channels)

    return DatasetConfig(
        name=name,
        num_classes=meta["num_classes"],
        image_size=tuple(meta["image_size"]),
        channels=channels,
        mean=mean,
        std=std,
        num_frames=1,
        data_key="image",
    )


# ===========================================================================
# MODIFIED: Replace the existing create_dataset with this version
# ===========================================================================


def create_dataset(
    name: str,
    transform_cfg,
    training_cfg,
    data_dir: str | None = None,
    cache_dir: str | None = CACHE_DIR_DEFAULT,
) -> tuple:  # tuple[spt.data.DataModule, DatasetConfig]
    """Create a dataset with config-driven transforms.

    If a pre-cached version exists at `cache_dir/{name}/`, uses fast numpy
    memmap loading instead of HF Arrow deserialization. Falls back to the
    original HF loading path if no cache is found.

    Args:
        name: Dataset name (case-insensitive).
        transform_cfg: Model's transform config (type + params).
        training_cfg: Training config with batch_size and num_workers.
        data_dir: Root directory for HF downloads/cache.
        cache_dir: Root directory for pre-cached 224x224 numpy files.
            Set to None to disable caching and always use HF loading.

    Returns:
        Tuple of (DataModule, DatasetConfig).
    """
    name_lower = name.lower()

    # ------------------------------------------------------------------
    # Fast path: load from pre-cached numpy memmaps
    # ------------------------------------------------------------------
    if cache_dir is not None:
        ds_config = _load_cached_config(name_lower, cache_dir)
        if ds_config is not None:
            cache_path = Path(cache_dir) / name_lower
            train_images = str(cache_path / "train_images.npy")
            train_labels = str(cache_path / "train_labels.npy")
            val_images = str(cache_path / "val_images.npy")
            val_labels = str(cache_path / "val_labels.npy")

            if Path(train_images).exists() and Path(val_images).exists():
                log.info(f"Using cached dataset for '{name_lower}' from {cache_path}")

                train_transform, val_transform = create_transforms(ds_config, transform_cfg)
                batch_size = training_cfg.batch_size
                num_workers = training_cfg.num_workers

                train_dataset = CachedDataset(
                    train_images, train_labels, train_transform,
                    data_key=ds_config.data_key,
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=_collate_views,
                    multiprocessing_context="fork" if num_workers > 0 else None,
                    persistent_workers=num_workers > 0,
                    pin_memory=True,
                )

                val_dataset = CachedDataset(
                    val_images, val_labels, val_transform,
                    data_key=ds_config.data_key,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=_collate_views,
                    multiprocessing_context="fork" if num_workers > 0 else None,
                    persistent_workers=num_workers > 0,
                    pin_memory=True,
                )

                data_module = spt.data.DataModule(train=train_loader, val=val_loader)
                return data_module, ds_config

    # ------------------------------------------------------------------
    # Slow path: original HF loading (unchanged)
    # ------------------------------------------------------------------
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
        log.warning(
            f"No test/validation split for '{name_lower}'; "
            f"holding out 10%% of train as validation."
        )
        splits = train_hf.train_test_split(test_size=0.1, seed=42)
        train_hf = splits["train"]
        val_hf = splits["test"]

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
        multiprocessing_context="fork" if num_workers > 0 else None,
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
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    return data_module, ds_config