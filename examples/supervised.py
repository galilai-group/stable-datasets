"""
Supervised Learning Example with stable-datasets
=================================================

This example demonstrates how to train models using supervised learning
with stable-pretraining, using datasets from stable-datasets.
"""

import argparse
import fcntl
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from stable_pretraining.data import transforms
from transformers import AutoConfig, AutoModelForImageClassification


# Set SLURM_NTASKS_PER_NODE if SLURM_NTASKS is set but SLURM_NTASKS_PER_NODE is not
# This prevents Lightning from erroring when it detects SLURM but can't find the expected variable
if "SLURM_NTASKS" in os.environ and "SLURM_NTASKS_PER_NODE" not in os.environ:
    if "SLURM_NNODES" in os.environ:
        # Calculate tasks per node
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))
        os.environ["SLURM_NTASKS_PER_NODE"] = str(ntasks // nnodes)
    else:
        # If we can't determine nodes, just set it to the same as NTASKS
        os.environ["SLURM_NTASKS_PER_NODE"] = os.environ.get("SLURM_NTASKS", "1")


def get_dataset_class(dataset_name: str):
    """Dynamically load dataset class from stable_datasets.images."""
    import importlib

    try:
        module = importlib.import_module("stable_datasets.images")
        dataset_class = getattr(module, dataset_name)
        return dataset_class
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Dataset '{dataset_name}' not found in stable_datasets.images. Error: {e}")


def get_hyperparams_dict(image_size, batch_size, lr, weight_decay, max_epochs, seed, config_name=None):
    """Generate hyperparameters dictionary."""
    hyperparams = {
        "image_size": image_size,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "seed": seed,
    }
    if config_name is not None:
        hyperparams["config_name"] = config_name
    return hyperparams


def load_results(results_file="results.json", max_retries=10, retry_delay=0.1):
    """Load results from JSON file with file locking to prevent race conditions."""
    results_path = Path(results_file)
    if not results_path.exists():
        return {}

    for attempt in range(max_retries):
        try:
            with open(results_path) as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                print(f"Warning: Failed to load results after {max_retries} attempts: {e}")
                return {}
    return {}


def save_results(results, results_file="results.json", max_retries=10, retry_delay=0.1):
    """Save results to JSON file with file locking to prevent race conditions."""
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Use atomic write: write to temp file first, then rename
    temp_path = results_path.with_suffix(".tmp")

    for attempt in range(max_retries):
        try:
            # Write to temp file with exclusive lock
            with open(temp_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(results, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename (rename is atomic on Unix)
            temp_path.replace(results_path)
            return

        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                print(f"Error: Failed to save results after {max_retries} attempts: {e}")
                raise


def get_num_classes(dataset):
    """Get number of classes from dataset features.

    Supports ClassLabel created with either names= or num_classes= parameter.
    ClassLabel instances in HuggingFace datasets always have a num_classes property.
    """
    if not hasattr(dataset, "features"):
        raise ValueError("Dataset does not have 'features' attribute")

    if "label" not in dataset.features:
        raise ValueError(
            "Dataset does not have 'label' feature. "
            "This script requires a classification dataset with a 'label' field."
        )

    label_feature = dataset.features["label"]

    # ClassLabel always has num_classes property (works for both names= and num_classes= cases)
    if hasattr(label_feature, "num_classes"):
        return int(label_feature.num_classes)

    # Fall back to names length if num_classes is not available
    if hasattr(label_feature, "names") and label_feature.names is not None:
        return len(label_feature.names)

    raise ValueError("Could not determine number of classes from dataset label feature")


def compute_normalization_stats(dataset, sample_size=10000, seed=42):
    """Compute mean and standard deviation for normalization from a dataset.

    Args:
        dataset: HuggingFace dataset containing 'image' field
        sample_size: Number of samples to use for computation (default: 10000)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (mean, std) as lists of length 3 for RGB channels
    """
    print(f"Computing normalization statistics from {min(sample_size, len(dataset))} samples...")

    mean_sum = np.zeros(3, dtype=np.float64)
    std_sum = np.zeros(3, dtype=np.float64)
    count = 0

    # Sample subset for faster computation
    actual_sample_size = min(sample_size, len(dataset))
    random.seed(seed)
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), actual_sample_size, replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        img = sample["image"]
        if not isinstance(img, Image.Image):
            # Convert to PIL if needed
            img = Image.fromarray(img)
        img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        mean_sum += img_array.mean(axis=(0, 1))
        std_sum += img_array.std(axis=(0, 1))
        count += 1

    mean = (mean_sum / count).tolist()
    std = (std_sum / count).tolist()
    print(f"Computed mean: {mean}, std: {std}")

    return mean, std


class HFDatasetWrapper(spt.data.Dataset):
    """Wrapper for pre-loaded HuggingFace datasets with transform support."""

    def __init__(self, hf_dataset, transform=None):
        super().__init__(transform)
        self.hf_dataset = hf_dataset
        # Add sample_idx if not present
        if "sample_idx" not in hf_dataset.column_names:
            self.hf_dataset = hf_dataset.add_column("sample_idx", list(range(hf_dataset.num_rows)))

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        return self.process_sample(sample)

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def column_names(self):
        return self.hf_dataset.column_names


def get_data_loaders(args, dataset_class, seed=42, config_name=None):
    """Get train, validation, and test data loaders for the specified dataset.

    Handles cases where dataset has:
    - Only train split: splits 80% train, 10% val, 10% test
    - Only test split: splits 80% train, 10% val, 10% test
    - Both train and test: uses test as-is, splits train 80/20 (train/val)
    - All three splits: uses them as-is

    Args:
        args: Arguments object containing dataset configuration
        dataset_class: Dataset class to instantiate
        seed: Random seed for data splitting (default: 42)
        config_name: Optional config name for datasets with multiple configurations
            (e.g., 'balanced' for EMNIST, 'pathmnist' for MedMNIST)
    """
    # Load the dataset to check available splits
    try:
        if config_name is not None:
            all_splits = dataset_class(split=None, config_name=config_name)
        else:
            all_splits = dataset_class(split=None)
    except Exception:
        # If split=None fails, try to detect available splits manually
        all_splits = {}
        for split_name in ["train", "test", "validation", "val", "valid"]:
            try:
                if config_name is not None:
                    all_splits[split_name] = dataset_class(split=split_name, config_name=config_name)
                else:
                    all_splits[split_name] = dataset_class(split=split_name)
            except (ValueError, KeyError):
                continue

    # Determine available splits
    has_train = False
    has_test = False
    has_validation = False
    validation_split_name = None

    if isinstance(all_splits, dict):
        has_train = "train" in all_splits
        has_test = "test" in all_splits
        # Check for validation split
        for split_name in ["validation", "val", "valid"]:
            if split_name in all_splits:
                validation_split_name = split_name
                has_validation = True
                break
    else:
        # Fallback: try loading individually
        try:
            if config_name is not None:
                all_splits = {"train": dataset_class(split="train", config_name=config_name)}
            else:
                all_splits = {"train": dataset_class(split="train")}
            has_train = True
        except (ValueError, KeyError):
            pass
        try:
            if not isinstance(all_splits, dict):
                all_splits = {}
            if config_name is not None:
                all_splits["test"] = dataset_class(split="test", config_name=config_name)
            else:
                all_splits["test"] = dataset_class(split="test")
            has_test = True
        except (ValueError, KeyError):
            pass
        # Try validation splits
        for split_name in ["validation", "val", "valid"]:
            try:
                if not isinstance(all_splits, dict):
                    all_splits = {}
                if config_name is not None:
                    all_splits[split_name] = dataset_class(split=split_name, config_name=config_name)
                else:
                    all_splits[split_name] = dataset_class(split=split_name)
                validation_split_name = split_name
                has_validation = True
                break
            except (ValueError, KeyError):
                continue

    # Handle different split scenarios
    if has_validation and has_train and has_test:
        # All three splits exist - use them as-is
        train_dataset_raw = all_splits["train"]
        val_dataset_raw = all_splits[validation_split_name]
        test_dataset_raw = all_splits["test"]
        print("Using existing train/val/test splits")
    elif has_train and has_test:
        # Both train and test exist - use test as-is, split train 80/20
        train_dataset_raw = all_splits["train"]
        test_dataset_raw = all_splits["test"]
        # Split train: 80% train, 20% val
        split_dict = train_dataset_raw.train_test_split(test_size=0.2, seed=seed)
        train_dataset_raw = split_dict["train"]
        val_dataset_raw = split_dict["test"]
        print("Using existing train/test splits, splitting train 80/20 for train/val")
    elif has_train:
        # Only train exists - split 80/10/10
        train_dataset_raw = all_splits["train"]
        # First split: 80% train, 20% temp
        split_dict = train_dataset_raw.train_test_split(test_size=0.2, seed=seed)
        train_dataset_raw = split_dict["train"]
        temp_dataset = split_dict["test"]
        # Second split: split temp 50/50 for val/test (10% each of original)
        split_dict2 = temp_dataset.train_test_split(test_size=0.5, seed=seed)
        val_dataset_raw = split_dict2["train"]
        test_dataset_raw = split_dict2["test"]
        print("Only train split available, splitting 80/10/10 for train/val/test")
    elif has_test:
        # Only test exists - split 80/10/10
        test_dataset_raw = all_splits["test"]
        # First split: 80% train, 20% temp
        split_dict = test_dataset_raw.train_test_split(test_size=0.2, seed=seed)
        train_dataset_raw = split_dict["train"]
        temp_dataset = split_dict["test"]
        # Second split: split temp 50/50 for val/test (10% each of original)
        split_dict2 = temp_dataset.train_test_split(test_size=0.5, seed=seed)
        val_dataset_raw = split_dict2["train"]
        test_dataset_raw = split_dict2["test"]
        print("Only test split available, splitting 80/10/10 for train/val/test")
    else:
        raise ValueError(f"Dataset {dataset_class.__name__} must have at least 'train' or 'test' split")

    # Infer number of classes from the dataset
    num_classes = get_num_classes(train_dataset_raw)

    # Get image size from config
    image_size = args.image_size

    # Compute normalization statistics from training set
    mean, std = compute_normalization_stats(train_dataset_raw, seed=seed)

    # Scale GaussianBlur kernel size with image size
    # Use (3, 3) for small images, (5, 5) for larger images
    blur_kernel_size = (3, 3) if image_size <= 64 else (5, 5)

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=blur_kernel_size, p=0.5),
        transforms.ToImage(mean=mean, std=std),
    )

    # Wrap the HuggingFace dataset for stable-pretraining
    train_dataset = HFDatasetWrapper(
        hf_dataset=train_dataset_raw,
        transform=train_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Validation transform: use CenterCrop for larger images to avoid distortion
    if image_size >= 224:
        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((256, 256)),  # Resize to slightly larger than target
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToImage(mean=mean, std=std),
        )
    else:
        # For smaller images, just resize
        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((image_size, image_size)),
            transforms.ToImage(mean=mean, std=std),
        )

    val_dataset = HFDatasetWrapper(
        hf_dataset=val_dataset_raw,
        transform=val_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create test loader
    test_dataset = HFDatasetWrapper(
        hf_dataset=test_dataset_raw,
        transform=val_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader, num_classes


def main(args):
    # Create results keys
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    dataset_name = args.dataset.lower()
    hyperparams = get_hyperparams_dict(
        args.image_size, args.batch_size, args.lr, args.weight_decay, args.max_epochs, args.seed, args.config_name
    )

    # Create a unique key for this hyperparams combination (sorted tuple for consistency)
    hyperparams_key = tuple(sorted(hyperparams.items()))

    # Check if results already exist
    results_file = args.results_file
    results = load_results(results_file)

    # Check if this exact model/dataset/hyperparams combination already exists
    model_results = results.get(model_name, {})
    if dataset_name in model_results:
        dataset_results = model_results[dataset_name]
        # Check if it's a dict with hyperparams key (new format) or old format (single entry)
        if isinstance(dataset_results, dict) and "entries" in dataset_results:
            # New format: multiple entries per dataset
            entries = dataset_results["entries"]
            for entry in entries:
                existing_hyperparams = entry.get("hyperparams", {})
                existing_key = tuple(sorted(existing_hyperparams.items()))
                if existing_key == hyperparams_key:
                    print(f"Results already exist for {model_name}/{dataset_name} with matching hyperparameters:")
                    print(json.dumps(entry, indent=2))
                    if not args.force_rerun:
                        print("Skipping training. Use --force_rerun to override.")
                        return
                    else:
                        print("--force_rerun specified, continuing with training...")
                        break
        else:
            # Old format: single entry, check if hyperparams match
            existing_entry = dataset_results if isinstance(dataset_results, dict) else {}
            existing_hyperparams = existing_entry.get("hyperparams", {})
            existing_key = tuple(sorted(existing_hyperparams.items()))
            if existing_key == hyperparams_key:
                print(f"Results already exist for {model_name}/{dataset_name} with matching hyperparameters:")
                print(json.dumps(existing_entry, indent=2))
                if not args.force_rerun:
                    print("Skipping training. Use --force_rerun to override.")
                    return
                else:
                    print("--force_rerun specified, continuing with training...")

    # Load dataset class
    dataset_class = get_dataset_class(args.dataset)

    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args, dataset_class, seed=args.seed, config_name=args.config_name
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader, test=test_loader)

    # Define forward function
    def forward(self, batch, stage):
        batch["logits"] = self.backbone(batch["image"])["logits"]

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            batch["logits"],
            batch["label"],
        )
        batch["loss"] = loss

        # Log loss
        if stage == "fit":
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "validate":
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        elif stage == "test":
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log accuracy for validation/test
        if stage in ["validate", "test"]:
            preds = torch.argmax(batch["logits"], dim=1)
            # Update metric (accumulates correctly across batches for epoch-level accuracy)
            self.val_accuracy(preds, batch["label"])
            # Log the metric - Lightning will compute epoch-level value automatically
            # Use consistent naming: "val_accuracy" for validation, "test_accuracy" for test
            metric_name = "val_accuracy" if stage == "validate" else "test_accuracy"
            self.log(metric_name, self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return batch

    # Create backbone
    config = AutoConfig.from_pretrained(args.model)
    backbone = AutoModelForImageClassification.from_config(config)
    backbone = spt.backbone.utils.set_embedding_dim(backbone, num_classes)

    # Create module
    hparams = {
        "model": args.model,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
    }

    # Use multi-optimizer format (even with single optimizer) to ensure Lightning returns a list
    # Add accuracy metric as module attribute for proper epoch-level computation
    module = spt.Module(
        backbone=backbone,
        forward=forward,
        hparams=hparams,
        val_accuracy=torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        optim={
            "optimizer": partial(
                torch.optim.AdamW,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            "scheduler": "LinearWarmupCosineAnnealing",
        },
    )

    # Setup trainer
    lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(
        logging_interval="step", log_momentum=True, log_weight_decay=True
    )

    # Create run name from model and dataset
    run_name = f"{model_name}-{args.dataset.lower()}"

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
    )

    # Configure WandB to use epoch as x-axis
    logger.experiment.define_metric("*", step_metric="epoch")

    # Log hyperparameters to wandb
    logger.log_hyperparams(hparams)

    # Setup checkpoint callback to save best model based on validation accuracy
    # Include model and dataset name in filename, but don't include epoch/accuracy to avoid multiple checkpoints
    checkpoint_filename = f"best-{model_name}-{args.dataset.lower()}"
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename=checkpoint_filename,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_sanity_val_steps=1,
        callbacks=[lr_monitor, checkpoint_callback],
        precision="16-mixed",
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=True,
    )

    # Train and validate
    manager = spt.Manager(trainer=trainer, module=module, data=data_module, seed=args.seed)
    manager()

    # Load best checkpoint and evaluate on test set
    if checkpoint_callback.best_model_path and checkpoint_callback.best_model_path != "":
        print(f"Loading best checkpoint: {checkpoint_callback.best_model_path}")
        # Load best module from checkpoint
        best_module = spt.Module.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            backbone=backbone,
            forward=forward,
            hparams=hparams,
            val_accuracy=torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            optim={
                "optimizer": partial(
                    torch.optim.AdamW,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                ),
                "scheduler": "LinearWarmupCosineAnnealing",
            },
        )

        # Evaluate on test set with best checkpoint using the same data_module
        test_manager = spt.Manager(trainer=trainer, module=best_module, data=data_module)
        # Set _trainer manually since we're not calling __call__ which would set it
        test_manager._trainer = trainer
        test_manager.test()
        test_trainer = test_manager._trainer
    else:
        print("No checkpoint saved, evaluating on test set with current model")
        test_manager = spt.Manager(trainer=trainer, module=module, data=data_module)
        # Set _trainer manually since we're not calling __call__ which would set it
        test_manager._trainer = trainer
        test_manager.test()
        test_trainer = test_manager._trainer

    # Extract test_accuracy from trainer metrics (most reliable after test completes)
    test_accuracy = None

    # Debug: print available metrics
    print(
        f"Available logged_metrics: {list(test_trainer.logged_metrics.keys()) if hasattr(test_trainer, 'logged_metrics') else 'None'}"
    )
    print(
        f"Available callback_metrics: {list(test_trainer.callback_metrics.keys()) if hasattr(test_trainer, 'callback_metrics') else 'None'}"
    )

    # Check trainer metrics first (where Lightning stores final test results)
    if hasattr(test_trainer, "logged_metrics") and "test_accuracy" in test_trainer.logged_metrics:
        value = test_trainer.logged_metrics["test_accuracy"]
        if torch.is_tensor(value):
            value = value.item()
        test_accuracy = float(value)
    elif hasattr(test_trainer, "callback_metrics") and "test_accuracy" in test_trainer.callback_metrics:
        value = test_trainer.callback_metrics["test_accuracy"]
        if torch.is_tensor(value):
            value = value.item()
        test_accuracy = float(value)

    if test_accuracy is None:
        print("Warning: test_accuracy not found in metrics")
        test_accuracy = 0.0

    # Save results in nested structure: model -> dataset -> entries (list)
    result_entry = {
        "hyperparams": hyperparams,
        "test_accuracy": test_accuracy,
    }

    # Reload results right before saving to merge with any concurrent updates
    # This prevents overwriting results from other jobs that finished during training
    results = load_results(results_file)

    # Initialize nested structure if needed
    if model_name not in results:
        results[model_name] = {}

    if dataset_name not in results[model_name]:
        # First entry for this dataset
        results[model_name][dataset_name] = {"entries": [result_entry]}
    else:
        # Check if we need to update existing entry or add new one
        dataset_results = results[model_name][dataset_name]
        if isinstance(dataset_results, dict) and "entries" in dataset_results:
            # New format: list of entries
            entries = dataset_results["entries"]
            # Check if entry with same hyperparams exists
            found = False
            for i, entry in enumerate(entries):
                existing_hyperparams = entry.get("hyperparams", {})
                existing_key = tuple(sorted(existing_hyperparams.items()))
                if existing_key == hyperparams_key:
                    # Update existing entry
                    entries[i] = result_entry
                    found = True
                    break
            if not found:
                # Add new entry
                entries.append(result_entry)
        else:
            # Old format: convert to new format
            existing_entry = dataset_results if isinstance(dataset_results, dict) else {}
            existing_hyperparams = existing_entry.get("hyperparams", {})
            existing_key = tuple(sorted(existing_hyperparams.items()))
            if existing_key == hyperparams_key:
                # Same hyperparams, update
                results[model_name][dataset_name] = {"entries": [result_entry]}
            else:
                # Different hyperparams, add as new entry
                results[model_name][dataset_name] = {"entries": [existing_entry, result_entry]}

    save_results(results, results_file)
    print(f"\nResults saved for {model_name}/{dataset_name}:")
    print(json.dumps(result_entry, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised learning training script with stable-datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset name from stable_datasets.images (default: CIFAR10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/resnet-50",
        help="Model name from HuggingFace (default: microsoft/resnet-18)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loader workers (default: 8)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.02,
        help="Weight decay (default: 0.02)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stable-datasets",
        help="W&B project name (default: stable-datasets)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="supervised_results.json",
        help="Path to JSON file for storing/loading results (default: supervised_results.json)",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force re-run training even if results already exist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config name for datasets with multiple configurations (e.g., 'balanced' for EMNIST, 'pathmnist' for MedMNIST). Required for EMNIST and MedMNIST.",
    )

    args = parser.parse_args()
    main(args)
