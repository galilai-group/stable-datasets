"""
Supervised Learning Example with stable-datasets
=================================================

This example demonstrates how to train models using supervised learning
with stable-pretraining, using datasets from stable-datasets.
"""

import argparse
from functools import partial

import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoConfig, AutoModelForImageClassification
import stable_pretraining as spt

from stable_pretraining.data import transforms


def get_dataset_class(dataset_name: str):
    """Dynamically load dataset class from stable_datasets.images."""
    import importlib

    try:
        module = importlib.import_module("stable_datasets.images")
        dataset_class = getattr(module, dataset_name)
        return dataset_class
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in stable_datasets.images. "
            f"Error: {e}"
        )


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


class HFDatasetWrapper(spt.data.Dataset):
    """Wrapper for pre-loaded HuggingFace datasets with transform support."""

    def __init__(self, hf_dataset, transform=None):
        super().__init__(transform)
        self.hf_dataset = hf_dataset
        # Add sample_idx if not present
        if "sample_idx" not in hf_dataset.column_names:
            self.hf_dataset = hf_dataset.add_column(
                "sample_idx", list(range(hf_dataset.num_rows))
            )

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        return self.process_sample(sample)

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def column_names(self):
        return self.hf_dataset.column_names


def get_data_loaders(args, dataset_class):
    """Get train and validation data loaders for the specified dataset."""
    # Load the dataset
    train_dataset_raw = dataset_class(split="train")
    test_dataset_raw = dataset_class(split="test")

    # Infer number of classes from the dataset
    num_classes = get_num_classes(train_dataset_raw)

    # Get image size from config
    image_size = args.image_size

    # Use default normalization values (can be made dataset-specific later)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(3, 3), p=0.5),
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

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((image_size, image_size)),
        transforms.ToImage(mean=mean, std=std),
    )

    val_dataset = HFDatasetWrapper(
        hf_dataset=test_dataset_raw,
        transform=val_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, num_classes


def main(args):
    # Load dataset class
    dataset_class = get_dataset_class(args.dataset)

    # Get data loaders
    train_loader, val_loader, num_classes = get_data_loaders(args, dataset_class)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    # Define forward function
    def forward(self, batch, stage):
        batch["logits"] = self.backbone(batch["image"])["logits"]
        if self.training:
            loss = torch.nn.functional.cross_entropy(
                batch["logits"],
                batch["label"],
            )
            batch["loss"] = loss
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
    }

    module = spt.Module(
        backbone=backbone,
        forward=forward,
        hparams=hparams,
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
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    run_name = f"{model_name}-{args.dataset.lower()}"

    logger = WandbLogger(
        project=args.wandb_project if args.wandb_project else "stable-datasets-testing",
        name=run_name,
    )

    # Log hyperparameters to wandb
    logger.log_hyperparams(hparams)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        num_sanity_val_steps=1,
        callbacks=[lr_monitor],
        precision="16-mixed",
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=False,
    )

    # Train and validate
    manager = spt.Manager(trainer=trainer, module=module, data=data_module)
    manager()
    manager.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised learning training script with stable-datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset name from stable_datasets.images (default: CIFAR10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/resnet-18",
        help="Model name from HuggingFace (default: microsoft/resnet-18)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (default: 512)",
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
        default=32,
        help="Image size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs (default: 50)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (default: stable-datasets-testing)",
    )

    args = parser.parse_args()
    main(args)

