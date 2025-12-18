#!/usr/bin/env python
"""
Generate teaser figures for datasets.

Usage:
    python generate_teaser.py --name MNIST --num-samples 5
    python generate_teaser.py --name CIFAR10 --num-samples 8 --output cifar10_teaser.png
    python generate_teaser.py --name MedMNIST --variant dermamnist --num-samples 5 --output medmnist_dermamnist_teaser.png
"""

import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np


def generate_teaser(
    dataset_name: str,
    num_samples: int = 5,
    image_key: str = "image",
    label_key: str = "label",
    output_path: str | None = None,
    figsize_per_sample: float = 1.5,
    variant: str | None = None,
):
    """
    Generate a teaser figure for a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'CIFAR10', 'MNIST')
        num_samples: Number of samples to display
        image_key: Key for image data in the dataset
        label_key: Key for label data in the dataset
        output_path: Path to save the figure (if None, display instead)
        figsize_per_sample: Width per sample in inches
        variant: Optional dataset variant/config name (e.g. MedMNIST variants)
    """
    # Try to import the dataset
    try:
        # Try loading from stable_datasets.images
        module = importlib.import_module("stable_datasets.images")
        dataset_class = getattr(module, dataset_name)
    except (ImportError, AttributeError):
        try:
            # Try loading from stable_datasets.timeseries
            module = importlib.import_module("stable_datasets.timeseries")
            dataset_class = getattr(module, dataset_name)
        except (ImportError, AttributeError):
            raise ValueError(
                f"Dataset '{dataset_name}' not found in stable_datasets.images or stable_datasets.timeseries"
            )

    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    if variant is None:
        dataset = dataset_class(split="train")
    else:
        dataset = dataset_class(split="train", config_name=variant)

    # Get samples from different classes
    samples = []
    seen_classes = set()
    idx = 0

    while len(samples) < num_samples and idx < len(dataset):
        sample = dataset[idx]
        label = sample.get(label_key)

        # Only add if we haven't seen this class yet
        if label not in seen_classes:
            samples.append(sample)
            seen_classes.add(label)

        idx += 1

    # If we couldn't get enough unique classes, fill with remaining samples
    if len(samples) < num_samples:
        print(f"Warning: Only found {len(samples)} unique classes, but {num_samples} samples requested.")
        idx = 0
        while len(samples) < num_samples and idx < len(dataset):
            samples.append(dataset[idx])
            idx += 1

    # Create figure
    fig = plt.figure(figsize=(figsize_per_sample * num_samples, figsize_per_sample * 1.1))
    gs = fig.add_gridspec(
        2,
        num_samples,
        height_ratios=[0.05, 1],
        hspace=0,
        wspace=0.05,
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.01,
    )

    # Get class names if available
    class_names = None
    if hasattr(dataset, "info") and dataset.info.features:
        label_feature = dataset.info.features.get(label_key)
        if hasattr(label_feature, "names"):
            class_names = label_feature.names

    for idx, sample in enumerate(samples):
        # Top: Label
        ax_label = fig.add_subplot(gs[0, idx])
        ax_label.axis("off")
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)

        label = sample.get(label_key, "N/A")

        # Convert label to class name if available
        if class_names is not None and isinstance(label, int):
            label_text = class_names[label]
        else:
            label_text = str(label)

        ax_label.text(
            0.5,
            0.5,
            label_text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax_label.transAxes,
        )

        # Bottom: Image
        ax_image = fig.add_subplot(gs[1, idx])
        ax_image.axis("off")
        ax_image.margins(0)

        image = sample[image_key]

        # Convert PIL Image to numpy if needed
        if hasattr(image, "numpy"):
            image = image.numpy()
        elif hasattr(image, "convert"):  # PIL Image
            image = np.array(image)

        # Handle different image formats
        if len(image.shape) == 3:
            # RGB or CHW format
            if image.shape[0] in [1, 3]:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 1:  # Grayscale
                image = image.squeeze(-1)

        # Normalize if needed
        if image.max() <= 1.0:
            image = image
        else:
            image = image / 255.0

        # Display image
        if len(image.shape) == 2:  # Grayscale
            ax_image.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:  # RGB
            ax_image.imshow(image)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved teaser figure to {output_path}")
    else:
        plt.tight_layout(pad=0.1)
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate teaser figures for stable-datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_teaser.py --name MNIST --num-samples 5
  python generate_teaser.py --name CIFAR10 --num-samples 8 --output cifar10.png
  python generate_teaser.py --name ArabicCharacters --num-samples 10
  python generate_teaser.py --name MedMNIST --variant dermamnist --num-samples 5
        """,
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., MNIST, CIFAR10, CIFAR100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help='Optional dataset variant/config name (e.g., "dermamnist" for MedMNIST).',
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="image",
        help="Key for image data in the dataset (default: 'image')",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="label",
        help="Key for label/title data in the dataset (default: 'label')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the figure (if not specified, will display instead)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        default=1.5,
        help="Width per sample in inches (default: 1.5)",
    )

    args = parser.parse_args()

    generate_teaser(
        dataset_name=args.name,
        num_samples=args.num_samples,
        image_key=args.image_key,
        label_key=args.label_key,
        output_path=args.output,
        figsize_per_sample=args.figsize,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
