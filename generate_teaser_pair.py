#!/usr/bin/env python
"""
Generate teaser figures for pair datasets, where each example has two images (e.g. LFW: image1, image2, label).

Each example is drawn as its two images side by side, with the class name above. Examples are balanced across classes
(for LFW: alternating "different" and "same" pairs).

Usage:
    python generate_pair_teaser.py --name LFW --num-samples 4 --output docs/source/datasets/teasers/lfw_teaser.png
"""

import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np


def to_display_array(image):
    """Convert a PIL image or array to an HxW or HxWxC float array in [0, 1]."""
    if hasattr(image, "convert"):  # PIL
        image = np.array(image)
    if image.dtype == np.uint8 or image.max() > 1.0:
        image = image / 255.0
    return image


def load_dataset_class(dataset_name):
    for module_name in ("stable_datasets.images", "stable_datasets.timeseries"):
        module = importlib.import_module(module_name)
        if hasattr(module, dataset_name):
            return getattr(module, dataset_name)
    raise ValueError(f"Dataset '{dataset_name}' not found in stable_datasets.images or stable_datasets.timeseries")


def pick_balanced_indices(labels, num_samples, seed):
    """Pick num_samples indices, cycling through classes so the figure shows each class equally often."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    pools = {c: list(rng.permutation(np.flatnonzero(labels == c))) for c in classes}
    chosen = []
    while len(chosen) < num_samples and any(pools.values()):
        for c in classes:
            if pools[c] and len(chosen) < num_samples:
                chosen.append(int(pools[c].pop()))
    return chosen


def generate_pair_teaser(
    dataset_name: str,
    num_samples: int = 4,
    image_keys: tuple = ("image1", "image2"),
    label_key: str = "label",
    output_path: str | None = None,
    figsize_per_sample: float = 3.0,
    gap: int = 6,
    seed: int = 0,
    split: str = "train",
):
    dataset = load_dataset_class(dataset_name)(split=split)
    print(f"Loaded {dataset_name} ({split}): {len(dataset)} examples")

    # Reading a single column does not decode the images, so this is fast even for large splits.
    indices = pick_balanced_indices(dataset[label_key], num_samples, seed)

    class_names = None
    label_feature = dataset.features.get(label_key) if hasattr(dataset, "features") else None
    if hasattr(label_feature, "names"):
        class_names = label_feature.names

    fig = plt.figure(figsize=(figsize_per_sample * len(indices), figsize_per_sample * 0.55))
    gs = fig.add_gridspec(
        2,
        len(indices),
        height_ratios=[0.08, 1],
        hspace=0,
        wspace=0.05,
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.01,
    )

    for col, idx in enumerate(indices):
        sample = dataset[idx]

        label = sample[label_key]
        label_text = class_names[label] if class_names is not None and isinstance(label, int) else str(label)

        ax_label = fig.add_subplot(gs[0, col])
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, label_text, ha="center", va="center", fontsize=9, fontweight="bold")

        arrays = [to_display_array(sample[key]) for key in image_keys]
        spacer = np.ones((arrays[0].shape[0], gap) + arrays[0].shape[2:])
        parts = []
        for array in arrays:
            if parts:
                parts.append(spacer)
            parts.append(array)
        pair = np.concatenate(parts, axis=1)

        ax_image = fig.add_subplot(gs[1, col])
        ax_image.axis("off")
        ax_image.margins(0)
        if pair.ndim == 2:
            ax_image.imshow(pair, cmap="gray", vmin=0, vmax=1)
        else:
            ax_image.imshow(pair)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved teaser figure to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate teaser figures for pair datasets (two images per example)")
    parser.add_argument("--name", type=str, required=True, help="Dataset class name (e.g., LFW)")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of pairs to display (default: 4)")
    parser.add_argument(
        "--image-keys",
        type=str,
        nargs="+",
        default=["image1", "image2"],
        help="Keys of the image columns, drawn left to right (default: image1 image2)",
    )
    parser.add_argument("--label-key", type=str, default="label", help="Key for the label (default: 'label')")
    parser.add_argument("--split", type=str, default="train", help="Split to sample from (default: train)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing which pairs to show (default: 0)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path (displays the figure if omitted)")
    parser.add_argument("--figsize", type=float, default=3.0, help="Width per pair in inches (default: 3.0)")
    args = parser.parse_args()

    generate_pair_teaser(
        dataset_name=args.name,
        num_samples=args.num_samples,
        image_keys=tuple(args.image_keys),
        label_key=args.label_key,
        output_path=args.output,
        figsize_per_sample=args.figsize,
        seed=args.seed,
        split=args.split,
    )


if __name__ == "__main__":
    main()