"""Shared schema constants and validation for CIFAR-10-compatible datasets."""

from __future__ import annotations

import numpy as np


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def validate_cifar10_arrays(images: np.ndarray, labels: np.ndarray, *, dataset_name: str) -> None:
    """Validate the image and label contract shared by CIFAR-10 variants."""
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(
            f"{dataset_name} images must have shape (N, 32, 32, 3), got {images.shape}."
        )
    if images.dtype != np.uint8:
        raise ValueError(f"{dataset_name} images must use uint8 pixels, got {images.dtype}.")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError(
            f"{dataset_name} labels must be a one-dimensional array matching the image count; "
            f"got {labels.shape} for {len(images)} images."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{dataset_name} labels must be integers, got {labels.dtype}.")
    if np.any((labels < 0) | (labels >= len(CIFAR10_LABELS))):
        raise ValueError(f"{dataset_name} labels must be in [0, {len(CIFAR10_LABELS) - 1}].")
