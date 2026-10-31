"""Integration test using the official full-resolution Imagewoof archive."""

import numpy as np
from PIL import Image

from stable_datasets.images import Imagewoof


def test_imagewoof_dataset() -> None:
    # Load the official training split (downloads/prepares data on a cache miss).
    imagewoof_train = Imagewoof(split="train")

    # Check the training count observed in the official archive.
    expected_num_train_samples = 9025
    assert len(imagewoof_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(imagewoof_train)}."
    )

    # Check sample fields, image type, and RGB representation.
    sample = imagewoof_train[0]
    assert set(sample) == {"image", "label"}
    image = sample["image"]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    image_np = np.array(image)
    # Full-resolution images have variable height and width.
    assert image_np.ndim == 3
    assert image_np.shape[2] == 3
    assert image_np.dtype == np.uint8

    # Check the classification label type and range.
    label = sample["label"]
    assert isinstance(label, int)
    assert 0 <= label < 10

    # The source archive's val directory is exposed as validation, not test.
    imagewoof_val = Imagewoof(split="validation")
    expected_num_val_samples = 3929
    assert len(imagewoof_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(imagewoof_val)}."
    )
