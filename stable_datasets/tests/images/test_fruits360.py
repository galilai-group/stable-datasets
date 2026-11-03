import numpy as np
from PIL import Image

from stable_datasets.images import Fruits360


def test_fruits360_dataset():
    # Load training split
    fruits_train = Fruits360(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 147416
    assert len(fruits_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(fruits_train)}."
    )

    # Test 2: Check sample keys
    sample = fruits_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (100, 100, 3), f"Fruits-360 images should be 100x100x3, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    num_classes = 274
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < num_classes, f"Label should be in [0, {num_classes}), got {label}."

    # Test 5: Validate label names (case-variant upstream folders are merged)
    names = fruits_train.info.features["label"].names
    assert len(names) == num_classes, f"Expected {num_classes} classes, got {len(names)}."
    assert len({name.lower() for name in names}) == num_classes, "Class names should be unique ignoring case."

    # Test 6: Load and validate test split
    fruits_test = Fruits360(split="test")
    expected_num_test_samples = 49106
    assert len(fruits_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(fruits_test)}."
    )

    print("All Fruits-360 dataset tests passed successfully!")
