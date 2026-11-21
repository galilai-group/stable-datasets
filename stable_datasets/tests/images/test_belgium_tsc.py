import numpy as np
from PIL import Image

from stable_datasets.images import BelgiumTSC


def test_belgium_tsc_dataset():
    # Load training split
    btsc_train = BelgiumTSC(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 4575
    assert len(btsc_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(btsc_train)}."
    )

    # Test 2: Check sample keys
    sample = btsc_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and shape (image sizes vary)
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"BelgiumTSC images should be RGB, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    num_classes = 62
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < num_classes, f"Label should be in [0, {num_classes}), got {label}."
    assert len(btsc_train.info.features["label"].names) == num_classes

    # Test 5: Load and validate test split
    btsc_test = BelgiumTSC(split="test")
    expected_num_test_samples = 2520
    assert len(btsc_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(btsc_test)}."
    )

    print("All BelgiumTSC dataset tests passed successfully!")
