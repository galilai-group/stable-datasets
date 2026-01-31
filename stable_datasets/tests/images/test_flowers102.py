import numpy as np
from PIL import Image

from stable_datasets.images.flowers102 import Flowers102


def test_flowers102_dataset():
    # Load training split
    flowers_train = Flowers102(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 1020
    assert len(flowers_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(flowers_train)}."
    )

    # Test 2: Check sample keys
    sample = flowers_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)

    # Flowers102 images have variable height/width, so we only check dimensions and channels
    assert image_np.ndim == 3, f"Flowers102 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Flowers102 images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 102, f"Label should be in range [0, 101], got {label}."

    # Test 5: Load and validate validation split
    flowers_val = Flowers102(split="validation")
    expected_num_val_samples = 1020
    assert len(flowers_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(flowers_val)}."
    )

    # Test 6: Load and validate test split
    flowers_test = Flowers102(split="test")
    expected_num_test_samples = 6149
    assert len(flowers_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(flowers_test)}."
    )

    print("All Flowers102 dataset tests passed successfully!")
