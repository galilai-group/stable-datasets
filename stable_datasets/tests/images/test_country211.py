import numpy as np
from PIL import Image

from stable_datasets.images.country211 import Country211


def test_country211_dataset():
    # Load training split
    country211_train = Country211(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 31650
    assert len(country211_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(country211_train)}."
    )

    # Test 2: Check sample keys
    sample = country211_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Country211 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Country211 images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 212, f"Label should be in range [0, 211], got {label}."

    # Test 5: Load and validate test split
    country211_test = Country211(split="test")
    expected_num_test_samples = 21100
    assert len(country211_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(country211_test)}."
    )

    # Test 6: Load and validate validation split
    country211_val = Country211(split="valid")
    expected_num_val_samples = 10550
    assert len(country211_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(country211_val)}."
    )

    print("All Country211 dataset tests passed successfully!")
