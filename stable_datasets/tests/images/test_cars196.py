import numpy as np
from PIL import Image

from stable_datasets.images.cars196 import Cars196


def test_cars196_dataset():
    # Load training split
    cars196_train = Cars196(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 8144
    assert len(cars196_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cars196_train)}."
    )

    # Test 2: Check sample keys
    sample = cars196_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Cars196 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Cars196 images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 196, f"Label should be in range [0, 195], got {label}."

    # Test 5: Load and validate test split
    cars196_test = Cars196(split="test")
    expected_num_test_samples = 8041
    assert len(cars196_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cars196_test)}."
    )

    print("All Cars196 dataset tests passed successfully!")
