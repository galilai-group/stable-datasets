import numpy as np
from PIL import Image

from stable_datasets.images.cub200 import CUB200


def test_cub200_dataset():
    # Load training split
    cub200_train = CUB200(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 5994
    assert len(cub200_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cub200_train)}."
    )

    # Test 2: Check sample keys
    sample = cub200_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"CUB200 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"CUB200 images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 200, f"Label should be in range [0, 199], got {label}."

    # Test 5: Load and validate test split
    cub200_test = CUB200(split="test")
    expected_num_test_samples = 5794
    assert len(cub200_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cub200_test)}."
    )

    print("All CUB200 dataset tests passed successfully!")
