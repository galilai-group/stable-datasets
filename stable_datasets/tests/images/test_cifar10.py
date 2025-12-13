import numpy as np
from PIL import Image

from stable_datasets.images.cifar10 import CIFAR10


def test_cifar10_dataset():
    # CIFAR10(split="train") automatically downloads and loads the dataset
    cifar10 = CIFAR10(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 50000
    assert len(cifar10) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cifar10)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = cifar10[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.shape == (
        32,
        32,
        3,
    ), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Check the test split
    cifar10_test = CIFAR10(split="test")
    expected_num_test_samples = 10000
    assert len(cifar10_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cifar10_test)}."
    )

    print("All CIFAR10 dataset tests passed successfully!")
