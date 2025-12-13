import numpy as np
from PIL import Image

from stable_datasets.images.cifar100 import CIFAR100


def test_cifar100_dataset():
    # CIFAR100(split="train") automatically downloads and loads the dataset
    cifar100 = CIFAR100(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 50000
    assert len(cifar100) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cifar100)}."
    )

    # Test 2: Check that each sample has the keys "image", "label", and "superclass"
    sample = cifar100[0]
    expected_keys = {"image", "label", "superclass"}
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
    assert 0 <= label < 100, f"Label should be between 0 and 99, got {label}."

    # Test 5: Validate superclass type and range
    superclass = sample["superclass"]
    assert isinstance(superclass, int), f"Superclass should be an integer, got {type(superclass)}."
    assert 0 <= superclass < 20, f"Superclass should be between 0 and 19, got {superclass}."

    # Test 6: Check the test split
    cifar100_test = CIFAR100(split="test")
    expected_num_test_samples = 10000
    assert len(cifar100_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cifar100_test)}."
    )

    print("All CIFAR100 dataset tests passed successfully!")
