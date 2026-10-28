import numpy as np
from PIL import Image

from stable_datasets.images import CIFAR100N


def test_cifar100n_dataset():
    # CIFAR100N(split="train") automatically downloads and loads the dataset
    cifar100n = CIFAR100N(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 50000
    assert len(cifar100n) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cifar100n)}."
    )

    # Test 2: Check the clean, coarse, and human annotation fields
    sample = cifar100n[0]
    expected_keys = {"image", "label", "coarse_label", "noisy_label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate fine label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 100, f"Label should be between 0 and 99, got {label}."

    # Test 5: Validate coarse label type and range
    coarse_label = sample["coarse_label"]
    assert isinstance(coarse_label, int), f"Coarse label should be an integer, got {type(coarse_label)}."
    assert 0 <= coarse_label < 20, f"Coarse label should be between 0 and 19, got {coarse_label}."

    # Test 6: Validate the human noisy label type and range
    noisy_label = sample["noisy_label"]
    assert isinstance(noisy_label, int), f"Noisy label should be an integer, got {type(noisy_label)}."
    assert 0 <= noisy_label < 100, f"Noisy label should be between 0 and 99, got {noisy_label}."

    # Test 7: Check the test split and missing human annotations
    cifar100n_test = CIFAR100N(split="test")
    expected_num_test_samples = 10000
    assert len(cifar100n_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cifar100n_test)}."
    )
    test_sample = cifar100n_test[0]
    assert set(test_sample.keys()) == expected_keys
    assert test_sample["noisy_label"] is None

    print("All CIFAR100N dataset tests passed successfully!")
