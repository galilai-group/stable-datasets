import numpy as np
from PIL import Image

from stable_datasets.images import CIFAR10N


def test_cifar10n_dataset():
    # CIFAR10N(split="train") automatically downloads and loads the dataset
    cifar10n = CIFAR10N(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 50000
    assert len(cifar10n) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cifar10n)}."
    )

    # Test 2: Check the clean and human annotation fields
    sample = cifar10n[0]
    annotation_fields = {"aggre_label", "worse_label", "random_label1", "random_label2", "random_label3"}
    expected_keys = {"image", "label"} | annotation_fields
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate clean label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Validate each human annotation type and range
    for field in annotation_fields:
        annotation = sample[field]
        assert isinstance(annotation, int), f"{field} should be an integer, got {type(annotation)}."
        assert 0 <= annotation < 10, f"{field} should be between 0 and 9, got {annotation}."

    # Test 6: Check the test split and missing human annotations
    cifar10n_test = CIFAR10N(split="test")
    expected_num_test_samples = 10000
    assert len(cifar10n_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cifar10n_test)}."
    )
    test_sample = cifar10n_test[0]
    assert set(test_sample.keys()) == expected_keys
    assert all(test_sample[field] is None for field in annotation_fields)

    print("All CIFAR10N dataset tests passed successfully!")
