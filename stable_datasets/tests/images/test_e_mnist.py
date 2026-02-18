import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import EMNIST


def test_emnist_digits_dataset():
    """
    Tests the 'digits' configuration of EMNIST.
    This variant mimics MNIST (0-9) but has significantly more samples.
    """
    ds_train = EMNIST(config_name="digits", split="train")

    # Test 1: Check expected number of samples
    expected_num_train_samples = 240000
    assert len(ds_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(ds_train)}."
    )

    # Test 2: Check keys
    sample = ds_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range for Digits (0-9)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label for 'digits' should be between 0 and 9, got {label}."

    # Test 5: Check the test split size
    ds_test = EMNIST(config_name="digits", split="test")
    expected_num_test_samples = 40000
    assert len(ds_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(ds_test)}."
    )

    print("EMNIST 'digits' tests passed successfully!")


def test_emnist_letters_dataset():
    """
    Tests the 'letters' configuration of EMNIST.
    """
    ds_train = EMNIST(config_name="letters", split="train")

    # Test 1: Check expected number of samples
    expected_num_train_samples = 124800
    assert len(ds_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(ds_train)}."
    )

    # Test 2: Validate Label Range Standardization
    labels = [ds_train[i]["label"] for i in range(100)]

    assert max(labels) <= 25, f"Max label should be 25, found {max(labels)}. Indexing fix might be missing."
    assert min(labels) >= 0, f"Min label should be 0, found {min(labels)}."

    # Test 3: Check Test split size
    ds_test = EMNIST(config_name="letters", split="test")
    expected_num_test_samples = 20800
    assert len(ds_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(ds_test)}."
    )

    print("EMNIST 'letters' tests passed successfully!")


def test_emnist_config_enforcement():
    """
    Ensures that the user cannot initialize EMNIST without specifying a config.
    """
    with pytest.raises(ValueError) as excinfo:
        # This should fail because EMNIST has multiple configs and no default is forced in base class
        _ = EMNIST(split="train")

    assert "Config name is missing" in str(excinfo.value)
    print("EMNIST config enforcement test passed!")
