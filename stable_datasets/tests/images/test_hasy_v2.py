import numpy as np
from PIL import Image

from stable_datasets.images.hasy_v2 import HASYv2


def test_hasy_v2_fold1_integrity():
    """
    Tests the 'fold-1' configuration (Default benchmark split).
    """
    # Load both splits
    ds_train = HASYv2(config_name="fold-1", split="train")
    ds_test = HASYv2(config_name="fold-1", split="test")

    # Test 1: Check Total Dataset Size
    total_samples = len(ds_train) + len(ds_test)
    expected_total = 168233
    assert total_samples == expected_total, (
        f"Expected {expected_total} total samples (train+test), got {total_samples}."
    )

    # Test 2: Check keys
    sample = ds_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    # There are 369 classes
    assert 0 <= label < 369, f"Label should be between 0 and 368, got {label}."

    print("HASYv2 'fold-1' integrity tests passed successfully!")


def test_hasy_v2_default_config():
    """
    Ensures that HASYv2 defaults to 'fold-1' if no config is specified.
    """
    ds = HASYv2(split="train")

    assert ds.info.config_name == "fold-1", f"Expected default config to be 'fold-1', got '{ds.info.config_name}'"
    assert len(ds) > 0, "Default dataset should not be empty."

    print("HASYv2 default config test passed!")


def test_hasy_v2_fold_switching():
    """
    Ensures that the user can load a different fold (e.g., fold-5)
    and that it contains different data than fold-1.
    """
    ds_fold1 = HASYv2(config_name="fold-1", split="test")

    ds_fold5 = HASYv2(config_name="fold-5", split="test")

    assert ds_fold5.info.config_name == "fold-5"
    assert len(ds_fold5) > 10000

    sample1 = ds_fold1[0]
    sample5 = ds_fold5[0]

    if len(ds_fold1) == len(ds_fold5):
        assert (
            sample1["label"] != sample5["label"]
            or np.array(sample1["image"]).tobytes() != np.array(sample5["image"]).tobytes()
        ), "Fold 1 and Fold 5 appear to be identical! Config switching might be broken."

    print("HASYv2 fold switching test passed!")
