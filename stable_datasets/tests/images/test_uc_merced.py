import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.uc_merced import UCMerced


def test_uc_merced_dataset():
    ds_train = UCMerced(split="train")

    # Test 1: Check number of samples
    expected_num_samples = 2100
    assert len(ds_train) == expected_num_samples, f"Expected {expected_num_samples} samples, got {len(ds_train)}."

    # Test 2: Check sample keys
    sample = ds_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy array to check shape
    image_np = np.array(image)
    assert image_np.shape == (256, 256, 3), f"Image should have shape (256, 256, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 21, f"Label should be between 0 and 20, got {label}."

    # Test 5: Ensure accessing a non-existent split raises an error
    # Since UCMerced doesn't have a 'test' split defined in splits, this should fail
    with pytest.raises(ValueError):
        UCMerced(split="test")

    print("All UC Merced dataset tests passed successfully!")
