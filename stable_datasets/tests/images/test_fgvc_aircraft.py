import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import FGVCAircraft


pytestmark = pytest.mark.large


def test_fgvc_aircraft_dataset():
    # Load training split (variant as label)
    aircraft_train = FGVCAircraft(config_name="variant", split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 3334
    assert len(aircraft_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(aircraft_train)}."
    )

    # Test 2: Check sample keys
    sample = aircraft_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)

    # FGVC Aircraft images have variable height/width, so we only check dimensions and channels
    assert image_np.ndim == 3, f"Aircraft images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Aircraft images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 100, f"Label should be in range [0, 99] (100 variant classes), got {label}."

    # Test 5: Load and validate validation split
    aircraft_val = FGVCAircraft(config_name="variant", split="validation")
    expected_num_val_samples = 3333
    assert len(aircraft_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(aircraft_val)}."
    )

    # Test 6: Load and validate test split
    aircraft_test = FGVCAircraft(config_name="variant", split="test")
    expected_num_test_samples = 3333
    assert len(aircraft_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(aircraft_test)}."
    )

    print("All FGVC Aircraft dataset tests passed successfully!")
