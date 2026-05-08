import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import OxfordPet


pytestmark = pytest.mark.large


def test_oxford_pet_dataset():
    # Load training split
    pet_train = OxfordPet(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 3680
    assert len(pet_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(pet_train)}."
    )

    # Test 2: Check sample keys
    sample = pet_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"OxfordPet images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"OxfordPet images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 37, f"Label should be in range [0, 36], got {label}."

    # Test 5: Load and validate test split
    pet_test = OxfordPet(split="test")
    expected_num_test_samples = 3669
    assert len(pet_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(pet_test)}."
    )

    print("All OxfordPet dataset tests passed successfully!")
