import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import StanfordDogs


pytestmark = pytest.mark.large


def test_stanford_dogs_dataset():
    # Test 1: Train split count
    sd_train = StanfordDogs(split="train")
    expected_train = 12000
    assert len(sd_train) == expected_train, f"Expected {expected_train} training samples, got {len(sd_train)}."

    # Test 2: Sample keys
    sample = sd_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Image type/shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"StanfordDogs images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"StanfordDogs images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Label
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 120, f"Label should be in range [0, 119], got {label}."

    # Test 5: Test split count
    sd_test = StanfordDogs(split="test")
    expected_test = 8580
    assert len(sd_test) == expected_test, f"Expected {expected_test} test samples, got {len(sd_test)}."

    # Test 6: Class label features
    assert len(sd_train.features["label"].names) == 120, (
        f"Expected 120 breed names, got {len(sd_train.features['label'].names)}."
    )

    print("All StanfordDogs dataset tests passed successfully!")
