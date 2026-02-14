import numpy as np
from PIL import Image

from stable_datasets.images.linnaeus5 import Linnaeus5


def test_linnaeus5_dataset():
    linnaeus_train = Linnaeus5(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 6000
    assert len(linnaeus_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(linnaeus_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = linnaeus_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (256, 256, 3), f"Image should have shape (256, 256, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 5, f"Label should be between 0 and 4, got {label}."

    # Test 5: Check the test split
    linnaeus_test = Linnaeus5(split="test")
    expected_num_test_samples = 2000
    assert len(linnaeus_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(linnaeus_test)}."
    )

    print("All Linnaeus 5 dataset tests passed successfully!")
