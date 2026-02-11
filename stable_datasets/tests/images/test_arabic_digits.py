import numpy as np
from PIL import Image

from stable_datasets.images.arabic_digits import ArabicDigits


def test_arabic_digits_dataset():
    # ArabicDigits(split="train") automatically downloads and loads the dataset
    arabic_digits = ArabicDigits(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 60000
    assert len(arabic_digits) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(arabic_digits)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = arabic_digits[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check shape
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Check the test split
    arabic_digits_test = ArabicDigits(split="test")
    expected_num_test_samples = 10000
    assert len(arabic_digits_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(arabic_digits_test)}."
    )

    print("All ArabicDigits dataset tests passed successfully!")
