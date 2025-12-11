import datasets
from PIL import Image
import numpy as np


def test_arabic_characters_dataset():
    # Load the ArabicCharacters dataset
    arabic_characters = datasets.load_dataset("../../stable_datasets/images/arabic_characters.py", split="train", trust_remote_code=True)

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 13440
    assert len(
        arabic_characters) == expected_num_train_samples, f"Expected {expected_num_train_samples} training samples, got {len(arabic_characters)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = arabic_characters[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.shape in [(32, 32)], f"Image should have shape (32, 32), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 28, f"Label should be between 0 and 27, got {label}."

    # Test 5: Check the test split
    arabic_characters_test = datasets.load_dataset("../../stable_datasets/images/arabic_characters.py", split="test", trust_remote_code=True)
    expected_num_test_samples = 3360
    assert len(
        arabic_characters_test) == expected_num_test_samples, f"Expected {expected_num_test_samples} test samples, got {len(arabic_characters_test)}."

    print("All ArabicCharacters dataset tests passed successfully!")


if __name__ == "__main__":
    test_arabic_characters_dataset()
