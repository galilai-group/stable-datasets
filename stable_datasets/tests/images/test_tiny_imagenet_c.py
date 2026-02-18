import numpy as np
from PIL import Image

from stable_datasets.images import TinyImagenetC


def test_tiny_imagenet_c_dataset():
    # Load test split
    tiny_c_test = TinyImagenetC(split="test")

    # Test 1: Expected number of corrupted samples
    # ImageNet-C style corruptions typically include 15 corruption types * 5 severity levels
    expected_num_test_samples = 15 * 5 * 10000  # 15 corruptions * 5 levels * 10k validation images
    assert len(tiny_c_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(tiny_c_test)}."
    )

    # Test 2: Check that each sample has the required keys
    sample = tiny_c_test[0]
    expected_keys = {"image", "label", "corruption_name", "corruption_level"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (
        64,
        64,
        3,
    ), f"Image should have shape (64, 64, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, str) or isinstance(label, int), (
        f"Label should be a string id or integer, got {type(label)}."
    )

    # Test 5: Validate corruption_name and corruption_level
    corruption_name = sample["corruption_name"]
    assert isinstance(corruption_name, str), f"Corruption name should be a string, got {type(corruption_name)}."

    corruption_level = sample["corruption_level"]
    assert isinstance(corruption_level, int | np.integer), (
        f"Corruption level should be an integer, got {type(corruption_level)}."
    )
    assert 1 <= int(corruption_level) <= 5, f"Corruption level should be between 1 and 5, got {corruption_level}."

    # Now load all splits (split=None) to mirror not_mnist style checking of full builder behavior
    all_splits = TinyImagenetC(split=None)
    # Ensure the returned object includes the test split
    assert "test" in all_splits, "Expected 'test' split to be present when loading all splits."
    assert len(all_splits["test"]) == expected_num_test_samples, "Mismatch in test split size from all_splits view."

    print("All Tiny ImageNet-C dataset tests passed successfully!")


if __name__ == "__main__":
    test_tiny_imagenet_c_dataset()
