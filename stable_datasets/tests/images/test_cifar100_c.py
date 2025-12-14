import numpy as np
from PIL import Image

from stable_datasets.images.cifar100_c import CIFAR100C


def test_cifar100_c_dataset():
    # CIFAR100C(split="test") automatically downloads and loads the dataset
    cifar100_c = CIFAR100C(split="test")

    # Test 1: Check that the dataset has the expected number of samples
    # CIFAR-100-C has 19 corruptions * 5 levels * 10,000 images = 950,000 samples
    expected_num_test_samples = 19 * 5 * 10000
    assert len(cifar100_c) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(cifar100_c)}."
    )

    # Test 2: Check that each sample has the keys "image", "label", "corruption_name", and "corruption_level"
    sample = cifar100_c[0]
    expected_keys = {"image", "label", "corruption_name", "corruption_level"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.shape == (
        32,
        32,
        3,
    ), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 100, f"Label should be between 0 and 99, got {label}."

    # Test 5: Validate corruption_name type
    corruption_name = sample["corruption_name"]
    assert isinstance(corruption_name, str), f"Corruption name should be a string, got {type(corruption_name)}."
    expected_corruptions = CIFAR100C._corruptions()
    assert corruption_name in expected_corruptions, f"Corruption name {corruption_name} not in expected corruptions."

    # Test 6: Validate corruption_level type and range
    corruption_level = sample["corruption_level"]
    assert isinstance(corruption_level, int | np.integer), (
        f"Corruption level should be an integer, got {type(corruption_level)}."
    )
    assert 1 <= corruption_level <= 5, f"Corruption level should be between 1 and 5, got {corruption_level}."

    print("All CIFAR-100-C dataset tests passed successfully!")
