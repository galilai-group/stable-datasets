import numpy as np
import datasets
from PIL import Image
from stable_datasets.images.cifar100_c import CIFAR100C


def test_cifar100_c_dataset():
    # Load the CIFAR-100-C dataset
    cifar100c = datasets.load_dataset("../../stable_datasets/images/cifar100_c.py", split="test", trust_remote_code=True)

    # Test 1: Check that the dataset has the correct number of samples
    # CIFAR-100-C typically contains 10,000 images per corruption type, with 5 levels each.
    # For 19 corruptions, this would be 19 * 5 * 10000 = 950000 images.
    expected_num_samples = 950000
    assert len(cifar100c) == expected_num_samples, f"Expected {expected_num_samples} samples, got {len(cifar100c)}."

    # Test 2: Check that each sample has the keys "image", "label", "corruption_name", and "corruption_level"
    sample = cifar100c[0]
    expected_keys = {"image", "label", "corruption_name", "corruption_level"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy array to check shape
    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 99, f"Label should be between 0 and 99, got {label}."

    # Test 5: Validate corruption_name and corruption_level
    corruption_name = sample["corruption_name"]
    corruption_level = sample["corruption_level"]
    corruptions = CIFAR100C._corruptions()
    assert corruption_name in corruptions, f"Unexpected corruption_name: {corruption_name}"
    assert 1 <= corruption_level <= 5, f"corruption_level should be between 1 and 5, got {corruption_level}"

    print("All CIFAR-100-C tests passed successfully!")


if __name__ == "__main__":
    test_cifar100_c_dataset()
