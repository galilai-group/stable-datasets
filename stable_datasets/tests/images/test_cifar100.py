import datasets
from PIL import Image
import numpy as np


def test_cifar100_dataset():
    # Load the CIFAR-100 dataset
    cifar100 = datasets.load_dataset("../../stable_datasets/images/cifar100.py", split="train", trust_remote_code=True)

    # Test 1: Check that the dataset has 50,000 training samples and 10,000 test samples
    assert len(cifar100) == 50000, f"Expected 50000 training samples, got {len(cifar100)}."

    # Test 2: Check that each sample has the keys "image", "label", and "superclass"
    sample = cifar100[0]
    expected_keys = {"image", "label", "superclass"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally: convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 99, f"Label should be between 0 and 99, got {label}."

    # Test 5: Validate superclass type and range
    superclass = sample["superclass"]
    assert isinstance(superclass, int), f"Superclass should be an integer, got {type(superclass)}."
    assert 0 <= superclass <= 19, f"Superclass should be between 0 and 19, got {superclass}."

    # Test 6: Check that test split has 10,000 samples
    cifar100_test = datasets.load_dataset("../../stable_datasets/images/cifar100.py", split="test", trust_remote_code=True)
    assert len(cifar100_test) == 10000, f"Expected 10000 test samples, got {len(cifar100_test)}."

    print("All CIFAR-100 tests passed successfully!")


if __name__ == "__main__":
    test_cifar100_dataset()
