import datasets
from PIL import Image
import numpy as np


def test_cifar10_dataset():
    # Load the CIFAR-10 dataset
    cifar10 = datasets.load_dataset("../../stable_datasets/images/cifar10.py", split="train", trust_remote_code=True)

    # Test 1: Check that the dataset has 50,000 training samples and 10,000 test samples
    assert len(cifar10) == 50000, f"Expected 50000 training samples, got {len(cifar10)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = cifar10[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type (PIL.Image)
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally: convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 9, f"Label should be between 0 and 9, got {label}."

    # Test 5: Check that test split has 10,000 samples
    cifar10_test = datasets.load_dataset("../../stable_datasets/images/cifar10.py", split="test", trust_remote_code=True)
    assert len(cifar10_test) == 10000, f"Expected 10000 test samples, got {len(cifar10_test)}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_cifar10_dataset()
