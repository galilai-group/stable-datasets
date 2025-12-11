import datasets
from PIL import Image
import numpy as np


def test_mnist_dataset():
    # Load the MNIST dataset
    mnist_train = datasets.load_dataset("../../stable_datasets/images/mnist.py", split="train", trust_remote_code=True)

    # Test 1: Check training sample count
    assert len(mnist_train) == 60000, f"Expected 60000 training samples, got {len(mnist_train)}."

    # Test 2: Validate keys in a sample
    sample = mnist_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image format and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image shape should be (28, 28), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Validate test set sample count
    mnist_test = datasets.load_dataset("../../stable_datasets/images/mnist.py", split="test", trust_remote_code=True)
    assert len(mnist_test) == 10000, f"Expected 10000 test samples, got {len(mnist_test)}."

    print("All MNIST tests passed successfully!")


if __name__ == "__main__":
    test_mnist_dataset()
