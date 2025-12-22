import numpy as np
from PIL import Image

from stable_datasets.images.not_mnist import NotMNIST


def test_not_mnist_dataset():
    # NotMNIST(split="train") automatically downloads and loads the dataset
    not_mnist_train = NotMNIST(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 60000
    assert len(not_mnist_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(not_mnist_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = not_mnist_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Test 4: Validate image shape (convert to numpy array)
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"

    # Test 5: Validate image dtype
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 6: Validate label type
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."

    # Test 7: Validate label range (0-9 for letters A-J)
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 8: Check the test split
    not_mnist_test = NotMNIST(split="test")
    expected_num_test_samples = 10000
    assert len(not_mnist_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(not_mnist_test)}."
    )

    print("All NotMNIST dataset tests passed successfully!")


if __name__ == "__main__":
    test_not_mnist_dataset()
