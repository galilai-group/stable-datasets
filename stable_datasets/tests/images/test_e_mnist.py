import datasets
from PIL import Image
import numpy as np


def test_e_mnist_dataset():
    # Choose a variant to test, for example "balanced"
    # According to the EMNIST documentation:
    # EMNIST Balanced: 47 balanced classes, 112,800 training samples, 18,800 test samples.
    variant = "balanced"

    # Load the EMNIST dataset (train split)
    emnist_train = datasets.load_dataset("../../stable_datasets/images/e_mnist.py", variant, split="train", trust_remote_code=True)

    # Test 1: Check that the dataset has the expected number of training samples
    # For EMNIST Balanced: expected training size ~112,800
    # (This is from the EMNIST documentation. Adjust if the dataset differs.)
    expected_train_size = 112800
    assert len(emnist_train) == expected_train_size, (
        f"Expected {expected_train_size} training samples for {variant}, got {len(emnist_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = emnist_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type (PIL.Image)
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Check image shape and dtype
    image_np = np.array(image)
    # EMNIST images should be 28x28 and grayscale
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    # For the balanced variant, we have 47 classes. Labels should be in [0, 46].
    assert 0 <= label < 47, f"Label should be between 0 and 46, got {label}."

    # Test 5: Check the test split size
    # EMNIST Balanced test set should have ~18,800 samples
    emnist_test = datasets.load_dataset("../../stable_datasets/images/e_mnist.py", variant, split="test")
    expected_test_size = 18800
    assert len(emnist_test) == expected_test_size, (
        f"Expected {expected_test_size} test samples for {variant}, got {len(emnist_test)}."
    )

    # Check a sample from the test set
    test_sample = emnist_test[0]
    assert "image" in test_sample and "label" in test_sample, "Test sample must have 'image' and 'label' keys."
    test_img = test_sample["image"]
    assert isinstance(test_img, Image.Image), f"Test image should be PIL, got {type(test_img)}."
    test_img_np = np.array(test_img)
    assert test_img_np.shape == (28, 28), f"Test image should have shape (28, 28), got {test_img_np.shape}."
    assert test_img_np.dtype == np.uint8, f"Test image should have dtype uint8, got {test_img_np.dtype}."

    test_label = test_sample["label"]
    assert isinstance(test_label, int), f"Test label should be integer, got {type(test_label)}."
    assert 0 <= test_label < 47, f"Test label should be between 0 and 46, got {test_label}."

    print("All tests for EMNIST dataset (balanced variant) passed successfully!")


if __name__ == "__main__":
    test_e_mnist_dataset()
