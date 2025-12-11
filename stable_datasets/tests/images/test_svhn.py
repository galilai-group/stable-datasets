import datasets
from PIL import Image
import numpy as np


def test_svhn_dataset():
    # Load the SVHN dataset
    svhn_train = datasets.load_dataset("../../stable_datasets/images/svhn.py", split="train", trust_remote_code=True)

    # Test 1: Check that the training dataset is non-empty
    assert len(svhn_train) > 0, "The training dataset should not be empty."

    # Test 2: Check that each sample has "image" and "label" keys
    sample = svhn_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9, got {label}."

    # Test 5: Verify the test dataset is non-empty
    svhn_test = datasets.load_dataset("../../stable_datasets/images/svhn.py", split="test", trust_remote_code=True)
    assert len(svhn_test) > 0, "The test dataset should not be empty."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_svhn_dataset()
