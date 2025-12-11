import datasets
from PIL import Image
import numpy as np


def test_linnaeus5_dataset():
    # Load the Linnaeus5 dataset
    linnaeus5_train = datasets.load_dataset("../../stable_datasets/images/linnaeus5.py", split="train", trust_remote_code=True)
    linnaeus5_test = datasets.load_dataset("../../stable_datasets/images/linnaeus5.py", split="test", trust_remote_code=True)

    # Test 1: Check number of samples in train and test splits
    assert len(linnaeus5_train) == 6000, f"Expected 6000 training samples, got {len(linnaeus5_train)}."
    assert len(linnaeus5_test) == 2000, f"Expected 2000 test samples, got {len(linnaeus5_test)}."

    # Test 2: Validate keys in a sample
    sample = linnaeus5_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (256, 256, 3), f"Image should have shape (256, 256, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 4, f"Label should be between 0 and 4, got {label}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_linnaeus5_dataset()
