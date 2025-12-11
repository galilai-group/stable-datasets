import datasets
from PIL import Image
import numpy as np


def test_flowers102_dataset():
    # Load the Flowers102 dataset
    flowers102 = datasets.load_dataset("../../stable_datasets/images/flowers102.py", split="train", trust_remote_code=True)

    # Test 1: Check the number of samples in each split
    assert len(flowers102) > 0, "The training set should not be empty."

    # Test 2: Check that each sample has "image" and "label"
    sample = flowers102[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type (PIL.Image)
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Image should have 3 dimensions, got {image_np.ndim}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 102, f"Label should be in range [0, 101], got {label}."

    # Test additional splits (e.g., validation and test)
    flowers102_val = datasets.load_dataset("../../stable_datasets/images/flowers102.py", split="validation", trust_remote_code=True)
    assert len(flowers102_val) > 0, "The validation set should not be empty."

    flowers102_test = datasets.load_dataset("../../stable_datasets/images/flowers102.py", split="test", trust_remote_code=True)
    assert len(flowers102_test) > 0, "The test set should not be empty."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_flowers102_dataset()
