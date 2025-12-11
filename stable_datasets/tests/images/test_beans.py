import datasets
from PIL import Image
import numpy as np


def test_beans_dataset():
    # Load the IBeans dataset
    ibeans_train = datasets.load_dataset("../../stable_datasets/images/beans.py", split="train", trust_remote_code=True)
    ibeans_test = datasets.load_dataset("../../stable_datasets/images/beans.py", split="test", trust_remote_code=True)
    ibeans_valid = datasets.load_dataset("../../stable_datasets/images/beans.py", split="validation", trust_remote_code=True)

    # Test 1: Check dataset split sizes
    assert len(ibeans_train) > 0, "Training dataset is empty."
    assert len(ibeans_test) > 0, "Test dataset is empty."
    assert len(ibeans_valid) > 0, "Validation dataset is empty."

    # Test 2: Check that each sample has "image" and "label"
    sample = ibeans_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and dimensions
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Image should have 3 dimensions, got {image_np.ndim}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 3, f"Label should be between 0 and 2, got {label}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_beans_dataset()
