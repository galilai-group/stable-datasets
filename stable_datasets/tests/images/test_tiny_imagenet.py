import datasets
from PIL import Image
import numpy as np


def test_tiny_imagenet_dataset():
    # Load the Tiny ImageNet dataset
    dataset = datasets.load_dataset("../../stable_datasets/images/tiny_imagenet.py", split="train", trust_remote_code=True)

    # Test 1: Check the number of samples in the train split
    assert len(dataset) > 0, "The training dataset should not be empty."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = dataset[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (64, 64, 3), f"Image should have shape (64, 64, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 200, f"Label should be between 0 and 199, got {label}."

    # Test 5: Check the validation split
    val_dataset = datasets.load_dataset("../../stable_datasets/images/tiny_imagenet.py", split="validation", trust_remote_code=True)
    assert len(val_dataset) > 0, "The validation dataset should not be empty."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_tiny_imagenet_dataset()
