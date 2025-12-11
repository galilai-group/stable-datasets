import datasets
from pathlib import Path
from PIL import Image
import numpy as np


def test_imagenet_dataset():
    # Test for default config (imagenette)
    imagenette = datasets.load_dataset("../../stable_datasets/images/imagenette.py", split="train", trust_remote_code=True)

    # Test 1: Check dataset length
    assert len(imagenette) > 0, "Expected non-zero samples in train split."

    # Test 2: Validate structure of each sample
    sample = imagenette[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and content
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Expected a PIL Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Expected a 3D image array, got {image_np.ndim} dimensions."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Expected label to be an integer, got {type(label)}."
    valid_labels = list(range(10))  # Assuming labels range from 0 to 9 for imagenette
    assert label in valid_labels, f"Label {label} is not in the valid range {valid_labels}."

    # Test 5: Validate the test split
    imagenette_test = datasets.load_dataset("../../stable_datasets/images/imagenette.py", split="test", trust_remote_code=True)
    assert len(imagenette_test) > 0, "Expected non-zero samples in test split."

    print("All tests passed for Imagenet dataset!")


if __name__ == "__main__":
    test_imagenet_dataset()
