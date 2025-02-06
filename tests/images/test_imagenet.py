import datasets
from PIL import Image
import numpy as np


def test_imagenet_dataset():
    # Load the ImageNet training split.
    # Note: The path below assumes that your ImageNet dataset builder is located at
    imagenet_train = datasets.load_dataset(
        "../../aidatasets/images/imagenet.py",
        split="train",
        trust_remote_code=True,
    )

    # Test 1: Check that the training dataset has the expected number of samples.
    # The official ImageNet ILSVRC2012 training set contains 1,281,167 images.
    expected_train_length = 1281167
    assert len(imagenet_train) == expected_train_length, (
        f"Expected {expected_train_length} training samples, got {len(imagenet_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = imagenet_train[0]
    assert "image" in sample and "label" in sample, (
        "Each sample must have 'image' and 'label' keys."
    )

    # Test 3: Validate image type (PIL.Image)
    image = sample["image"]
    assert isinstance(image, Image.Image), (
        f"Image should be a PIL image, got {type(image)}."
    )

    # For ImageNet the image sizes vary; here we check that when converted to a NumPy array,
    # the image has 3 channels.
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, (
        f"Image should have 3 channels, got shape {image_np.shape}."
    )

    # Test 4: Validate label type and range (labels 0 to 999)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 999, f"Label should be between 0 and 999, got {label}."

    # Load the validation split.
    imagenet_val = datasets.load_dataset(
        "../../aidatasets/images/imagenet.py",
        split="validation",
        trust_remote_code=True,
    )
    expected_val_length = 50000
    assert len(imagenet_val) == expected_val_length, (
        f"Expected {expected_val_length} validation samples, got {len(imagenet_val)}."
    )

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_imagenet_dataset()
