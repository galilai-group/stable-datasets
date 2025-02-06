import datasets
from PIL import Image
import numpy as np


def test_imagenet100_dataset():
    # Load the ImageNet100 training split.
    imagenet100_train = datasets.load_dataset(
        "../../aidatasets/images/imagenet100.py",
        split="train",
        trust_remote_code=True
    )

    # Check that the training dataset is not empty.
    assert len(imagenet100_train) > 0, (
        f"Training dataset should contain at least one sample, got {len(imagenet100_train)}."
    )

    # Test that each sample has the keys "image" and "label".
    sample = imagenet100_train[0]
    assert "image" in sample and "label" in sample, (
        "Each sample must have 'image' and 'label' keys."
    )

    # Validate that the image is a PIL image.
    image = sample["image"]
    assert isinstance(image, Image.Image), (
        f"Image should be a PIL.Image, got {type(image)}."
    )

    # Validate that the image, when converted to a numpy array, has 3 channels (RGB).
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, (
        f"Image should have 3 channels (RGB), got shape {image_np.shape}."
    )

    # Validate the label type and range (0 to 99).
    label = sample["label"]
    assert isinstance(label, int), (
        f"Label should be an integer, got {type(label)}."
    )
    assert 0 <= label <= 99, (
        f"Label should be between 0 and 99 for ImageNet100, got {label}."
    )

    # Load the validation split.
    imagenet100_val = datasets.load_dataset(
        "../../aidatasets/images/imagenet100.py",
        split="validation",
        trust_remote_code=True
    )

    # Check that the validation dataset is not empty.
    assert len(imagenet100_val) > 0, (
        f"Validation dataset should contain at least one sample, got {len(imagenet100_val)}."
    )

    # Check a validation sample.
    sample_val = imagenet100_val[0]
    assert "image" in sample_val and "label" in sample_val, (
        "Each validation sample must have 'image' and 'label' keys."
    )

    image_val = sample_val["image"]
    assert isinstance(image_val, Image.Image), (
        f"Validation image should be a PIL.Image, got {type(image_val)}."
    )

    image_val_np = np.array(image_val)
    assert image_val_np.ndim == 3 and image_val_np.shape[2] == 3, (
        f"Validation image should have 3 channels (RGB), got shape {image_val_np.shape}."
    )

    label_val = sample_val["label"]
    assert isinstance(label_val, int), (
        f"Validation label should be an integer, got {type(label_val)}."
    )
    assert 0 <= label_val <= 99, (
        f"Validation label should be between 0 and 99, got {label_val}."
    )

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_imagenet100_dataset()
