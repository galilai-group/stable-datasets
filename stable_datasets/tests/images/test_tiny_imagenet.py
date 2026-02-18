import numpy as np
from PIL import Image

from stable_datasets.images.tiny_imagenet import TinyImagenet


def test_tiny_imagenet_dataset():
    # Load train split
    tiny_train = TinyImagenet(split="train")

    # Test 1: Check that the train split has the expected number of samples
    expected_num_train_samples = 200 * 500  # 200 classes * 500 images each
    assert len(tiny_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(tiny_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = tiny_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (64, 64, 3), f"Image should have shape (64, 64, 3), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 200, f"Label should be between 0 and 199, got {label}."

    # Now load validation split and validate count (mirrors not_mnist's train/test checks)
    tiny_val = TinyImagenet(split="validation")
    expected_num_val_samples = 200 * 50
    assert len(tiny_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(tiny_val)}."
    )

    tiny_test = TinyImagenet(split="test")
    expected_num_test_samples = 10000  # Tiny ImageNet test set has 10,000 images
    assert len(tiny_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(tiny_test)}."
    )

    print("All Tiny ImageNet dataset tests passed successfully!")


if __name__ == "__main__":
    test_tiny_imagenet_dataset()
