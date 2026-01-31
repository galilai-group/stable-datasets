import numpy as np
from PIL import Image

from stable_datasets.images.beans import Beans


def test_beans_dataset():
    # Test train split
    beans_train = Beans(split="train")

    # Test 1: Check that the dataset is not empty
    assert len(beans_train) > 0, "Expected non-empty train dataset."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = beans_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy array to check properties
    image_np = np.array(image)
    assert len(image_np.shape) == 3, f"Image should have 3 dimensions (H, W, C), got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]} channels"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 3, f"Label should be between 0 and 2 (3 classes), got {label}."

    # Test 5: Test validation split
    beans_val = Beans(split="validation")
    assert len(beans_val) > 0, "Expected non-empty validation dataset."
    val_sample = beans_val[0]
    assert set(val_sample.keys()) == expected_keys, f"Expected keys {expected_keys} in validation split"

    # Test 6: Test test split
    beans_test = Beans(split="test")
    assert len(beans_test) > 0, "Expected non-empty test dataset."
    test_sample = beans_test[0]
    assert set(test_sample.keys()) == expected_keys, f"Expected keys {expected_keys} in test split"

    # Test 7: Validate label names match expected classes
    # Classes should be: healthy, angular_leaf_spot, bean_rust
    expected_classes = ["healthy", "angular_leaf_spot", "bean_rust"]
    info = beans_train.info
    assert info.features["label"].names == expected_classes, (
        f"Expected class names {expected_classes}, got {info.features['label'].names}"
    )

    print("All Beans dataset tests passed successfully!")
