import numpy as np
from PIL import Image

from stable_datasets.images.stl10 import STL10


def test_stl10_dataset():
    # STL10(split="train") automatically downloads and loads the dataset
    stl10_train = STL10(split="train")
    stl10_test = STL10(split="test")
    stl10_unlabeled = STL10(split="unlabeled")

    # Test 1: Check that the dataset has the expected number of samples
    assert len(stl10_train) == 5000, f"Expected 5000 training samples, got {len(stl10_train)}."
    assert len(stl10_test) == 8000, f"Expected 8000 test samples, got {len(stl10_test)}."
    assert len(stl10_unlabeled) == 100000, f"Expected 100000 unlabeled samples, got {len(stl10_unlabeled)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    train_sample = stl10_train[0]
    expected_keys = {"image", "label"}
    assert set(train_sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(train_sample.keys())}"

    # Test 3: Validate image type and shape
    image = train_sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy to validate shape
    image_np = np.array(image)
    assert image_np.shape == (96, 96, 3), f"Image should have shape (96, 96, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = train_sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 9, f"Label should be between 0 and 9, got {label}."

    # Test 5: Check unlabeled split has label -1
    unlabeled_sample = stl10_unlabeled[0]
    assert unlabeled_sample["label"] == -1, "Unlabeled samples should have label -1."

    print("All STL10 dataset tests passed successfully!")
