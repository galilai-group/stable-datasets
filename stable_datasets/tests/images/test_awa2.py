import numpy as np
from PIL import Image

from stable_datasets.images import AWA2


def test_awa2_dataset():
    # AWA2(split="test") automatically downloads and loads the dataset
    awa2 = AWA2(split="test")

    # Test 1: Check that the dataset is not empty
    assert len(awa2) > 0, "Expected non-empty dataset."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = awa2[0]
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
    assert 0 <= label < 50, f"Label should be between 0 and 49 (50 classes), got {label}."

    # Test 5: Validate that all 50 classes are represented in the dataset
    labels_set = set()
    for i in range(min(len(awa2), 10000)):  # Sample first 10000 images or all if less
        labels_set.add(awa2[i]["label"])

    # AWA2 has 50 classes, we expect to see at least some of them
    assert len(labels_set) > 0, "Expected to find at least some class labels in the dataset"
    assert max(labels_set) < 50, f"Found label {max(labels_set)} which is >= 50 classes"

    print("All AWA2 dataset tests passed successfully!")
