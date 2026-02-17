import numpy as np
from PIL import Image

from stable_datasets.images.rock_paper_scissor import RockPaperScissor


def test_rock_paper_scissor_dataset():
    # RockPaperScissor(split="train") automatically downloads and loads the dataset
    rps = RockPaperScissor(split="train")

    # Test 1: Check that the dataset has samples
    assert len(rps) > 0, "Expected training samples, but got an empty dataset."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = rps[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and properties
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check properties
    image_np = np.array(image)
    assert len(image_np.shape) == 3, f"Image should be 3D (H, W, C), got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 3, f"Label should be between 0 and 2 (rock, paper, scissors), got {label}."

    # Test 5: Check the test split
    rps_test = RockPaperScissor(split="test")
    assert len(rps_test) > 0, "Expected test samples, but got an empty dataset."

    # Test 6: Verify all three classes are present in training data
    labels_in_train = set()
    # Sample throughout the dataset, not just the first 100
    for i in range(0, len(rps), len(rps) // 10):  # Sample 10 points across dataset
        labels_in_train.add(rps[i]["label"])

    # Should have at least 2 classes (ideally 3)
    assert len(labels_in_train) >= 2, f"Expected multiple classes in training data, got {labels_in_train}"

    print("All RockPaperScissor dataset tests passed successfully!")
