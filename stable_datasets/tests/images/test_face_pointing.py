import numpy as np
from PIL import Image

from stable_datasets.images.face_pointing import FacePointing


def test_face_pointing_dataset():
    # FacePointing(split="train") automatically downloads and loads the dataset
    fp = FacePointing(split="train")

    # Test 1: Check that the dataset has samples
    assert len(fp) > 0, "Expected training samples, but got an empty dataset."

    # Test 2: Check that each sample has the keys "image", "person_id", and "angles"
    sample = fp[0]
    expected_keys = {"image", "person_id", "angles"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and properties
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally convert to numpy array to check properties
    image_np = np.array(image)
    assert len(image_np.shape) == 3, f"Image should be 3D (H, W, C), got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate person_id type and range
    person_id = sample["person_id"]
    assert isinstance(person_id, int), f"person_id should be an integer, got {type(person_id)}."
    assert 1 <= person_id <= 15, f"person_id should be between 1 and 15, got {person_id}."

    # Test 5: Validate angles type and structure
    angles = sample["angles"]
    assert isinstance(angles, list), f"angles should be a list, got {type(angles)}."
    assert len(angles) == 2, f"angles should have 2 elements (vertical, horizontal), got {len(angles)}."

    for angle in angles:
        assert isinstance(angle, int), f"Each angle should be an integer, got {type(angle)}."

    # Test 6: Verify multiple persons are present
    person_ids = set()
    # Sample throughout the dataset, not just the first 100
    for i in range(0, len(fp), len(fp) // 10):  # Sample 10 points across dataset
        person_ids.add(fp[i]["person_id"])

    # Should have at least 2 persons (ideally more)
    assert len(person_ids) >= 2, f"Expected multiple persons in dataset, got {person_ids}"

    print("All FacePointing dataset tests passed successfully!")
