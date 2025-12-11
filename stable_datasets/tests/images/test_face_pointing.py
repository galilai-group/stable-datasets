import datasets
from PIL import Image
import numpy as np


def test_face_pointing_dataset():
    # Load the Face Pointing dataset
    dataset = datasets.load_dataset("../../stable_datasets/images/face_pointing.py", split="train", trust_remote_code=True)

    # Test 1: Check dataset size
    assert len(dataset) > 0, f"Dataset should have samples, but found {len(dataset)}."

    # Test 2: Check sample keys
    sample = dataset[0]
    assert "image" in sample and "person_id" in sample and "angles" in sample, "Sample keys are incorrect."

    # Test 3: Validate image type and content
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Image should be 3-dimensional, got {image_np.ndim}."

    # Test 4: Validate angles and person_id
    person_id = sample["person_id"]
    angles = sample["angles"]
    assert isinstance(person_id, int) and 1 <= person_id <= 15, f"Invalid person_id: {person_id}."
    assert len(angles) == 2, f"Angles should have exactly 2 values, got {len(angles)}."
    assert all(-90 <= angle <= 90 for angle in angles), f"Angles out of range: {angles}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_face_pointing_dataset()
