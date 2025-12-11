import datasets
from PIL import Image


def test_dtd_dataset():
    # Load the DTD dataset
    dtd = datasets.load_dataset("../../stable_datasets/images/dtd.py", split="train", trust_remote_code=True)

    # Test 1: Validate dataset size
    expected_test_size = 1880
    assert len(dtd) == expected_test_size, f"Expected {expected_test_size} samples in the test split, got {len(dtd)}."

    # Test 2: Validate features
    sample = dtd[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Sample keys mismatch. Expected {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image feature
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Expected image to be a PIL Image, got {type(image)}."
    assert image.size[0] >= 300 and image.size[1] >= 300 and image.size[0] <= 640 and image.size[1] <= 640, f"Expected image size to be between 300x300 and 640x640, got {image.size}."

    # Test 4: Validate label feature
    label = sample["label"]
    assert 0 <= label < 47, f"Expected label to be an integer between 0 and 46, got {label}."

    print("All DTD dataset tests passed successfully!")


if __name__ == "__main__":
    test_dtd_dataset()
