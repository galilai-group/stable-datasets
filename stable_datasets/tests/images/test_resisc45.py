import numpy as np
from PIL import Image

from stable_datasets.images.resisc45 import RESISC45


def test_resisc45_dataset():
    # RESISC45(split="train") automatically downloads and loads the dataset
    resisc45_train = RESISC45(split="train")

    # Test 1: Check that the dataset
    assert len(resisc45_train) > 0, "Training set should not be empty."
    print(f"Training samples: {len(resisc45_train)}")

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = resisc45_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Test 4: Validate image shape (convert to numpy array)
    image_np = np.array(image)
    assert image_np.shape == (256, 256, 3), f"Image should have shape (256, 256, 3), got {image_np.shape}"

    # Test 5: Validate image dtype
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 6: Validate label type
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."

    # Test 7: Validate label range (0-44 for 45 classes)
    assert 0 <= label < 45, f"Label should be between 0 and 44, got {label}."

    # Test 8: Validate class names
    class_names = resisc45_train.features["label"].names
    expected_num_classes = 45
    assert len(class_names) == expected_num_classes, (
        f"Expected {expected_num_classes} classes, got {len(class_names)}."
    )

    # Test 10: Check that some expected class names are present
    expected_classes = ["airplane", "airport", "beach", "bridge", "forest", "harbor", "stadium"]
    for expected_class in expected_classes:
        assert expected_class in class_names, f"Expected class '{expected_class}' not found in class names."

    # Test 10: Verify total dataset size is 31,500
    assert len(resisc45_train) == 31500, f"Expected total samples to be around 31,500, got {len(resisc45_train)}."

    print("All RESISC45 dataset tests passed successfully!")


if __name__ == "__main__":
    test_resisc45_dataset()
