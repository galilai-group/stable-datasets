import numpy as np
from PIL import Image

from stable_datasets.images.rsscn7 import RSSCN7


def test_rsscn7_dataset():
    # RSSCN7(split="train") automatically downloads and loads the dataset
    rsscn7_train = RSSCN7(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    # 7 classes * 200 images per class = 1,400 training samples
    expected_num_train_samples = 2800
    assert len(rsscn7_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(rsscn7_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = rsscn7_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Test 4: Validate image shape (convert to numpy array)
    image_np = np.array(image)
    assert image_np.shape == (400, 400, 3), f"Image should have shape (400, 400, 3), got {image_np.shape}"

    # Test 5: Validate image dtype
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 6: Validate label type
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."

    # Test 7: Validate label range (0-6 for 7 classes)
    assert 0 <= label < 7, f"Label should be between 0 and 6, got {label}."

    # Test 8: Validate class names
    class_names = rsscn7_train.features["label"].names
    expected_class_names = ["aGrass", "bField", "cIndustry", "dRiverLake", "eForest", "fResident", "gParking"]
    assert class_names == expected_class_names, f"Expected class names {expected_class_names}, got {class_names}"

    # Test 9: Check that each class has the expected number of samples in train split
    label_counts = {}
    for i in range(len(rsscn7_train)):
        label = rsscn7_train[i]["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Each class should have 200 images in training
    for label_idx in range(7):
        assert label_counts[label_idx] == 400, (
            f"Class {label_idx} should have 400 training samples, got {label_counts[label_idx]}."
        )

    print("All RSSCN7 dataset tests passed successfully!")


if __name__ == "__main__":
    test_rsscn7_dataset()
