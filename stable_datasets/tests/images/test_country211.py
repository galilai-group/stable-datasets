import datasets
from PIL import Image
import numpy as np


def test_country211_dataset():
    # Load the dataset for each split with trust_remote_code enabled
    country211_train = datasets.load_dataset("../../stable_datasets/images/country211.py", split="train",
                                             trust_remote_code=True)
    country211_valid = datasets.load_dataset("../../stable_datasets/images/country211.py", split="validation",
                                             trust_remote_code=True)
    country211_test = datasets.load_dataset("../../stable_datasets/images/country211.py", split="test",
                                            trust_remote_code=True)

    # Expected number of classes for Country211
    expected_num_classes = 211

    # Test 1: Check that each split has a non-zero number of samples
    assert len(country211_train) > 0, f"Expected non-zero samples in training set, but got {len(country211_train)}."
    assert len(country211_valid) > 0, f"Expected non-zero samples in validation set, but got {len(country211_valid)}."
    assert len(country211_test) > 0, f"Expected non-zero samples in test set, but got {len(country211_test)}."

    # Test 2: Verify that each sample in the training set has the keys "image" and "label"
    sample = country211_train[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate the "image" field is a PIL Image instance
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Expected 'image' to be a PIL Image, got {type(image)}."

    # Optionally: Convert the image to a numpy array and check dimensions
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Image should have shape (H, W, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate that the "label" field is an integer and within the expected range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < expected_num_classes, f"Label should be between 0 and {expected_num_classes - 1}, got {label}."

    # Test 5: Verify label consistency across splits
    label_names = country211_train.features["label"].names
    assert len(label_names) == expected_num_classes, f"Expected {expected_num_classes} classes, got {len(label_names)}."

    # Validate label consistency in validation and test splits
    for dataset_split, name in zip([country211_train, country211_valid, country211_test], ["train", "valid", "test"]):
        for idx, example in enumerate(dataset_split):
            assert 0 <= example[
                "label"] < expected_num_classes, f"In {name} split, example {idx} has invalid label {example['label']}."

    print("All Country211 dataset tests passed successfully!")


if __name__ == "__main__":
    test_country211_dataset()
