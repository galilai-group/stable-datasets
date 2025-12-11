import datasets
from PIL import Image
import numpy as np


def test_places365_small_dataset():
    # Load the Places365Small dataset
    places365_train = datasets.load_dataset("../../stable_datasets/images/places365_small.py", split="train", trust_remote_code=True)
    places365_val = datasets.load_dataset("../../stable_datasets/images/places365_small.py", split="validation", trust_remote_code=True)

    # Test 1: Check that the train dataset is not empty
    assert len(places365_train) > 0, "Train dataset should not be empty."

    # Test 2: Check that the validation dataset is not empty
    assert len(places365_val) > 0, "Validation dataset should not be empty."

    # Test 3: Check that each sample in train has the keys "image" and "label"
    train_sample = places365_train[0]
    assert "image" in train_sample and "label" in train_sample, "Each train sample must have 'image' and 'label' keys."

    # Test 4: Check that each sample in validation has the keys "image" and "label"
    val_sample = places365_val[0]
    assert "image" in val_sample and "label" in val_sample, "Each validation sample must have 'image' and 'label' keys."

    # Test 5: Validate image type (PIL.Image) in train split
    train_image = train_sample["image"]
    assert isinstance(train_image, Image.Image), f"Train image should be a PIL image, got {type(train_image)}."

    # Test 6: Validate image type (PIL.Image) in validation split
    val_image = val_sample["image"]
    assert isinstance(val_image, Image.Image), f"Validation image should be a PIL image, got {type(val_image)}."

    # Test 7: Validate train image dimensions
    train_image_np = np.array(train_image)
    assert train_image_np.shape == (256, 256, 3), f"Train image should have shape (256, 256, 3), got {train_image_np.shape}."

    # Test 8: Validate validation image dimensions
    val_image_np = np.array(val_image)
    assert val_image_np.shape == (256, 256, 3), f"Validation image should have shape (256, 256, 3), got {val_image_np.shape}."

    # Test 9: Validate label type in train split
    train_label = train_sample["label"]
    assert isinstance(train_label, int), f"Train label should be an integer, got {type(train_label)}."

    # Test 10: Validate label type in validation split
    val_label = val_sample["label"]
    assert isinstance(val_label, int), f"Validation label should be an integer, got {type(val_label)}."

    # Test 11: Check that train label is within valid range
    assert 0 <= train_label <= 364, f"Train label should be between 0 and 364, got {train_label}."

    # Test 12: Check that validation label is within valid range
    assert 0 <= val_label <= 364, f"Validation label should be between 0 and 364, got {val_label}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_places365_small_dataset()
