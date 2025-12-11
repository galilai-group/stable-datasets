import datasets
from PIL import Image
import numpy as np


def test_food101_dataset():
    # Load the Food101 dataset
    food101_train = datasets.load_dataset("../../stable_datasets/images/food101.py", split="train", trust_remote_code=True)
    food101_test = datasets.load_dataset("../../stable_datasets/images/food101.py", split="test", trust_remote_code=True)

    # Test 1: Check the number of samples in the train and test splits
    assert len(food101_train) == 75750, f"Expected 75750 training samples, got {len(food101_train)}."
    assert len(food101_test) == 25250, f"Expected 25250 test samples, got {len(food101_test)}."

    # Test 2: Check that each sample contains the keys "image" and "class"
    train_sample = food101_train[0]
    assert "image" in train_sample and "class" in train_sample, "Each sample must have 'image' and 'class' keys."

    # Test 3: Validate image type (PIL.Image) and ensure it can be converted to numpy
    image = train_sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally: convert to numpy array to check image shape
    image_np = np.array(image)
    assert len(image_np.shape) == 3 and image_np.shape[
        2] == 3, f"Image should have 3 color channels, got shape {image_np.shape}."

    # Test 4: Validate label type and validity
    label = train_sample["class"]
    assert isinstance(label, int), f"Class should be an integer, got {type(label)}."
    assert 0 <= label < len(food101_train.features["class"].names), f"Class index out of range: {label}."

    # Test 5: Check the labels against predefined categories
    class_names = food101_train.features["class"].names
    assert len(class_names) == 101, f"Expected 101 classes, got {len(class_names)}."
    assert "apple_pie" in class_names, "Class 'apple_pie' not found in the label list."

    # Test 6: Validate test image properties
    test_sample = food101_test[0]
    test_image = test_sample["image"]
    assert isinstance(test_image, Image.Image), f"Test image should be a PIL image, got {type(test_image)}."
    test_image_np = np.array(test_image)
    assert test_image_np.dtype == np.uint8, f"Test image should have dtype uint8, got {test_image_np.dtype}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_food101_dataset()
