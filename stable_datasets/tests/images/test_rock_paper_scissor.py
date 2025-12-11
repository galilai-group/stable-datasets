import datasets
from PIL import Image
import numpy as np


def test_rock_paper_scissor_dataset():
    # Load the train dataset
    rps_train = datasets.load_dataset("../../stable_datasets/images/rock_paper_scissor.py", split="train", trust_remote_code=True)

    # Test 1: Check train dataset size
    assert len(rps_train) > 0, "Train dataset should not be empty."

    # Test 2: Check sample keys in train split
    sample_train = rps_train[0]
    assert "image" in sample_train and "label" in sample_train, "Sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and shape in train split
    image_train = sample_train["image"]
    assert isinstance(image_train, Image.Image), f"Image should be of type PIL.Image, got {type(image_train)}."
    image_np_train = np.array(image_train)
    assert image_np_train.shape[-1] == 3, f"Image should have 3 channels, got shape {image_np_train.shape}."

    # Test 4: Validate label in train split
    label_train = sample_train["label"]
    assert isinstance(label_train, int), f"Label should be an integer, got {type(label_train)}."
    assert 0 <= label_train <= 2, f"Label should be between 0 and 2, got {label_train}."

    # Load the test dataset
    rps_test = datasets.load_dataset("../../stable_datasets/images/rock_paper_scissor.py", split="test", trust_remote_code=True)

    # Test 5: Check test dataset size
    assert len(rps_test) > 0, "Test dataset should not be empty."

    # Test 6: Check sample keys in test split
    sample_test = rps_test[0]
    assert "image" in sample_test and "label" in sample_test, "Sample must have 'image' and 'label' keys."

    # Test 7: Validate image type and shape in test split
    image_test = sample_test["image"]
    assert isinstance(image_test, Image.Image), f"Image should be of type PIL.Image, got {type(image_test)}."
    image_np_test = np.array(image_test)
    assert image_np_test.shape[-1] == 3, f"Image should have 3 channels, got shape {image_np_test.shape}."

    # Test 8: Validate label in test split
    label_test = sample_test["label"]
    assert isinstance(label_test, int), f"Label should be an integer, got {type(label_test)}."
    assert 0 <= label_test <= 2, f"Label should be between 0 and 2, got {label_test}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_rock_paper_scissor_dataset()
