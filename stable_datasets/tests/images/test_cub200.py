import datasets
from PIL import Image
import numpy as np


def test_cub200_dataset():
    # Load the CUB200 dataset for the training split
    cub200_train = datasets.load_dataset("../../stable_datasets/images/cub200.py", split="train", trust_remote_code=True)
    cub200_test = datasets.load_dataset("../../stable_datasets/images/cub200.py", split="test", trust_remote_code=True)

    # Expected number of samples in each split
    expected_num_train_samples = 5994
    expected_num_test_samples = 5794

    # Test 1: Check that the train split has the expected number of samples
    assert len(cub200_train) == expected_num_train_samples, f"Expected {expected_num_train_samples} training samples, got {len(cub200_train)}."

    # Test 2: Check that the test split has the expected number of samples
    assert len(cub200_test) == expected_num_test_samples, f"Expected {expected_num_test_samples} test samples, got {len(cub200_test)}."

    # Test 3: Check that each sample in the train split has the expected keys "image" and "label"
    train_sample = cub200_train[0]
    expected_keys = {"image", "label"}
    assert set(train_sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(train_sample.keys())}"

    # Test 4: Check that each sample in the test split has the expected keys "image" and "label"
    test_sample = cub200_test[0]
    assert set(test_sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(test_sample.keys())}"

    # Test 5: Validate image type and shape in train split
    train_image = train_sample["image"]
    assert isinstance(train_image, Image.Image), f"Image should be a PIL image, got {type(train_image)}."

    # Optionally convert to numpy array to check shape if needed
    train_image_np = np.array(train_image)
    assert train_image_np.shape[2] == 3, f"Image should have 3 channels, got {train_image_np.shape[2]}"
    assert train_image_np.dtype == np.uint8, f"Image should have dtype uint8, got {train_image_np.dtype}"

    # Test 6: Validate image type and shape in test split
    test_image = test_sample["image"]
    assert isinstance(test_image, Image.Image), f"Image should be a PIL image, got {type(test_image)}."

    # Convert to numpy array to check shape if needed
    test_image_np = np.array(test_image)
    assert test_image_np.shape[2] == 3, f"Image should have 3 channels, got {test_image_np.shape[2]}"
    assert test_image_np.dtype == np.uint8, f"Image should have dtype uint8, got {test_image_np.dtype}"

    # Test 7: Validate label type and range in train split
    train_label = train_sample["label"]
    assert isinstance(train_label, int), f"Label should be an integer, got {type(train_label)}."
    assert 0 <= train_label < 200, f"Label should be between 0 and 199, got {train_label}."

    # Test 8: Validate label type and range in test split
    test_label = test_sample["label"]
    assert isinstance(test_label, int), f"Label should be an integer, got {type(test_label)}."
    assert 0 <= test_label < 200, f"Label should be between 0 and 199, got {test_label}."

    print("All CUB200 dataset tests passed successfully!")


if __name__ == "__main__":
    test_cub200_dataset()
