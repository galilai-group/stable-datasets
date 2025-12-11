import datasets
from PIL import Image
import numpy as np


def test_awa2_dataset():
    # Load the AWA2 dataset
    awa2 = datasets.load_dataset("../../stable_datasets/images/awa2.py", split="test", trust_remote_code=True)

    # Test 1: Check that the dataset has 37,322 samples
    assert len(awa2) == 37322, f"Expected 37,322 samples, got {len(awa2)}."

    # Test 2: Each sample should contain "image" and "label" keys
    sample = awa2[0]
    assert "image" in sample and "label" in sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type and dimensions
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Optionally: convert to numpy array to check shape if needed
    image_np = np.array(image)
    assert len(image_np.shape) == 3 and image_np.shape[2] == 3, f"Image should have 3 channels, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 50, f"Label should be between 0 and 49, got {label}."

    # Test 5: Verify label consistency with predefined class names in DatasetInfo
    class_names = awa2.info.features["label"].names
    assert len(class_names) == 50, f"Expected 50 class names, got {len(class_names)}."

    # Check that each label maps to the correct class name
    sample_class_name = class_names[label]
    assert sample_class_name in class_names, f"Sample label {label} does not correspond to a valid class name."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_awa2_dataset()
