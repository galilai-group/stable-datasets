import datasets
from PIL import Image
import numpy as np


def test_stl10_dataset():
    # Load STL-10 dataset
    stl10_train = datasets.load_dataset("../../stable_datasets/images/stl10.py", split="train", trust_remote_code=True)
    stl10_test = datasets.load_dataset("../../stable_datasets/images/stl10.py", split="test", trust_remote_code=True)
    stl10_unlabeled = datasets.load_dataset("../../stable_datasets/images/stl10.py", split="unlabeled", trust_remote_code=True)

    # Test 1: Check the number of samples
    assert len(stl10_train) == 5000, f"Expected 5000 training samples, got {len(stl10_train)}."
    assert len(stl10_test) == 8000, f"Expected 8000 test samples, got {len(stl10_test)}."
    assert len(stl10_unlabeled) == 100000, f"Expected 100000 unlabeled samples, got {len(stl10_unlabeled)}."

    # Test 2: Validate sample keys and data types
    train_sample = stl10_train[0]
    assert "image" in train_sample and "label" in train_sample, "Sample must have 'image' and 'label' keys."
    image = train_sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Convert to numpy to validate shape
    image_np = np.array(image)
    assert image_np.shape == (96, 96, 3), f"Image should have shape (96, 96, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    label = train_sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label <= 9, f"Label should be between 0 and 9, got {label}."

    # Test 3: Check unlabeled split
    unlabeled_sample = stl10_unlabeled[0]
    assert unlabeled_sample["label"] == -1, "Unlabeled samples should have label -1."

    print("All tests passed!")


if __name__ == "__main__":
    test_stl10_dataset()
