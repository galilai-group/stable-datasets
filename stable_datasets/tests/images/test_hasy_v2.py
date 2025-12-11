import datasets
from PIL import Image
import numpy as np


def test_hasyv2_dataset():
    # Load the HASYv2 dataset
    hasyv2_train = datasets.load_dataset("../../stable_datasets/images/hasy_v2.py", split="train", trust_remote_code=True)
    hasyv2_test = datasets.load_dataset("../../stable_datasets/images/hasy_v2.py", split="test", trust_remote_code=True)

    # Test 1: Validate dataset length
    assert len(hasyv2_train) == 151241, f"Expected 151241 samples in train dataset, got {len(hasyv2_train)}."
    assert len(hasyv2_test) == 16992, f"Expected 16992 samples in test dataset, got {len(hasyv2_test)}."

    # Test 2: Validate each sample contains "image" and "label"
    train_sample = hasyv2_train[0]
    assert "image" in train_sample and "label" in train_sample, "Each sample must have 'image' and 'label' keys."

    # Test 3: Validate image type (PIL.Image)
    image = train_sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    # Test 4: Validate image shape and data type
    image_np = np.array(image)
    assert image_np.shape == (32, 32), f"Image should have shape (32, 32), got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 5: Validate label type
    label = train_sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."

    # Test 6: Validate unique labels
    unique_labels_train = set(hasyv2_train["label"])
    unique_labels_test = set(hasyv2_test["label"])
    assert len(unique_labels_train) > 0, "Train dataset must have at least one unique label."
    assert len(unique_labels_test) > 0, "Test dataset must have at least one unique label."

    # Test 7: Validate label range
    max_label = max(unique_labels_train.union(unique_labels_test))
    assert max_label < 369, f"Label values must be less than 369, got {max_label}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_hasyv2_dataset()
