import datasets
from PIL import Image
import numpy as np


def test_celeb_a_dataset():
    # Load the dataset with trust_remote_code to allow loading from the custom script path
    celebA_train = datasets.load_dataset("../../stable_datasets/images/celeb_a.py", split="train", trust_remote_code=True)
    celebA_valid = datasets.load_dataset("../../stable_datasets/images/celeb_a.py", split="validation", trust_remote_code=True)
    celebA_test = datasets.load_dataset("../../stable_datasets/images/celeb_a.py", split="test", trust_remote_code=True)

    # Test 1: Check that each split has a non-zero number of samples
    assert len(celebA_train) > 0, f"Expected non-zero samples in training set, but got {len(celebA_train)}."
    assert len(celebA_valid) > 0, f"Expected non-zero samples in validation set, but got {len(celebA_valid)}."
    assert len(celebA_test) > 0, f"Expected non-zero samples in test set, but got {len(celebA_test)}."

    # Test 2: Verify that each sample in the training set has the keys "image" and "attributes"
    sample = celebA_train[0]
    assert "image" in sample and "attributes" in sample, "Each sample must have 'image' and 'attributes' keys."

    # Test 3: Validate the "image" field is a PIL Image instance
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Expected 'image' to be a PIL Image, got {type(image)}."

    # Optionally: convert the image to a numpy array and check dimensions
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Image should have shape (H, W, 3), got {image_np.shape}."

    # Test 4: Validate "attributes" field length and binary values (-1 or 1)
    attributes = sample["attributes"]
    assert len(attributes) == 40, f"Expected 40 attributes, got {len(attributes)}."
    assert all(attr in [-1, 1] for attr in attributes), f"Attributes should contain only -1 or 1, got {attributes}."

    # Test 5: Verify consistency of attribute length across splits
    for dataset_split, name in zip([celebA_train, celebA_valid, celebA_test], ["train", "validation", "test"]):
        for idx, example in enumerate(dataset_split):
            assert len(example["attributes"]) == 40, f"In {name} split, example {idx} has incorrect attribute length {len(example['attributes'])}."

    print("All tests passed successfully!")


if __name__ == "__main__":
    test_celeb_a_dataset()
