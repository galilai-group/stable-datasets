import numpy as np
from PIL import Image

from stable_datasets.images import LFW


def test_lfw_dataset():
    # Load training split (official View 1 development pairs)
    lfw_train = LFW(split="train")

    # Test 1: Check number of training pairs (1100 matched + 1100 mismatched).
    # Verify against the source file: `wc -l pairsDevTrain.txt` minus the header line.
    expected_num_train_samples = 2200
    assert len(lfw_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training pairs, got {len(lfw_train)}."
    )

    # Test 2: Check sample keys
    sample = lfw_train[0]
    expected_keys = {"image1", "image2", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate both images
    for key in ("image1", "image2"):
        image = sample[key]
        assert isinstance(image, Image.Image), f"{key} should be a PIL.Image.Image, got {type(image)}."
        image_np = np.array(image)
        assert image_np.shape == (250, 250, 3), f"LFW images should be 250x250x3, got {image_np.shape}."
        assert image_np.dtype == np.uint8, f"{key} dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range (0 = different, 1 = same)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert label in (0, 1), f"Label should be 0 or 1, got {label}."

    # Test 5: Both classes are present (sampled with a stride, so images are not all decoded)
    labels = {lfw_train[i]["label"] for i in range(0, len(lfw_train), 50)}
    assert labels == {0, 1}, f"Expected both matched and mismatched pairs, got labels {labels}."

    # Test 6: Load and validate the test split (500 matched + 500 mismatched).
    lfw_test = LFW(split="test")
    expected_num_test_samples = 1000
    assert len(lfw_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test pairs, got {len(lfw_test)}."
    )