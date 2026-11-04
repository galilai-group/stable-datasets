import numpy as np
from PIL import Image

from stable_datasets.images import TUBerlin


def test_tu_berlin_dataset():
    # Load training split
    train = TUBerlin(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 16000
    assert len(train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(train)}."
    )

    # Test 2: Check sample keys
    sample = train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type, shape, and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.ndim == 2, f"TU-Berlin images should be single-channel HxW, got shape {image_np.shape}."
    assert image_np.shape[0] == image_np.shape[1], f"TU-Berlin sketches should be square, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 250, f"Label should be in range [0, 249], got {label}."

    # Test 5: Load and validate validation split
    val = TUBerlin(split="validation")
    expected_num_val_samples = 2000
    assert len(val) == expected_num_val_samples, f"Expected {expected_num_val_samples} val samples, got {len(val)}."

    # Test 6: Load and validate test split
    test = TUBerlin(split="test")
    expected_num_test_samples = 2000
    assert len(test) == expected_num_test_samples, f"Expected {expected_num_test_samples} test samples, got {len(test)}."