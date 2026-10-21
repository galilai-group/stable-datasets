import numpy as np
from PIL import Image

from stable_datasets.images import RenderedSST2


def test_rendered_sst2_dataset():
    # Load training split
    sst2_train = RenderedSST2(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 6920
    assert len(sst2_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(sst2_train)}."
    )

    # Test 2: Check sample keys
    sample = sst2_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and shape
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (448, 448, 3), f"Rendered SST2 images should be 448x448x3, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert label in (0, 1), f"Label should be 0 or 1, got {label}."

    # Test 5: Validate label names
    assert sst2_train.info.features["label"].names == ["negative", "positive"]

    # Test 6: Load and validate validation and test splits
    sst2_val = RenderedSST2(split="valid")
    expected_num_val_samples = 872
    assert len(sst2_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(sst2_val)}."
    )

    sst2_test = RenderedSST2(split="test")
    expected_num_test_samples = 1821
    assert len(sst2_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(sst2_test)}."
    )

    print("All Rendered SST2 dataset tests passed successfully!")
