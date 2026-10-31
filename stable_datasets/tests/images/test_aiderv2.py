import numpy as np
from PIL import Image

from stable_datasets.images import AIDERv2


def test_aiderv2():
    # Test all dataset splits.
    expected_sizes = {
        "train": 13399,
        "validation": 1670,
        "test": 1654,
    }

    datasets = {split: AIDERv2(split=split) for split in expected_sizes}

    # Test 1: Check the expected number of samples.
    for split, expected_size in expected_sizes.items():
        assert len(datasets[split]) == expected_size, (
            f"Expected {expected_size} samples in {split}, got {len(datasets[split])}."
        )

    # Continue existing sample, image, and label checks using training data.
    aiderv2 = datasets["train"]

    # Test 2: Check that each sample has the expected keys.
    expected_keys = {"image", "label"}
    sample = aiderv2[0]
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and format.
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.mode == "RGB", f"Image should be RGB, got mode {image.mode}."

    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, (
        f"Image should have three RGB channels, got shape {image_np.shape}."
    )
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}."

    # Test 4: Validate image dimensions.
    expected_image_size = (224, 224)
    assert image.size == expected_image_size, f"Expected image size {expected_image_size}, got {image.size}."

    # Test 5: Validate the label is a valid integer class index.
    label = sample["label"]
    assert isinstance(label, (int, np.integer)), f"Label should be an integer, got {type(label)}."
    assert label in (0, 1, 2, 3), f"Label should be between 0 and 3, got {label}."

    # Test 6: Check label metadata.
    expected_classes = ["Earthquake", "Flood", "Normal", "Wildfire"]
    label_feature = aiderv2.info.features["label"]

    assert label_feature.names == expected_classes, f"Expected classes {expected_classes}, got {label_feature.names}."

    print("All AIDERv2 tests passed successfully!")
