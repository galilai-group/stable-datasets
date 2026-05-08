import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import EuroSAT


pytestmark = pytest.mark.large


_EXPECTED_COUNTS = {
    "train": 16200,
    "validation": 5400,
    "test": 5400,
}


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_eurosat_dataset(split):
    ds = EuroSAT(split=split)

    # Test 1: Check sample count
    expected = _EXPECTED_COUNTS[split]
    assert len(ds) == expected, f"[{split}] expected {expected} samples, got {len(ds)}."

    # Test 2: Check sample keys
    sample = ds[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"[{split}] expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type and dimensions
    image = sample["image"]
    assert isinstance(image, Image.Image), f"[{split}] image should be a PIL.Image.Image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.shape == (64, 64, 3), f"[{split}] EuroSAT images should be 64x64x3, got {image_np.shape}."
    assert image_np.dtype == np.uint8, f"[{split}] image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label
    label = sample["label"]
    assert isinstance(label, int), f"[{split}] label should be int, got {type(label)}."
    assert 0 <= label < 10, f"[{split}] label should be in [0, 9], got {label}."

    # Test 5: Verify class names
    expected_class_names = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    assert ds.features["label"].names == expected_class_names, (
        f"[{split}] class name mismatch. Expected {expected_class_names}, got {ds.features['label'].names}."
    )

    print(f"All EuroSAT[{split}] dataset tests passed successfully!")
