import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import PlantVillage


pytestmark = pytest.mark.large


_EXPECTED_COUNTS = {
    "color": {"train": 43596, "test": 10709},
    "grayscale": {"train": 43203, "test": 11102},
    "segmented": {"train": 42984, "test": 11322},
}


@pytest.mark.parametrize("variant", ["color", "grayscale", "segmented"])
def test_plant_village_dataset(variant):
    # Test 1: Load training split and check count
    pv_train = PlantVillage(split="train", config_name=variant)
    expected_train = _EXPECTED_COUNTS[variant]["train"]
    assert len(pv_train) == expected_train, (
        f"[{variant}] expected {expected_train} training samples, got {len(pv_train)}."
    )

    # Test 2: Check sample keys
    sample = pv_train[0]
    expected_keys = {"image", "label", "crop", "disease"}
    assert set(sample.keys()) == expected_keys, f"[{variant}] expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"[{variant}] image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"[{variant}] images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"[{variant}] images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"[{variant}] image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label/crop/disease
    label = sample["label"]
    assert isinstance(label, int), f"[{variant}] label should be int, got {type(label)}."
    assert 0 <= label < 38, f"[{variant}] label should be in [0, 37], got {label}."

    crop = sample["crop"]
    disease = sample["disease"]
    assert isinstance(crop, str) and crop, f"[{variant}] crop must be a non-empty string, got {crop!r}."
    assert isinstance(disease, str) and disease, f"[{variant}] disease must be a non-empty string, got {disease!r}."

    # Test 5: Load test split and check count
    pv_test = PlantVillage(split="test", config_name=variant)
    expected_test = _EXPECTED_COUNTS[variant]["test"]
    assert len(pv_test) == expected_test, f"[{variant}] expected {expected_test} test samples, got {len(pv_test)}."

    print(f"All PlantVillage[{variant}] dataset tests passed successfully!")
