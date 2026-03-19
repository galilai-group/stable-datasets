import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Galaxy10Decal


pytestmark = pytest.mark.large


def test_galaxy10_dataset():
    # Load training split (Galaxy10 only has one split as it's not pre-split)
    galaxy10_train = Galaxy10Decal(split="train")

    # Test 1: Check number of samples
    expected_num_train_samples = 17736
    assert len(galaxy10_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} samples, got {len(galaxy10_train)}."
    )

    # Test 2: Check sample keys
    sample = galaxy10_train[0]
    expected_keys = {"image", "label", "ra", "dec", "redshift", "pxscale"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."

    # Convert to numpy for basic sanity checks
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Galaxy10 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape == (256, 256, 3), f"Image should have shape (256, 256, 3), got {image_np.shape}."
    assert image_np.shape[2] == 3, f"Galaxy10 images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 10, f"Label should be in range [0, 9] (10 galaxy classes), got {label}."

    # Test 5: Validate astronomical metadata fields
    ra = sample["ra"]
    dec = sample["dec"]
    redshift = sample["redshift"]
    pxscale = sample["pxscale"]

    # Check types
    assert isinstance(ra, float | np.floating), f"RA (right ascension) should be float, got {type(ra)}."
    assert isinstance(dec, float | np.floating), f"DEC (declination) should be float, got {type(dec)}."
    assert isinstance(redshift, float | np.floating), f"Redshift should be float, got {type(redshift)}."
    assert isinstance(pxscale, float | np.floating), f"Pixel scale should be float, got {type(pxscale)}."

    # Check value ranges (basic sanity checks for astronomical data)
    # Note: Some galaxies may not have redshift measurements (NaN values are acceptable)
    # Also, due to measurement errors, small negative values close to 0 are acceptable
    assert 0 <= ra <= 360, f"RA should be in range [0, 360] degrees, got {ra}."
    assert -90 <= dec <= 90, f"DEC should be in range [-90, 90] degrees, got {dec}."
    assert np.isnan(redshift) or redshift >= -0.01, (
        f"Redshift should be non-negative (or small negative due to measurement error), got {redshift}."
    )
    assert pxscale > 0, f"Pixel scale should be positive (arcsec/pixel), got {pxscale}."

    # Test 6: Verify class label names are correct
    expected_num_classes = 10
    class_names = [
        "Disturbed Galaxies",
        "Merging Galaxies",
        "Round Smooth Galaxies",
        "In-between Round Smooth Galaxies",
        "Cigar Shaped Smooth Galaxies",
        "Barred Spiral Galaxies",
        "Unbarred Tight Spiral Galaxies",
        "Unbarred Loose Spiral Galaxies",
        "Edge-on Galaxies without Bulge",
        "Edge-on Galaxies with Bulge",
    ]
    dataset_class_names = galaxy10_train.features["label"].names
    assert len(dataset_class_names) == expected_num_classes, (
        f"Expected {expected_num_classes} classes, got {len(dataset_class_names)}."
    )
    assert dataset_class_names == class_names, (
        f"Class names mismatch. Expected {class_names}, got {dataset_class_names}."
    )

    print("All Galaxy10 DECaLS dataset tests passed successfully!")
