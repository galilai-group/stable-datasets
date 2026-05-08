import numpy as np
from PIL import Image

from stable_datasets.images import DSprites


def test_dsprites_variants():
    """Download/integration test for all DSprites variants."""

    variants = ["original", "color", "noise", "scream"]

    for variant in variants:
        ds = DSprites(split="train", config_name=variant)

        # Check dataset size
        expected_num_samples = 737280
        assert len(ds) == expected_num_samples, f"{variant}: Expected {expected_num_samples} samples, got {len(ds)}."

        sample = ds[0]

        # Check keys
        expected_keys = {
            "image",
            "index",
            "label",
            "label_values",
            "color",
            "shape",
            "scale",
            "orientation",
            "posX",
            "posY",
            "colorValue",
            "shapeValue",
            "scaleValue",
            "orientationValue",
            "posXValue",
            "posYValue",
        }
        if variant == "color":
            expected_keys.add("colorRGB")

        assert set(sample.keys()) == expected_keys, (
            f"{variant}: Expected keys {sorted(expected_keys)}, got {sorted(sample.keys())}."
        )

        # Check image type and shape
        image = sample["image"]
        assert isinstance(image, Image.Image), f"{variant}: 'image' should be PIL.Image, got {type(image)}."
        image_np = np.array(image)
        assert image_np.dtype == np.uint8, f"{variant}: Image dtype should be uint8, got {image_np.dtype}."

        if variant == "original":
            assert image_np.shape == (64, 64), f"{variant}: Expected shape (64, 64), got {image_np.shape}."
        else:
            assert image_np.shape == (64, 64, 3), f"{variant}: Expected shape (64, 64, 3), got {image_np.shape}."

        # Check label structure
        label = sample["label"]
        label_values = sample["label_values"]
        assert isinstance(label, list), f"{variant}: label should be list, got {type(label)}."
        assert isinstance(label_values, list), f"{variant}: label_values should be list, got {type(label_values)}."
        assert len(label) == 6, f"{variant}: label should have 6 elements, got {len(label)}."
        assert len(label_values) == 6, f"{variant}: label_values should have 6 elements, got {len(label_values)}."

        # Check factor ranges
        assert 0 <= sample["color"] < 1, f"{variant}: color out of range."
        assert 0 <= sample["shape"] < 3, f"{variant}: shape out of range."
        assert 0 <= sample["scale"] < 6, f"{variant}: scale out of range."
        assert 0 <= sample["orientation"] < 40, f"{variant}: orientation out of range."
        assert 0 <= sample["posX"] < 32, f"{variant}: posX out of range."
        assert 0 <= sample["posY"] < 32, f"{variant}: posY out of range."

        # Check factor values
        assert sample["colorValue"] == 1.0, f"{variant}: colorValue should be 1.0."
        assert sample["shapeValue"] in [1.0, 2.0, 3.0], f"{variant}: shapeValue out of range."
        assert 0.5 <= sample["scaleValue"] <= 1, f"{variant}: scaleValue out of range."
        assert 0 <= sample["orientationValue"] <= 2 * np.pi, f"{variant}: orientationValue out of range."
        assert 0 <= sample["posXValue"] <= 1, f"{variant}: posXValue out of range."
        assert 0 <= sample["posYValue"] <= 1, f"{variant}: posYValue out of range."

        # Variant-specific checks
        if variant == "color":
            color_rgb = sample["colorRGB"]
            assert isinstance(color_rgb, list), f"{variant}: colorRGB should be list."
            assert len(color_rgb) == 3, f"{variant}: colorRGB should have 3 elements."
            for c in color_rgb:
                assert 0.5 <= c <= 1.0, f"{variant}: colorRGB values should be in [0.5, 1.0], got {c}."
