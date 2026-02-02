import numpy as np
from PIL import Image

from stable_datasets.images.dsprites_noise import DSpritesNoise


def test_dsprites_noise_dataset():
    # Load training split
    dsprites_train = DSpritesNoise(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 737280
    assert len(dsprites_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(dsprites_train)}."
    )

    # Test 2: Check sample keys and label range
    sample = dsprites_train[0]
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
    assert sorted(set(sample.keys())) == sorted(set(expected_keys)), (
        f"Expected keys {expected_keys}, got {set(sample.keys())}."
    )

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"DSprites images should be HxWx3, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."
    assert image_np.shape == (64, 64, 3), f"Image should have shape (64, 64, 3), got {image_np.shape}"

    # Test 4: Validate label type and range
    label = sample["label"]
    label_values = sample["label_values"]
    assert isinstance(label, list), f"Label should be list, got {type(label)}."
    assert isinstance(label_values, list), f"Label values should be list, got {type(label_values)}."
    assert len(label) == 6, f"Label should have 6 elements, got {len(label)}."
    assert len(label_values) == 6, f"Label values should have 6 elements, got {len(label_values)}."

    color = sample["color"]
    shape = sample["shape"]
    scale = sample["scale"]
    orientation = sample["orientation"]
    posX = sample["posX"]
    posY = sample["posY"]
    colorValue = sample["colorValue"]
    shapeValue = sample["shapeValue"]
    scaleValue = sample["scaleValue"]
    orientationValue = sample["orientationValue"]
    posXValue = sample["posXValue"]
    posYValue = sample["posYValue"]

    assert 0 <= color < 1, f"Color should be in range [0, 0], got {color}."
    assert 0 <= shape < 3, f"Shape should be in range [0, 2], got {shape}."
    assert 0 <= scale < 6, f"Scale should be in range [0, 5], got {scale}."
    assert 0 <= orientation < 40, f"Orientation should be in range [0, 39], got {orientation}."
    assert 0 <= posX < 32, f"PosX should be in range [0, 31], got {posX}."
    assert 0 <= posY < 32, f"PosY should be in range [0, 31], got {posY}."
    assert colorValue == 1.0, f"Color value should be 1.0, got {colorValue}."
    assert shapeValue in [1.0, 2.0, 3.0], f"Shape value should be in [1.0, 2.0, 3.0], got {shapeValue}."
    assert 0.5 <= scaleValue <= 1, f"Scale value should be in range [0.5, 1], got {scaleValue}."
    assert 0 <= orientationValue <= 2 * np.pi, (
        f"Orientation value should be in range [0, 2pi], got {orientationValue}."
    )
    assert 0 <= posXValue <= 1, f"PosX value should be in range [0, 1], got {posXValue}."
    assert 0 <= posYValue <= 1, f"PosY value should be in range [0, 1], got {posYValue}."

    print("All DSpritesNoise dataset tests passed successfully!")
