import numpy as np
from PIL import Image

from stable_datasets.images.cars3d import CARS3D


def test_cars3d_dataset():
    cars3d_train = CARS3D(split="train")

    expected_num_train_samples = 183 * 24 * 4
    assert len(cars3d_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(cars3d_train)}."
    )

    sample = cars3d_train[0]
    expected_keys = {
        "image",
        "car_type",
        "elevation",
        "azimuth",
        "label",
    }
    assert sorted(set(sample.keys())) == sorted(set(expected_keys)), (
        f"Expected keys {expected_keys}, got {set(sample.keys())}."
    )

    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Cars3D images should be HxWx3, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."
    assert image_np.shape == (128, 128, 3), f"Image should have shape (128, 128, 3), got {image_np.shape}."

    label = sample["label"]
    assert isinstance(label, list), f"Label should be list, got {type(label)}."
    assert len(label) == 3, f"Label should have 3 elements, got {len(label)}."

    car_type = sample["car_type"]
    elevation = sample["elevation"]
    azimuth = sample["azimuth"]

    assert isinstance(car_type, int | np.integer), f"car_type should be int, got {type(car_type)}."
    assert isinstance(elevation, int | np.integer), f"elevation should be int, got {type(elevation)}."
    assert isinstance(azimuth, int | np.integer), f"azimuth should be int, got {type(azimuth)}."

    assert 0 <= car_type < 183, f"car_type should be in range [0, 182], got {car_type}."
    assert 0 <= azimuth < 24, f"azimuth should be in range [0, 23], got {azimuth}."
    assert 0 <= elevation < 4, f"elevation should be in range [0, 3], got {elevation}."

    assert label[0] == car_type, f"label[0] should equal car_type, got {label[0]} vs {car_type}."
    assert label[1] == elevation, f"label[1] should equal elevation, got {label[1]} vs {elevation}."
    assert label[2] == azimuth, f"label[2] should equal azimuth, got {label[2]} vs {azimuth}."

    print("All Cars3D dataset tests passed successfully!")
