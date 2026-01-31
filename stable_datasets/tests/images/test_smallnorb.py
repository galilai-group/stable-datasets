import numpy as np
from PIL import Image

from stable_datasets.images.small_norb import SmallNORB


def test_smallnorb_dataset():
    smallnorb_train = SmallNORB(split="train")

    expected_num_train_samples = 24300
    assert len(smallnorb_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(smallnorb_train)}."
    )

    sample = smallnorb_train[0]
    expected_keys = {
        "left_image",
        "right_image",
        "label",
        "category",
        "instance",
        "elevation",
        "azimuth",
        "lighting",
    }
    assert sorted(set(sample.keys())) == sorted(set(expected_keys)), (
        f"Expected keys {expected_keys}, got {set(sample.keys())}."
    )

    left_image = sample["left_image"]
    assert isinstance(left_image, Image.Image), f"left_image should be a PIL.Image.Image, got {type(left_image)}."
    left_np = np.array(left_image)
    assert left_np.ndim == 2, f"SmallNORB images should be HxW, got shape {left_np.shape}."
    assert left_np.dtype == np.uint8, f"Image dtype should be uint8, got {left_np.dtype}."
    assert left_np.shape == (96, 96), f"Image should have shape (96, 96), got {left_np.shape}."

    right_image = sample["right_image"]
    assert isinstance(right_image, Image.Image), f"right_image should be a PIL.Image.Image, got {type(right_image)}."
    right_np = np.array(right_image)
    assert right_np.ndim == 2, f"SmallNORB images should be HxW, got shape {right_np.shape}."
    assert right_np.dtype == np.uint8, f"Image dtype should be uint8, got {right_np.dtype}."
    assert right_np.shape == (96, 96), f"Image should have shape (96, 96), got {right_np.shape}."

    label = sample["label"]
    assert isinstance(label, list), f"Label should be list, got {type(label)}."
    assert len(label) == 5, f"Label should have 5 elements, got {len(label)}."

    category = sample["category"]
    instance = sample["instance"]
    elevation = sample["elevation"]
    azimuth = sample["azimuth"]
    lighting = sample["lighting"]

    assert isinstance(category, int | np.integer), f"category should be int, got {type(category)}."
    assert isinstance(instance, int | np.integer), f"instance should be int, got {type(instance)}."
    assert isinstance(elevation, int | np.integer), f"elevation should be int, got {type(elevation)}."
    assert isinstance(azimuth, int | np.integer), f"azimuth should be int, got {type(azimuth)}."
    assert isinstance(lighting, int | np.integer), f"lighting should be int, got {type(lighting)}."

    assert 0 <= category < 5, f"category should be in range [0, 4], got {category}."
    assert 0 <= instance < 10, f"instance should be in range [0, 9], got {instance}."
    assert 0 <= elevation < 9, f"elevation should be in range [0, 8], got {elevation}."
    assert 0 <= azimuth < 18, f"azimuth should be in range [0, 17], got {azimuth}."
    assert 0 <= lighting < 6, f"lighting should be in range [0, 5], got {lighting}."

    assert label[0] == category, f"label[0] should equal category, got {label[0]} vs {category}."
    assert label[1] == instance, f"label[1] should equal instance, got {label[1]} vs {instance}."
    assert label[2] == elevation, f"label[2] should equal elevation, got {label[2]} vs {elevation}."
    assert label[3] == azimuth, f"label[3] should equal azimuth, got {label[3]} vs {azimuth}."
    assert label[4] == lighting, f"label[4] should equal lighting, got {label[4]} vs {lighting}."

    print("All SmallNORB dataset tests passed successfully!")
