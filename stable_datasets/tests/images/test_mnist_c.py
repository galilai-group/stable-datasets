import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import MNISTC


EXPECTED_CONFIGS = [
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
    "identity",
]


def test_mnist_c_builder_configs():
    """Every MNIST-C corruption (plus the clean ``identity`` reference) is exposed as a config."""
    assert [c.name for c in MNISTC.BUILDER_CONFIGS] == EXPECTED_CONFIGS
    assert MNISTC.DEFAULT_CONFIG_NAME == "shot_noise"

    with pytest.raises(ValueError, match="Unknown config"):
        MNISTC(config_name="not_a_corruption", split="test")


@pytest.mark.parametrize("config_name", ["shot_noise", "canny_edges", "identity"])
def test_mnist_c_dataset(config_name):
    ds_train = MNISTC(config_name=config_name, split="train")
    ds_test = MNISTC(config_name=config_name, split="test")

    # Test 1: Check expected split sizes (MNIST train/test)
    assert len(ds_train) == 60000, f"Expected 60000 train samples, got {len(ds_train)}."
    assert len(ds_test) == 10000, f"Expected 10000 test samples, got {len(ds_test)}."

    # Test 2: Check keys
    expected_keys = {"image", "label"}
    for ds in (ds_train, ds_test):
        sample = ds[0]
        assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, mode, shape and dtype
    image = ds_test[0]["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    assert image.mode == "L", f"Image should be grayscale (mode 'L'), got mode {image.mode}."
    image_np = np.array(image)
    assert image_np.shape == (28, 28), f"Image should have shape (28, 28), got {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range over the full split
    label = ds_test[0]["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    labels = ds_train.table.column("label").to_numpy()
    assert labels.min() == 0 and labels.max() == 9, f"Labels should span 0-9, got {labels.min()}-{labels.max()}."
    assert len(np.unique(labels)) == 10, "Expected all 10 digit classes in the train split."

    # Test 5: Validate class names and config metadata
    assert ds_train.info.features["label"].names == [str(i) for i in range(10)]
    assert ds_train.info.config_name == config_name
    assert ds_train.info.license == "CC BY 4.0"


def test_mnist_c_corruptions_share_labels_but_differ_from_identity():
    """Corruptions are applied to the same MNIST images in the same order as the clean reference."""
    clean = MNISTC(config_name="identity", split="test")
    corrupted = MNISTC(config_name="canny_edges", split="test")

    clean_labels = clean.table.column("label").to_numpy()
    corrupted_labels = corrupted.table.column("label").to_numpy()
    np.testing.assert_array_equal(clean_labels, corrupted_labels)

    clean_images = np.stack([np.array(clean[i]["image"]) for i in range(16)])
    corrupted_images = np.stack([np.array(corrupted[i]["image"]) for i in range(16)])
    assert not np.array_equal(clean_images, corrupted_images), "Corrupted images should differ from identity."
