"""Tests for the VisA builder.

VisA ships as a single 1.9 GB archive covering all twelve objects, so every
download-backed check is marked ``large``. The metadata tests run offline.
"""

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.images import VisA
from stable_datasets.images.visa import _IMAGES_URL, _SPLIT_CSV_URL, VISA_OBJECTS


_EXPECTED_FEATURES = {"image", "mask", "label", "object", "image_path"}

# Per-object row counts from the official 1-class split CSV.
_TRAIN_COUNTS = {
    "candle": 900,
    "capsules": 542,
    "cashew": 450,
    "chewinggum": 453,
    "fryum": 450,
    "macaroni1": 900,
    "macaroni2": 900,
    "pcb1": 904,
    "pcb2": 901,
    "pcb3": 905,
    "pcb4": 904,
    "pipe_fryum": 450,
}
_TEST_COUNTS = {
    "candle": 200,
    "capsules": 160,
    "cashew": 150,
    "chewinggum": 150,
    "fryum": 150,
    "macaroni1": 200,
    "macaroni2": 200,
    "pcb1": 200,
    "pcb2": 200,
    "pcb3": 201,
    "pcb4": 201,
    "pipe_fryum": 150,
}


def test_visa_declares_a_config_per_object():
    names = {config.name for config in VisA.BUILDER_CONFIGS}
    assert names == {"all", *VISA_OBJECTS}
    assert VisA.DEFAULT_CONFIG_NAME == "all"


def test_visa_sources_images_and_split_separately():
    """The image archive carries no split definition; it comes from spot-diff."""
    assets = VisA.SOURCE["assets"]
    assert assets["images"].url == _IMAGES_URL
    assert assets["split_csv"].url == _SPLIT_CSV_URL


def test_visa_split_counts_sum_to_the_published_total():
    assert sum(_TRAIN_COUNTS.values()) == 8659
    assert sum(_TEST_COUNTS.values()) == 2162
    assert sum(_TRAIN_COUNTS.values()) + sum(_TEST_COUNTS.values()) == 10821


@pytest.mark.large
def test_visa_candle_splits_and_schema(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "candle",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = VisA(split="train", **kwargs)
    test = VisA(split="test", **kwargs)

    assert len(train) == _TRAIN_COUNTS["candle"]
    assert len(test) == _TEST_COUNTS["candle"]

    sample = train[0]
    assert set(sample.keys()) == _EXPECTED_FEATURES
    assert isinstance(sample["image"], PILImage.Image)
    assert sample["label"] == 0, "VisA training images are all normal."
    assert sample["mask"] is None
    assert sample["object"] == VISA_OBJECTS.index("candle")
    assert not sample["image_path"].startswith("/"), "Cached paths must stay relative."


@pytest.mark.large
def test_visa_train_is_normal_only(tmp_path, shared_download_dir):
    train = VisA(
        split="train",
        config_name="candle",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )
    assert {train[i]["label"] for i in range(len(train))} == {0}


@pytest.mark.large
def test_visa_anomalous_rows_carry_a_mask(tmp_path, shared_download_dir):
    test = VisA(
        split="test",
        config_name="candle",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )

    labels = [test[i]["label"] for i in range(len(test))]
    assert labels.count(1) == 100, "Every VisA object contributes 100 anomalous test images."

    for index in range(len(test)):
        sample = test[index]
        if sample["label"] == 1:
            mask = sample["mask"]
            assert isinstance(mask, PILImage.Image)
            assert mask.mode == "L", "Masks are single-channel label maps."
            # Unlike MVTec-AD's 0/255 binary masks, VisA masks are label maps:
            # 0 is background and each defect region carries its own small id
            # (values such as {0, 1, 6} occur), so only "some pixel is set" holds.
            array = np.asarray(mask)
            assert array.any(), "An anomalous image must have a non-empty defect region."
            assert mask.size == sample["image"].size, "Mask and image must be pixel aligned."
        else:
            assert sample["mask"] is None


@pytest.mark.large
def test_visa_all_objects(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "all",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    assert len(VisA(split="train", **kwargs)) == 8659
    assert len(VisA(split="test", **kwargs)) == 2162
