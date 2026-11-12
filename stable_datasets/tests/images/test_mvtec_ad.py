"""Tests for the MVTec-AD builder.

``toothbrush`` is the smallest category (~109 MB) so it stands in for the
download-backed checks; the 5 GB full bundle is marked ``large``.
"""

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.images import MVTecAD
from stable_datasets.images.mvtec_ad import _CATEGORY_URLS, MVTEC_AD_CATEGORIES


_EXPECTED_FEATURES = {"image", "mask", "label", "defect_type", "category", "image_path"}


def test_mvtec_ad_declares_a_config_per_category():
    names = {config.name for config in MVTecAD.BUILDER_CONFIGS}
    assert names == {"all", *MVTEC_AD_CATEGORIES}
    assert MVTecAD.DEFAULT_CONFIG_NAME == "all"


def test_mvtec_ad_every_category_has_its_own_archive():
    """Per-category archives are what make a single-category load cheap."""
    assert set(_CATEGORY_URLS) == set(MVTEC_AD_CATEGORIES)


def test_mvtec_ad_category_config_downloads_only_that_archive():
    # NB: MVTecAD(...) and MVTecAD.__new__(MVTecAD) both run the full download +
    # build pipeline, so metadata-only checks must bypass __new__ entirely.
    builder = object.__new__(MVTecAD)
    builder.__init__(config_name="toothbrush")
    assets = builder._source()["assets"]
    # Both splits live in one archive, so the two assets share a URL.
    assert {info.url for info in assets.values()} == {_CATEGORY_URLS["toothbrush"]}


def test_mvtec_ad_toothbrush_splits_and_schema(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "toothbrush",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = MVTecAD(split="train", **kwargs)
    test = MVTecAD(split="test", **kwargs)

    # Published MVTec-AD toothbrush counts.
    assert len(train) == 60
    assert len(test) == 42

    sample = train[0]
    assert set(sample.keys()) == _EXPECTED_FEATURES
    assert isinstance(sample["image"], PILImage.Image)
    assert sample["label"] == 0, "Training images are defect-free."
    assert sample["defect_type"] == "good"
    assert sample["mask"] is None, "Defect-free images carry no mask."
    assert sample["category"] == MVTEC_AD_CATEGORIES.index("toothbrush")
    assert not sample["image_path"].startswith("/"), "Cached paths must stay relative."


def test_mvtec_ad_test_split_mixes_normal_and_anomalous(tmp_path, shared_download_dir):
    test = MVTecAD(
        split="test",
        config_name="toothbrush",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )
    labels = [test[i]["label"] for i in range(len(test))]
    assert labels.count(0) == 12
    assert labels.count(1) == 30


def test_mvtec_ad_anomalous_rows_carry_a_mask(tmp_path, shared_download_dir):
    test = MVTecAD(
        split="test",
        config_name="toothbrush",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )

    for index in range(len(test)):
        sample = test[index]
        if sample["label"] == 1:
            mask = sample["mask"]
            assert isinstance(mask, PILImage.Image), "Anomalous test rows must carry a ground-truth mask."
            assert mask.size == sample["image"].size, "Mask and image must be pixel aligned."
            assert set(np.unique(np.asarray(mask))).issubset({0, 255}), "Masks are binary."
        else:
            assert sample["mask"] is None


@pytest.mark.parametrize("storage_format", ["arrow", "lance"])
def test_mvtec_ad_roundtrips_through_both_backends(tmp_path, shared_download_dir, storage_format):
    """Nullable masks must survive whichever storage layout the cache uses."""
    test = MVTecAD(
        split="test",
        config_name="toothbrush",
        storage_format=storage_format,
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / storage_format),
    )
    assert len(test) == 42

    labels = [test[i]["label"] for i in range(len(test))]
    assert labels.count(0) == 12
    assert labels.count(1) == 30

    anomalous = next(i for i in range(len(test)) if test[i]["label"] == 1)
    normal = next(i for i in range(len(test)) if test[i]["label"] == 0)
    assert isinstance(test[anomalous]["mask"], PILImage.Image)
    assert test[normal]["mask"] is None, "A null mask must stay null in both backends."


@pytest.mark.large
def test_mvtec_ad_all_categories(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "all",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = MVTecAD(split="train", **kwargs)
    test = MVTecAD(split="test", **kwargs)

    # Published MVTec-AD totals.
    assert len(train) == 3629
    assert len(test) == 1725

    categories = {train[i]["category"] for i in range(0, len(train), 97)}
    assert len(categories) > 1, "The 'all' config must span multiple categories."
