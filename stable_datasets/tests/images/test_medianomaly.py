"""Tests for the MedIAnomaly builder.

``BrainTumor`` is the smallest Zenodo subset (~42 MB) so it backs the
download-driven checks; the full 2.4 GB ``all`` config is marked ``large``.
"""

import pytest
from PIL import Image as PILImage

from stable_datasets.images import MedIAnomaly
from stable_datasets.images.medianomaly import (
    _ISIC_SUBSET,
    _ZENODO_SUBSETS,
    MEDIANOMALY_SUBSETS,
)


_EXPECTED_FEATURES = {"image", "label", "subset", "image_path"}


def test_medianomaly_configs_cover_every_subset():
    names = {config.name for config in MedIAnomaly.BUILDER_CONFIGS}
    assert names == {"all", *MEDIANOMALY_SUBSETS}
    assert MedIAnomaly.DEFAULT_CONFIG_NAME == "all"


def test_medianomaly_zenodo_assets_carry_checksums():
    """Zenodo publishes md5 digests, and download() verifies the "algo:hex" form."""
    builder = object.__new__(MedIAnomaly)
    builder.__init__(config_name="all")

    assets = builder._source()["assets"]
    assert set(assets) == set(_ZENODO_SUBSETS)
    for subset, info in assets.items():
        assert info.checksum == _ZENODO_SUBSETS[subset]
        assert info.checksum.startswith("md5:")
        # The API URL ends in /content, so an explicit filename is required.
        assert info.filename == f"{subset}.tar.gz"
        assert info.fallbacks, "Each subset should keep the classic Zenodo URL as a fallback."


def test_medianomaly_all_config_excludes_the_gated_subset():
    """ISIC2018_Task3 is not redistributed on Zenodo, so "all" must not request it."""
    builder = object.__new__(MedIAnomaly)
    builder.__init__(config_name="all")
    assert _ISIC_SUBSET not in builder._source()["assets"]


def test_medianomaly_brain_tumor_splits(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "BrainTumor",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = MedIAnomaly(split="train", **kwargs)
    test = MedIAnomaly(split="test", **kwargs)

    # Counts come from the subset's own data.json.
    assert len(train) == 1000
    assert len(test) == 1200

    sample = train[0]
    assert set(sample.keys()) == _EXPECTED_FEATURES
    assert isinstance(sample["image"], PILImage.Image)
    assert sample["subset"] == MEDIANOMALY_SUBSETS.index("BrainTumor")
    assert not sample["image_path"].startswith("/"), "Cached paths must stay relative and portable."


def test_medianomaly_train_is_normal_only(tmp_path, shared_download_dir):
    """The one-class protocol: training data contains no abnormal images."""
    train = MedIAnomaly(
        split="train",
        config_name="BrainTumor",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )
    assert {train[i]["label"] for i in range(len(train))} == {0}


def test_medianomaly_test_is_balanced(tmp_path, shared_download_dir):
    test = MedIAnomaly(
        split="test",
        config_name="BrainTumor",
        download_dir=shared_download_dir,
        processed_cache_dir=str(tmp_path / "processed"),
    )
    labels = [test[i]["label"] for i in range(len(test))]
    assert labels.count(0) == 600
    assert labels.count(1) == 600


def test_medianomaly_isic_without_data_dir_is_actionable(tmp_path, shared_download_dir):
    with pytest.raises(ValueError, match="challenge.isic-archive.com"):
        MedIAnomaly(
            split="train",
            config_name=_ISIC_SUBSET,
            download_dir=shared_download_dir,
            processed_cache_dir=str(tmp_path / "processed"),
        )


def test_medianomaly_isic_with_missing_data_dir_is_actionable(tmp_path, shared_download_dir):
    with pytest.raises(FileNotFoundError, match="ISIC2018_Task3"):
        MedIAnomaly(
            split="train",
            config_name=_ISIC_SUBSET,
            data_dir=str(tmp_path / "nothing-here"),
            download_dir=shared_download_dir,
            processed_cache_dir=str(tmp_path / "processed"),
        )


@pytest.mark.large
def test_medianomaly_all_subsets(tmp_path, shared_download_dir):
    kwargs = {
        "config_name": "all",
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = MedIAnomaly(split="train", **kwargs)
    test = MedIAnomaly(split="test", **kwargs)

    # Sum of the six Zenodo subsets.
    assert len(train) == 19650
    assert len(test) == 11831

    assert {train[i]["subset"] for i in range(0, len(train), 53)} == set(range(len(_ZENODO_SUBSETS))), (
        "The 'all' config must span every Zenodo subset."
    )


@pytest.mark.large
@pytest.mark.parametrize(
    ("subset", "train_rows", "test_normal", "test_abnormal"),
    [
        ("BrainTumor", 1000, 600, 600),
        ("BraTS2021", 4211, 828, 1948),
        ("LAG", 1500, 811, 811),
        ("RSNA", 3851, 1000, 1000),
        ("VinCXR", 4000, 1000, 1000),
        ("Camelyon16", 5088, 1120, 1113),
    ],
)
def test_medianomaly_subset_counts(tmp_path, shared_download_dir, subset, train_rows, test_normal, test_abnormal):
    kwargs = {
        "config_name": subset,
        "download_dir": shared_download_dir,
        "processed_cache_dir": str(tmp_path / "processed"),
    }
    train = MedIAnomaly(split="train", **kwargs)
    test = MedIAnomaly(split="test", **kwargs)

    assert len(train) == train_rows
    assert {train[i]["label"] for i in range(len(train))} == {0}, "One-class protocol: train is normal-only."

    labels = [test[i]["label"] for i in range(len(test))]
    assert labels.count(0) == test_normal
    assert labels.count(1) == test_abnormal
