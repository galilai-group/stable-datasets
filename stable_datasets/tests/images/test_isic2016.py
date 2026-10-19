"""Offline ISIC2016 label joins, error handling and real cache tests."""

import io
import zipfile

import pytest
from PIL import Image

from stable_datasets.images import ISIC2016


@pytest.fixture
def original_assets(tmp_path):
    payloads = {}
    for prefix, ids in (("Training", (2, 1)), ("Test", (4, 3))):
        with zipfile.ZipFile(tmp_path / f"ISBI2016_ISIC_Part3_{prefix}_Data.zip", "w") as archive:
            for i in ids:
                image_id = f"ISIC_{i:07d}"
                buffer = io.BytesIO()
                Image.new("RGB", (8, 6), (30 * i, 10, 90)).save(buffer, format="JPEG")
                payloads[image_id] = buffer.getvalue()
                archive.writestr(f"{prefix}/{image_id}.jpg", payloads[image_id])
        csv_path = tmp_path / f"ISBI2016_ISIC_Part3_{prefix}_GroundTruth.csv"
        # Reverse archive order so an incorrect positional join is observable.
        rows = (
            "ISIC_0000001,malignant\nISIC_0000002,benign\n"
            if prefix == "Training"
            else ("ISIC_0000003,0.0\nISIC_0000004,1.0\n")
        )
        csv_path.write_text(rows, encoding="utf-8")
    return tmp_path, payloads


def test_isic2016_join_and_cold_warm_cache(original_assets, monkeypatch):
    root, payloads = original_assets
    kwargs = {"data_dir": root, "processed_cache_dir": root / "processed"}
    datasets = ISIC2016(**kwargs)
    assert set(datasets) == {"train", "test"}
    assert [row["label"] for row in datasets["train"]] == [1, 0]
    assert [row["label"] for row in datasets["test"]] == [0, 1]
    for ds in datasets.values():
        assert ds.features["label"].names == ["benign", "malignant"]
        for row in ds.with_format("raw"):
            assert row["image"] == payloads[row["image_id"]]
        assert ds[0]["image"].size == (8, 6)
    monkeypatch.setattr(ISIC2016, "_split_generators", lambda self: pytest.fail("cache hit rebuilt raw sources"))
    reopened = ISIC2016(**kwargs)
    assert reopened["test"].with_format("raw")[0]["image"] == payloads["ISIC_0000003"]
    with pytest.raises(ValueError, match="Split 'validation' not found"):
        ISIC2016(split="validation", **kwargs)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("image_id,label\n", "Invalid"),
        ("ISIC_0000001,unknown\n", "Invalid"),
        ("not_an_id,benign\n", "Invalid"),
        ("ISIC_0000001,benign,extra\n", "exactly two"),
        ("ISIC_0000001,benign\nISIC_0000001,malignant\n", "Duplicate"),
    ],
)
def test_isic2016_rejects_bad_labels(tmp_path, content, message):
    path = tmp_path / "labels.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        ISIC2016._read_labels(path, "train")


def test_isic2016_rejects_image_label_mismatch(original_assets):
    root, _ = original_assets
    labels = root / "labels.csv"
    labels.write_text("ISIC_0000001,benign\nISIC_9999999,malignant\n")
    with pytest.raises(ValueError, match="missing images=.*9999999.*unlabelled images=.*0000002"):
        list(
            object.__new__(ISIC2016)._generate_examples(
                root / "ISBI2016_ISIC_Part3_Training_Data.zip", labels, "train"
            )
        )


def test_isic2016_missing_local_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing original"):
        ISIC2016(data_dir=tmp_path, processed_cache_dir=tmp_path / "processed")


def test_isic2016_download_assets_are_not_splits(original_assets, monkeypatch):
    root, _ = original_assets
    calls = []

    def bulk_download(assets, dest_folder):
        assets = list(assets)
        calls.extend(assets)
        return [root / asset.url.rsplit("/", 1)[1] for asset in assets]

    monkeypatch.setattr("stable_datasets.images.isic2016.bulk_download", bulk_download)
    result = ISIC2016(processed_cache_dir=root / "downloaded")
    assert set(result) == {"train", "test"}
    assert len(calls) == 4
