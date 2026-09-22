import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.aid import AID


@pytest.fixture
def aid_archive(tmp_path):
    archive_path = tmp_path / "AID.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for folder, filename, mode, size in [
            ("Viaduct", "viaduct_1.jpg", "RGB", (17, 13)),
            ("Airport", "airport_2.jpg", "L", (19, 11)),
            ("Airport", "airport_1.jpg", "RGB", (23, 15)),
        ]:
            buffer = io.BytesIO()
            Image.new(mode, size, color=64).save(buffer, format="JPEG")
            archive.writestr(f"AID/AID/{folder}/{filename}", buffer.getvalue())
        archive.writestr("AID/AID/Airport/", b"")
        archive.writestr("AID/AID/README.txt", "Dataset metadata")
        archive.writestr("__MACOSX/AID/AID/Airport/._airport_1.jpg", b"not an image")
    return archive_path


def test_aid_native_images_and_stable_labels(aid_archive):
    builder = object.__new__(AID)
    builder.__init__()
    examples = list(builder._generate_examples(aid_archive, "train"))

    assert [key for key, _ in examples] == [
        "AID/AID/Airport/airport_1.jpg",
        "AID/AID/Airport/airport_2.jpg",
        "AID/AID/Viaduct/viaduct_1.jpg",
    ]
    assert [example["label"] for _, example in examples] == [0, 0, 29]
    assert [example["image"].size for _, example in examples] == [(23, 15), (19, 11), (17, 13)]
    for _, example in examples:
        assert set(example) == {"image", "label"}
        assert isinstance(example["image"], Image.Image)
        assert example["image"].mode == "RGB"
        assert np.asarray(example["image"]).dtype == np.uint8
        assert isinstance(example["label"], int)


def test_aid_full_collection_train_split(aid_archive, monkeypatch, tmp_path):
    calls = []

    def fake_download(urls, dest_folder):
        calls.append((list(urls), dest_folder))
        return [aid_archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_download)
    dataset = AID(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "raw")

    assert list(dataset) == ["train"]
    assert len(dataset["train"]) == 3
    assert dataset["train"][1]["image"].size == (19, 11)
    assert dataset["train"].features["label"].names == AID._labels()
    assert len(AID._labels()) == len(set(AID._labels())) == 30
    assert AID._labels() == sorted(AID._labels())
    assert calls == [([AID.SOURCE["assets"]["train"]], tmp_path / "raw")]


def test_aid_rejects_unknown_class(tmp_path):
    archive_path = tmp_path / "unknown.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("AID/AID/Unknown/image.jpg", b"not an image")
    builder = object.__new__(AID)
    with pytest.raises(ValueError, match="Unknown AID class"):
        list(builder._generate_examples(archive_path, "train"))


def test_aid_rejects_empty_archive(tmp_path):
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "No images")
    builder = object.__new__(AID)
    with pytest.raises(ValueError, match="No AID images"):
        list(builder._generate_examples(archive_path, "train"))


@pytest.mark.large
def test_aid_dataset():
    dataset = AID()
    assert list(dataset) == ["train"]
    train = dataset["train"]
    assert len(train) == 10000
    assert len(train.features["label"].names) == 30
    for index in [0, len(train) // 2, len(train) - 1]:
        sample = train[index]
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        assert sample["image"].size == (600, 600)
        assert np.asarray(sample["image"]).dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 30
