import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.resisc45 import RESISC45


@pytest.fixture
def resisc45_archive(tmp_path):
    archive_path = tmp_path / "NWPU-RESISC45.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for folder, filename, mode, size in [
            ("wetland", "wetland_001.jpg", "RGB", (17, 13)),
            ("airplane", "airplane_002.jpg", "L", (19, 11)),
            ("airplane", "airplane_001.jpg", "RGB", (23, 15)),
        ]:
            buffer = io.BytesIO()
            Image.new(mode, size, color=64).save(buffer, format="JPEG")
            archive.writestr(f"NWPU-RESISC45/{folder}/{filename}", buffer.getvalue())
        archive.writestr("NWPU-RESISC45/airplane/", b"")
        archive.writestr("NWPU-RESISC45/README.txt", "Dataset metadata")
        archive.writestr("__MACOSX/NWPU-RESISC45/airplane/._airplane_001.jpg", b"not an image")
    return archive_path


def test_resisc45_native_images_and_stable_labels(resisc45_archive):
    builder = object.__new__(RESISC45)
    builder.__init__()
    examples = list(builder._generate_examples(resisc45_archive, "train"))

    assert [key for key, _ in examples] == [
        "NWPU-RESISC45/airplane/airplane_001.jpg",
        "NWPU-RESISC45/airplane/airplane_002.jpg",
        "NWPU-RESISC45/wetland/wetland_001.jpg",
    ]
    assert [example["label"] for _, example in examples] == [0, 0, 44]
    assert [example["image"].size for _, example in examples] == [(23, 15), (19, 11), (17, 13)]
    for _, example in examples:
        assert set(example) == {"image", "label"}
        assert isinstance(example["image"], Image.Image)
        assert example["image"].mode == "RGB"
        assert np.asarray(example["image"]).dtype == np.uint8
        assert isinstance(example["label"], int)


def test_resisc45_full_collection_train_split(resisc45_archive, monkeypatch, tmp_path):
    calls = []

    def fake_download(urls, dest_folder):
        calls.append((list(urls), dest_folder))
        return [resisc45_archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_download)
    dataset = RESISC45(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "raw")

    assert list(dataset) == ["train"]
    assert len(dataset["train"]) == 3
    assert dataset["train"][1]["image"].size == (19, 11)
    assert dataset["train"].features["label"].names == RESISC45._labels()
    assert len(RESISC45._labels()) == len(set(RESISC45._labels())) == 45
    assert RESISC45._labels() == sorted(RESISC45._labels())
    assert calls == [([RESISC45.SOURCE["assets"]["train"]], tmp_path / "raw")]


def test_resisc45_rejects_unknown_class(tmp_path):
    archive_path = tmp_path / "unknown.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("NWPU-RESISC45/unknown/image.jpg", b"not an image")
    builder = object.__new__(RESISC45)
    with pytest.raises(ValueError, match="Unknown RESISC45 class"):
        list(builder._generate_examples(archive_path, "train"))


def test_resisc45_rejects_empty_archive(tmp_path):
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "No images")
    builder = object.__new__(RESISC45)
    with pytest.raises(ValueError, match="No RESISC45 images"):
        list(builder._generate_examples(archive_path, "train"))


@pytest.mark.large
def test_resisc45_dataset():
    dataset = RESISC45()
    assert list(dataset) == ["train"]
    train = dataset["train"]
    assert len(train) == 31500
    assert len(train.features["label"].names) == 45
    for index in [0, len(train) // 2, len(train) - 1]:
        sample = train[index]
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        assert sample["image"].size == (256, 256)
        assert np.asarray(sample["image"]).dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 45
