import csv
import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.sign_language_mnist import SignLanguageMNIST


def make_test_zip(path, labels):
    """Create a small fake Sign Language MNIST ZIP file."""
    text = io.StringIO()
    writer = csv.writer(text)

    writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])

    for label in labels:
        writer.writerow([label] + [128] * 784)

    with ZipFile(path, "w") as archive:
        archive.writestr("sign_mnist_train.csv", text.getvalue())


def test_generate_examples(tmp_path):
    zip_path = tmp_path / "sample.zip"
    make_test_zip(zip_path, [0, 10, 24])

    builder = object.__new__(SignLanguageMNIST)
    samples = list(builder._generate_examples(zip_path, "train"))

    assert len(samples) == 3

    assert [sample["label"] for _, sample in samples] == [0, 9, 23]

    for _, sample in samples:
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].size == (28, 28)
        assert sample["image"].mode == "L"


def test_invalid_label(tmp_path):
    zip_path = tmp_path / "invalid.zip"
    make_test_zip(zip_path, [9])

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Invalid label"):
        list(builder._generate_examples(zip_path, "train"))


def test_all_label_mappings():
    """Check all 24 ASL classes and skipped letters."""
    mapping = SignLanguageMNIST.LABEL_MAP

    expected_original = set(range(26)) - {9, 25}

    assert set(mapping.keys()) == expected_original
    assert set(mapping.values()) == set(range(24))

    assert mapping[0] == 0
    assert mapping[10] == 9
    assert mapping[24] == 23


def test_invalid_csv_columns(tmp_path):
    """Reject a CSV row with missing pixel values."""
    zip_path = tmp_path / "bad_columns.zip"

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])
    writer.writerow([0] + [128] * 783)

    with ZipFile(zip_path, "w") as archive:
        archive.writestr("data.csv", text.getvalue())

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Invalid row"):
        list(builder._generate_examples(zip_path, "train"))


def test_pixel_values(tmp_path):
    """Verify pixel values survive CSV-to-image conversion."""
    zip_path = tmp_path / "pixels.zip"

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])

    pixels = [0, 127, 255] + [128] * 781
    writer.writerow([0] + pixels)

    with ZipFile(zip_path, "w") as archive:
        archive.writestr("data.csv", text.getvalue())

    builder = object.__new__(SignLanguageMNIST)
    samples = list(builder._generate_examples(zip_path, "train"))

    image = samples[0][1]["image"]

    assert image.size == (28, 28)
    assert image.mode == "L"
    assert np.asarray(image).flatten().tolist() == pixels


def test_invalid_pixel_value(tmp_path):
    """Reject pixel values outside the valid range."""
    zip_path = tmp_path / "invalid_pixels.zip"

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])
    writer.writerow([0] + [256] + [128] * 783)

    with ZipFile(zip_path, "w") as archive:
        archive.writestr("data.csv", text.getvalue())

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Pixel out of range"):
        list(builder._generate_examples(zip_path, "train"))
