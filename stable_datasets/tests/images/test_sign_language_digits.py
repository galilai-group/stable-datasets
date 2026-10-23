import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.features import ClassLabel, Value
from stable_datasets.images import SignLanguageDigits
from stable_datasets.schema import Image as ImageFeature


def _image_bytes(mode, color, image_format):
    buffer = io.BytesIO()
    Image.new(mode, (8, 6), color).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def sign_digits_archive(tmp_path):
    archive_path = tmp_path / "sign-language-digits.zip"
    root = "Sign-Language-Digits-Dataset-fixture"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root}/Dataset/9/z.JPEG", _image_bytes("RGB", (90, 91, 92), "JPEG"))
        archive.writestr(f"{root}/Dataset/0/a.JPG", _image_bytes("L", 10, "JPEG"))
        archive.writestr(f"{root}/Dataset/1/b.PnG", _image_bytes("RGBA", (20, 21, 22, 23), "PNG"))

        # These are not samples: previews, nested files, unknown classes, and non-images.
        archive.writestr(f"{root}/Examples/example_0.JPG", b"preview files must never be decoded")
        archive.writestr(f"{root}/Dataset/2/nested/ignored.JPG", _image_bytes("RGB", (2, 3, 4), "JPEG"))
        archive.writestr(f"{root}/Dataset/10/ignored.JPG", _image_bytes("RGB", (3, 4, 5), "JPEG"))
        archive.writestr(f"{root}/Dataset/3/README.txt", "not an image")

    return archive_path


def _patch_download(monkeypatch, archive_path):
    calls = []

    def fake_bulk_download(urls, dest_folder, checksums=None):
        calls.append((list(urls), dest_folder, checksums))
        return [archive_path]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    return calls


def test_sign_language_digits_fixture_and_metadata(tmp_path, monkeypatch, sign_digits_archive):
    calls = _patch_download(monkeypatch, sign_digits_archive)
    dataset = SignLanguageDigits(
        split="all",
        processed_cache_dir=tmp_path / "processed",
        download_dir=tmp_path / "downloads",
    )

    assert len(calls) == 1
    assert len(dataset) == 3
    assert dataset.info.supervised_keys == ("image", "label")
    assert dataset.info.homepage == "https://github.com/ardamavi/Sign-Language-Digits-Dataset"
    assert dataset.info.license == "Apache-2.0"
    assert isinstance(dataset.info.features["image"], ImageFeature)
    assert isinstance(dataset.info.features["label"], ClassLabel)
    assert dataset.info.features["label"].names == [str(digit) for digit in range(10)]
    assert isinstance(dataset.info.features["filename"], Value)
    assert dataset.info.features["filename"].dtype == "string"

    samples = [dataset[index] for index in range(len(dataset))]
    assert [sample["filename"] for sample in samples] == ["0/a.JPG", "1/b.PnG", "9/z.JPEG"]
    assert [sample["label"] for sample in samples] == [0, 1, 9]
    assert all(set(sample) == {"image", "label", "filename"} for sample in samples)
    assert all(isinstance(sample["image"], Image.Image) for sample in samples)
    assert all(np.asarray(sample["image"]).shape == (6, 8, 3) for sample in samples)
    assert all(np.asarray(sample["image"]).dtype == np.uint8 for sample in samples)


def test_sign_language_digits_uses_processed_cache(tmp_path, monkeypatch, sign_digits_archive):
    calls = _patch_download(monkeypatch, sign_digits_archive)
    kwargs = {
        "split": "all",
        "processed_cache_dir": tmp_path / "processed",
        "download_dir": tmp_path / "downloads",
    }

    first = SignLanguageDigits(**kwargs)
    second = SignLanguageDigits(**kwargs)

    assert len(first) == len(second) == 3
    assert second[0]["filename"] == first[0]["filename"]
    assert len(calls) == 1


def test_sign_language_digits_uses_filename_as_stable_key(sign_digits_archive):
    builder = object.__new__(SignLanguageDigits)
    builder.__init__()

    examples = list(builder._generate_examples(sign_digits_archive, "all"))

    assert [key for key, _ in examples] == ["0/a.JPG", "1/b.PnG", "9/z.JPEG"]
    assert all(key == example["filename"] for key, example in examples)


def test_sign_language_digits_source_is_revision_pinned():
    source = SignLanguageDigits.SOURCE
    asset = source["assets"]["all"]

    assert SignLanguageDigits.SOURCE_REVISION in asset.url
    assert asset.url.endswith(f"{SignLanguageDigits.SOURCE_REVISION}.zip")
    assert asset.filename.endswith(".zip")
    assert asset.checksum == f"sha256:{SignLanguageDigits.SOURCE_ARCHIVE_SHA256}"


def test_sign_language_digits_rejects_unknown_split(sign_digits_archive):
    builder = object.__new__(SignLanguageDigits)
    builder.__init__()

    with pytest.raises(ValueError, match="only provides the 'all' split"):
        list(builder._generate_examples(sign_digits_archive, "train"))


def test_sign_language_digits_rejects_archive_without_dataset_images(tmp_path):
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repository/Examples/example_0.JPG", _image_bytes("RGB", (1, 2, 3), "JPEG"))

    builder = object.__new__(SignLanguageDigits)
    builder.__init__()
    with pytest.raises(ValueError, match=r"No images found directly under Dataset/<digit>/"):
        list(builder._generate_examples(archive_path, "all"))


def test_sign_language_digits_reports_corrupt_images(tmp_path):
    archive_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repository/Dataset/4/broken.JPG", b"not a JPEG")

    builder = object.__new__(SignLanguageDigits)
    builder.__init__()
    with pytest.raises(ValueError, match=r"Failed to decode image .*broken\.JPG"):
        list(builder._generate_examples(archive_path, "all"))


def test_sign_language_digits_rejects_duplicate_logical_filenames(tmp_path):
    archive_path = tmp_path / "duplicates.zip"
    image = _image_bytes("RGB", (1, 2, 3), "JPEG")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("root-a/Dataset/4/same.JPG", image)
        archive.writestr("root-b/Dataset/4/same.JPG", image)

    builder = object.__new__(SignLanguageDigits)
    builder.__init__()
    with pytest.raises(ValueError, match="Duplicate dataset filename"):
        list(builder._generate_examples(archive_path, "all"))


def test_sign_language_digits_rejects_invalid_zip(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_bytes(b"not a ZIP archive")

    builder = object.__new__(SignLanguageDigits)
    builder.__init__()
    with pytest.raises(ValueError, match="Invalid Sign Language Digits ZIP archive"):
        list(builder._generate_examples(archive_path, "all"))
