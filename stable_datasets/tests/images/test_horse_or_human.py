"""Horses vs. Humans tests using local ZIP archives and the real dataset."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import HorseOrHuman


def _make_image(mode="RGB", color=None):
    """Create a 300x300 PNG image in memory."""
    if color is None:
        color = {
            "RGB": (255, 0, 0),
            "RGBA": (255, 0, 0, 128),
            "L": 128,
        }[mode]

    image = Image.new(mode, (300, 300), color=color)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


@pytest.fixture
def archive(tmp_path):
    """Create a small local ZIP archive with both classes."""
    path = tmp_path / "horse_or_human.zip"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr(
            "horses/horse01.png",
            _make_image("RGB", (255, 0, 0)),
        )
        zip_file.writestr(
            "horses/horse02.png",
            _make_image("RGB", (0, 255, 0)),
        )
        zip_file.writestr(
            "humans/human01.png",
            _make_image("RGB", (0, 0, 255)),
        )

    return path


def _builder():
    """Bypass BaseDatasetBuilder.__new__ to test the generator."""
    return object.__new__(HorseOrHuman)


def _load_rows(path, split="train"):
    return list(_builder()._generate_examples(path, split=split))


def test_local_zip_loading(archive):
    rows = _load_rows(archive)

    assert len(rows) == 3

    keys = [key for key, _ in rows]
    assert len(keys) == len(set(keys))

    labels = [sample["label"] for _, sample in rows]
    assert labels == [0, 0, 1]

    for _, sample in rows:
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].size == (300, 300)
        assert sample["image"].mode == "RGB"
        assert isinstance(sample["label"], int)


def test_pixel_values(archive):
    rows = _load_rows(archive)

    expected_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    for (_, sample), expected in zip(rows, expected_colors):
        assert sample["image"].getpixel((0, 0)) == expected


@pytest.mark.parametrize("split", ["train", "test"])
def test_valid_splits(archive, split):
    rows = _load_rows(archive, split)

    assert len(rows) == 3
    assert {sample["label"] for _, sample in rows} == {0, 1}


@pytest.mark.parametrize(
    "split",
    ["validation", "invalid", "", None],
)
def test_invalid_splits(archive, split):
    with pytest.raises(ValueError, match="Unknown split"):
        _load_rows(archive, split)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_rgb_conversion(tmp_path, mode):
    path = tmp_path / "images.zip"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr(
            "horses/sample.png",
            _make_image(mode),
        )

    rows = _load_rows(path)

    assert len(rows) == 1

    image = rows[0][1]["image"]
    assert image.mode == "RGB"
    assert image.size == (300, 300)


def test_ignore_non_image_files(tmp_path):
    path = tmp_path / "images.zip"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("README.txt", "Example dataset")
        zip_file.writestr("horses/notes.txt", "Not an image")

        zip_file.writestr(
            "horses/horse.png",
            _make_image(),
        )
        zip_file.writestr(
            "humans/human.png",
            _make_image(),
        )

    rows = _load_rows(path)

    assert len(rows) == 2
    assert {sample["label"] for _, sample in rows} == {0, 1}


def test_ignore_unknown_classes(tmp_path):
    path = tmp_path / "images.zip"

    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr(
            "horses/horse.png",
            _make_image(),
        )
        zip_file.writestr(
            "cats/cat.png",
            _make_image(),
        )

    rows = _load_rows(path)

    assert len(rows) == 1
    assert rows[0][1]["label"] == 0


def test_deterministic_order(archive):
    first = _load_rows(archive)
    second = _load_rows(archive)

    assert [key for key, _ in first] == [key for key, _ in second]

    assert [sample["label"] for _, sample in first] == [sample["label"] for _, sample in second]


def test_metadata():
    info = HorseOrHuman._info(_builder())

    assert str(HorseOrHuman.VERSION) == "1.0.0"
    assert info.description

    assert info.homepage == HorseOrHuman.SOURCE["homepage"]
    assert info.citation == HorseOrHuman.SOURCE["citation"]
    assert info.license == HorseOrHuman.SOURCE["license"]

    assert info.supervised_keys == ("image", "label")
    assert set(info.features) == {"image", "label"}
    assert info.features["label"].names == ["horses", "humans"]


def test_real_data():
    """Load and validate all images from the real dataset."""
    datasets = HorseOrHuman()

    assert set(datasets.keys()) == {"train", "test"}

    expected_counts = {
        "train": 1027,
        "test": 256,
    }

    for split, expected_count in expected_counts.items():
        dataset = datasets[split]

        assert len(dataset) == expected_count

        labels = set()

        for index, sample in enumerate(dataset):
            context = f"{split}[{index}]"

            assert set(sample) == {"image", "label"}, context

            image = sample["image"]

            assert isinstance(image, Image.Image), context
            assert image.mode == "RGB", context
            assert image.size == (300, 300), context

            array = np.asarray(image)

            assert array.shape == (300, 300, 3), context
            assert array.dtype == np.uint8, context

            label = sample["label"]

            assert isinstance(label, int), context
            assert label in (0, 1), context

            labels.add(label)

        assert labels == {0, 1}
