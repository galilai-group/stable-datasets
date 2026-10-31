"""Semeion tests using synthetic local files and the real dataset."""

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Semeion


def _make_row(index):
    """Create one Semeion sample with 256 pixels and 10 one-hot labels."""
    pixels = np.zeros(256, dtype=np.int32)

    # Encode the row index in binary to create a unique image.
    for bit in range(11):
        pixels[bit] = (index >> bit) & 1

    label = index % 10
    labels = np.zeros(10, dtype=np.int32)
    labels[label] = 1

    return " ".join(map(str, np.concatenate([pixels, labels])))


@pytest.fixture
def data_file(tmp_path):
    """Create a synthetic dataset matching Semeion's expected shape."""
    path = tmp_path / "semeion.data"

    rows = [_make_row(index) for index in range(1593)]
    path.write_text("\n".join(rows) + "\n", encoding="ascii")

    return path


def _builder():
    """Bypass BaseDatasetBuilder.__new__ to test the generator directly."""
    return object.__new__(Semeion)


def _load_rows(path, split):
    return list(_builder()._generate_examples(path, split=split))


def test_local_split_counts(data_file):
    train = _load_rows(data_file, "train")
    test = _load_rows(data_file, "test")

    assert len(train) == 1274
    assert len(test) == 319
    assert len(train) + len(test) == 1593


@pytest.mark.parametrize("split", ["train", "test"])
def test_local_sample_format(data_file, split):
    rows = _load_rows(data_file, split)

    assert len(rows) > 0

    keys = [key for key, _ in rows]
    assert len(keys) == len(set(keys))

    for _, sample in rows:
        assert set(sample) == {"image", "label"}

        image = sample["image"]
        array = np.asarray(image)

        assert array.shape == (16, 16)
        assert np.issubdtype(array.dtype, np.number)

        assert isinstance(sample["label"], (int, np.integer))
        assert 0 <= sample["label"] < 10


def test_one_hot_label_mapping(data_file):
    train = _load_rows(data_file, "train")
    test = _load_rows(data_file, "test")

    labels = [sample["label"] for _, sample in train + test]

    expected_labels = [index % 10 for index in range(1593)]

    assert sorted(labels) == sorted(expected_labels)


def test_pixel_values(data_file):
    train = _load_rows(data_file, "train")
    test = _load_rows(data_file, "test")

    for _, sample in train + test:
        array = np.asarray(sample["image"])

        assert array.shape == (16, 16)

        unique_values = set(np.unique(array).tolist())
        assert unique_values.issubset({0, 1, 255})


def test_reproducible_split(data_file):
    first_train = _load_rows(data_file, "train")
    second_train = _load_rows(data_file, "train")

    first_test = _load_rows(data_file, "test")
    second_test = _load_rows(data_file, "test")

    assert [key for key, _ in first_train] == [key for key, _ in second_train]

    assert [key for key, _ in first_test] == [key for key, _ in second_test]

    for first, second in zip(first_train, second_train):
        np.testing.assert_array_equal(
            np.asarray(first[1]["image"]),
            np.asarray(second[1]["image"]),
        )
        assert first[1]["label"] == second[1]["label"]

    for first, second in zip(first_test, second_test):
        np.testing.assert_array_equal(
            np.asarray(first[1]["image"]),
            np.asarray(second[1]["image"]),
        )
        assert first[1]["label"] == second[1]["label"]


def test_no_split_overlap(data_file):
    train = _load_rows(data_file, "train")
    test = _load_rows(data_file, "test")

    train_images = {np.asarray(sample["image"]).tobytes() for _, sample in train}

    test_images = {np.asarray(sample["image"]).tobytes() for _, sample in test}

    assert train_images.isdisjoint(test_images)


@pytest.mark.parametrize(
    "shape",
    [
        (20, 266),
        (1593, 265),
        (1593, 267),
    ],
)
def test_invalid_data_shape(tmp_path, shape):
    path = tmp_path / "invalid.data"

    data = np.zeros(shape, dtype=np.float32)
    np.savetxt(path, data, fmt="%.0f")

    with pytest.raises(ValueError, match="Unexpected Semeion shape"):
        _load_rows(path, "train")


def test_metadata():
    info = Semeion._info(_builder())

    assert str(Semeion.VERSION) == "1.0.0"
    assert info.description
    assert info.homepage == Semeion.SOURCE["homepage"]
    assert info.citation == Semeion.SOURCE["citation"]

    assert info.supervised_keys == ("image", "label")
    assert set(info.features) == {"image", "label"}

    assert info.features["label"].names == [str(i) for i in range(10)]


def test_real_data():
    """Load and validate the complete real Semeion dataset."""
    datasets = Semeion()

    assert set(datasets.keys()) == {"train", "test"}

    expected_counts = {
        "train": 1274,
        "test": 319,
    }

    all_labels = set()

    for split, expected_count in expected_counts.items():
        dataset = datasets[split]

        assert len(dataset) == expected_count

        for index, sample in enumerate(dataset):
            context = f"{split}[{index}]"

            assert set(sample) == {"image", "label"}, context

            image = sample["image"]
            assert isinstance(image, Image.Image), context
            assert image.size == (16, 16), context
            assert image.mode == "L", context

            array = np.asarray(image)
            assert array.shape == (16, 16), context
            assert array.dtype == np.uint8, context

            label = sample["label"]
            assert isinstance(label, int), context
            assert 0 <= label < 10, context

            all_labels.add(label)

    assert all_labels == set(range(10))
