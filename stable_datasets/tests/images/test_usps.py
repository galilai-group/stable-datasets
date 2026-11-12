"""USPS tests using tiny local BZ2 files and the real dataset."""

import bz2

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import USPS


def _write_bz2(path, text):
    with bz2.open(path, "wt", encoding="ascii") as stream:
        stream.write(text)
    return path


def _dense_row(label, values):
    return str(label) + " " + " ".join(f"{index}:{value}" for index, value in enumerate(values, start=1)) + "\n"


@pytest.fixture
def archives(tmp_path):
    train = _write_bz2(
        tmp_path / "usps.bz2",
        _dense_row(1, [-1, 0, 1, 0.5] + [-1] * 252) + _dense_row(10, [1] * 256),
    )
    test = _write_bz2(
        tmp_path / "usps.t.bz2",
        _dense_row(5, [0] * 256),
    )
    return train, test


def _builder():
    # Match the existing tests: bypass BaseDatasetBuilder.__new__ loading.
    builder = object.__new__(USPS)
    USPS.__init__(builder)
    return builder


def test_dense_samples(archives):
    rows = list(_builder()._generate_examples(archives[0], "train"))
    assert [key for key, _ in rows] == [0, 1]
    assert [sample["label"] for _, sample in rows] == [0, 9]

    expected = np.zeros((16, 16), dtype=np.uint8)
    expected[0, :4] = [0, 127, 255, 191]
    np.testing.assert_array_equal(rows[0][1]["image"], expected)
    np.testing.assert_array_equal(rows[1][1]["image"], np.full((16, 16), 255, dtype=np.uint8))
    for _, sample in rows:
        assert set(sample) == {"image", "label"}
        assert sample["image"].dtype == np.uint8
        assert isinstance(sample["label"], int)


def test_sparse_and_unordered_features(tmp_path):
    path = _write_bz2(tmp_path / "sparse.bz2", "2 256:1 17:-1 1:0.5\n")
    rows = list(_builder()._generate_examples(path, "train"))
    assert len(rows) == 1
    assert rows[0][1]["label"] == 1
    # Omitted LIBSVM features are zero before scaling (127 after scaling).
    expected = np.full((16, 16), 127, dtype=np.uint8)
    expected[0, 0] = 191
    expected[1, 0] = 0
    expected[15, 15] = 255
    np.testing.assert_array_equal(rows[0][1]["image"], expected)


@pytest.mark.parametrize("source_label", range(1, 11))
def test_label_mapping(tmp_path, source_label):
    path = _write_bz2(tmp_path / "labels.bz2", f"{source_label} 1:-1\n")
    rows = list(_builder()._generate_examples(path, "train"))
    assert rows[0][1]["label"] == source_label - 1


@pytest.mark.parametrize(
    "bad_row",
    [
        "0 1:0",
        "11 1:0",
        "x 1:0",
        "1 0:0",
        "1 257:0",
        "1 1:0 1:1",
        "1 1:nan",
        "1 1:inf",
        "1 1:1.01",
        "1 1:-1.01",
        "1 1",
        "1 x:0",
        "",
    ],
)
def test_invalid_samples(tmp_path, bad_row):
    # Put the error on line 2 to verify the diagnostic identifies its location.
    path = _write_bz2(tmp_path / "bad.bz2", "1 1:0\n" + bad_row + "\n")
    with pytest.raises(ValueError, match="Invalid USPS sample at line 2"):
        list(_builder()._generate_examples(path, "train"))


def test_loading_and_cache_reuse(archives, tmp_path, monkeypatch):
    calls = []

    def fake_download(specs, dest_folder):
        calls.append(dest_folder)
        assert len(specs) == 2
        return list(archives)

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_download)
    kwargs = {
        "download_dir": tmp_path / "raw",
        "processed_cache_dir": tmp_path / "processed",
    }
    datasets = USPS(**kwargs)
    assert set(datasets.keys()) == {"train", "test"}
    assert len(datasets["train"]) == 2
    assert len(datasets["test"]) == 1
    assert calls == [kwargs["download_dir"]]

    for split, expected_labels in [("train", [0, 9]), ("test", [4])]:
        dataset = datasets[split]
        assert [dataset[i]["label"] for i in range(len(dataset))] == expected_labels
        for sample in dataset:
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], Image.Image)
            assert sample["image"].size == (16, 16)
            assert sample["image"].mode == "L"
            assert isinstance(sample["label"], int)

    expected = np.zeros((16, 16), dtype=np.uint8)
    expected[0, :4] = [0, 127, 255, 191]
    np.testing.assert_array_equal(np.asarray(datasets["train"][0]["image"]), expected)

    def no_download(*args, **kwargs):
        pytest.fail("Loading a fully cached dataset must not download again")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    reopened = USPS(split="test", **kwargs)
    assert len(reopened) == 1
    assert reopened[0]["label"] == 4
    np.testing.assert_array_equal(np.asarray(reopened[0]["image"]), np.full((16, 16), 127, dtype=np.uint8))
    with pytest.raises(ValueError, match="not found"):
        USPS(split="validation", **kwargs)


def test_usps_metadata():
    builder = _builder()
    info = builder.info
    assert str(USPS.VERSION) == "1.0.0"
    assert info.description
    assert info.homepage == USPS.SOURCE["homepage"]
    assert info.citation == USPS.SOURCE["citation"]
    assert info.license == USPS.SOURCE["license"] == "Unknown"
    assert info.supervised_keys == ("image", "label")
    assert set(info.features) == {"image", "label"}
    assert info.features["label"].names == [str(i) for i in range(10)]
    assert builder._candidate_splits() == ["train", "test"]


def test_usps_real_data():
    datasets = USPS()
    for split, expected_count in [("train", 7291), ("test", 2007)]:
        dataset = datasets[split]
        assert len(dataset) == expected_count
        labels = set()
        for index, sample in enumerate(dataset):
            context = f"{split}[{index}]"
            assert set(sample) == {"image", "label"}, context
            assert isinstance(sample["image"], Image.Image), context
            assert sample["image"].mode == "L", context
            image = np.asarray(sample["image"])
            assert image.shape == (16, 16), context
            assert image.dtype == np.uint8, context
            label = sample["label"]
            assert isinstance(label, int) and 0 <= label < 10, context
            labels.add(label)
        assert labels == set(range(10)), split
