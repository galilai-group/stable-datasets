import gzip
import struct

import numpy as np
import pytest

from stable_datasets.images import KannadaMNIST


def _idx_pair(tmp_path, images=None, labels=None):
    if images is None:
        images = struct.pack(">IIII", 2051, 2, 28, 28) + bytes([17]) * 784 + bytes([219]) * 784
    if labels is None:
        labels = struct.pack(">II", 2049, 2) + bytes([3, 9])
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = [tmp_path / "images.gz", tmp_path / "labels.gz"]
    for path, data in zip(paths, [images, labels]):
        with gzip.open(path, "wb") as stream:
            stream.write(data)
    return paths


def test_idx_pixels_and_labels(tmp_path):
    images, labels = _idx_pair(tmp_path)
    builder = object.__new__(KannadaMNIST)
    rows = list(builder._generate_examples(images, labels, "train"))
    assert [key for key, _ in rows] == [0, 1]
    assert [row["label"] for _, row in rows] == [3, 9]
    assert np.all(rows[0][1]["image"] == 17)
    assert np.all(rows[1][1]["image"] == 219)
    assert rows[0][1]["image"].dtype == np.uint8
    assert rows[0][1]["image"].shape == (28, 28)


@pytest.mark.parametrize(
    ("images", "labels", "message"),
    [
        (b"bad", None, "header"),
        (struct.pack(">IIII", 2051, 0, 28, 28), struct.pack(">II", 2049, 0), "nonzero"),
        (None, b"bad", "header"),
        (struct.pack(">IIII", 0, 2, 28, 28) + bytes(1568), None, "magic"),
        (struct.pack(">IIII", 2051, 2, 27, 28) + bytes(1512), None, "dimensions"),
        (None, struct.pack(">II", 0, 2) + bytes(2), "magic"),
        (None, struct.pack(">II", 2049, 1) + bytes(1), "counts"),
        (struct.pack(">IIII", 2051, 2, 28, 28) + bytes(1567), None, "length"),
        (None, struct.pack(">II", 2049, 2) + bytes(3), "length"),
        (None, struct.pack(">II", 2049, 2) + bytes([0, 10]), "digits"),
    ],
)
def test_invalid_idx(tmp_path, images, labels, message):
    paths = _idx_pair(tmp_path, images, labels)
    with pytest.raises(ValueError, match=message):
        list(object.__new__(KannadaMNIST)._generate_examples(*paths, "train"))


def test_cache_roundtrip(tmp_path, monkeypatch):
    images, labels = _idx_pair(tmp_path / "train")
    test_images, test_labels = _idx_pair(
        tmp_path / "test",
        images=struct.pack(">IIII", 2051, 2, 28, 28) + bytes([41]) * 784 + bytes([72]) * 784,
        labels=struct.pack(">II", 2049, 2) + bytes([1, 2]),
    )
    calls = []

    def download(specs, dest_folder):
        calls.append(list(specs))
        assert dest_folder == tmp_path / "raw"
        return [images, labels, test_images, test_labels]

    monkeypatch.setattr("stable_datasets.images.kannada_mnist.bulk_download", download)
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "cache"}
    ds = KannadaMNIST(**kwargs)
    assert set(ds) == {"train", "test"}
    assert len(calls) == 1 and len(calls[0]) == 4
    assert len(ds["train"]) == len(ds["test"]) == 2
    assert ds["train"].features["label"].names == [str(i) for i in range(10)]
    assert ds["train"][0]["image"].mode == "L"
    assert ds["train"].with_format("torch")[0]["image"].shape == (1, 28, 28)

    def no_download(*args, **kwargs):
        raise AssertionError("Cache reload must not download")

    monkeypatch.setattr("stable_datasets.images.kannada_mnist.bulk_download", no_download)
    cached = KannadaMNIST(split="test", **kwargs)
    assert [ds["train"][i]["label"] for i in range(2)] == [3, 9]
    assert [ds["test"][i]["label"] for i in range(2)] == [1, 2]
    assert cached[1]["label"] == 2
    assert np.all(np.array(cached[1]["image"]) == 72)


def test_official_splits():
    ds = KannadaMNIST()
    for split, size, per_class in [("train", 60000, 6000), ("test", 10000, 1000)]:
        assert len(ds[split]) == size
        labels = [sample["label"] for sample in ds[split].with_format("raw")]
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, per_class))
        sample = ds[split][0]
        assert sample["image"].size == (28, 28)
        assert sample["image"].mode == "L"


def test_unknown_split_is_rejected(tmp_path):
    paths = _idx_pair(tmp_path)
    with pytest.raises(ValueError, match="Unknown Kannada-MNIST split"):
        list(object.__new__(KannadaMNIST)._generate_examples(*paths, "validation"))
