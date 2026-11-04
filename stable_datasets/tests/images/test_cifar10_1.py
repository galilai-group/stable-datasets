from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets.images import CIFAR10_1
from stable_datasets.images import cifar10_1 as cifar10_1_module
from stable_datasets.splits import Split


def test_cifar10_1_public_release(tmp_path: Path):
    datasets = CIFAR10_1(
        split=None,
        download_dir=tmp_path / "downloads",
        processed_cache_dir=tmp_path / "processed",
    )
    assert set(datasets) == {Split.TEST}
    dataset = datasets[Split.TEST]
    assert len(dataset) == 2000
    assert dataset.info.features["label"].names == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    counts = np.zeros(10, dtype=np.int64)
    for sample in dataset:
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], PILImage.Image)
        assert sample["image"].mode == "RGB"
        pixels = np.asarray(sample["image"])
        assert pixels.shape == (32, 32, 3)
        assert pixels.dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 10
        counts[sample["label"]] += 1
    np.testing.assert_array_equal(counts, [200] * 10)


def test_cifar10_1_reopens_cache_without_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images = np.arange(3 * 32 * 32 * 3, dtype=np.uint8).reshape(3, 32, 32, 3)
    labels = np.array([0, 4, 9], dtype=np.int64)
    images_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(images_path, images)
    np.save(labels_path, labels)

    monkeypatch.setattr(cifar10_1_module, "bulk_download", lambda *args, **kwargs: [images_path, labels_path])
    cache_args = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    dataset = CIFAR10_1(split="test", **cache_args)

    def fail_download(*args, **kwargs):
        pytest.fail("Reopening the processed cache must not download again")

    monkeypatch.setattr(cifar10_1_module, "bulk_download", fail_download)
    reopened = CIFAR10_1(split="test", **cache_args)
    assert len(dataset) == len(reopened) == 3
    for index, label in enumerate(labels):
        for cached in (dataset, reopened):
            np.testing.assert_array_equal(np.asarray(cached[index]["image"]), images[index])
            assert cached[index]["label"] == int(label)
    with pytest.raises(ValueError, match="Split 'train' not found"):
        CIFAR10_1(split="train", **cache_args)


@pytest.mark.parametrize(
    "images, labels, message",
    [
        (np.zeros((2, 32, 32), dtype=np.uint8), np.array([0, 1]), "shape"),
        (np.zeros((2, 31, 32, 3), dtype=np.uint8), np.array([0, 1]), "shape"),
        (np.zeros((2, 32, 32, 3), dtype=np.float32), np.array([0, 1]), "uint8"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([[0], [1]]), "one-dimensional"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0]), "matching the image count"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0.0, 1.0]), "integers"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([-1, 1]), "labels must be in"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0, 10]), "labels must be in"),
    ],
    ids=[
        "image-rank",
        "image-shape",
        "pixel-dtype",
        "label-rank",
        "label-count",
        "label-dtype",
        "negative-label",
        "label-10",
    ],
)
def test_cifar10_1_rejects_invalid_arrays(tmp_path: Path, images: np.ndarray, labels: np.ndarray, message: str):
    images_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(images_path, images)
    np.save(labels_path, labels)

    instance = object.__new__(CIFAR10_1)
    with pytest.raises(ValueError, match=message):
        list(instance._generate_examples(images_path, labels_path))
