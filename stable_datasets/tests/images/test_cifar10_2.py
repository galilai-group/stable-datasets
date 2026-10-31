from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets import utils
from stable_datasets.images import CIFAR10_2
from stable_datasets.splits import Split


def test_cifar10_2_public_release(tmp_path: Path):
    datasets = CIFAR10_2(
        split=None,
        download_dir=tmp_path / "downloads",
        processed_cache_dir=tmp_path / "processed",
    )
    assert set(datasets) == {Split.TRAIN, Split.TEST}
    for split, size, per_class in [(Split.TRAIN, 10000, 1000), (Split.TEST, 2000, 200)]:
        dataset = datasets[split]
        assert len(dataset) == size
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
        np.testing.assert_array_equal(counts, [per_class] * 10)


@pytest.mark.parametrize("split", ["train", "test"])
def test_cifar10_2_reopens_cache_without_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str):
    train_images = np.arange(4 * 32 * 32 * 3, dtype=np.uint8).reshape(4, 32, 32, 3)
    test_images = np.full((2, 32, 32, 3), 42, dtype=np.uint8)
    train_labels = np.array([0, 1, 2, 3], dtype=np.int64)
    test_labels = np.array([0, 9], dtype=np.int64)
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    np.savez(train_path, images=train_images, labels=train_labels)
    np.savez(test_path, images=test_images, labels=test_labels)

    monkeypatch.setattr(utils, "bulk_download", lambda *args, **kwargs: [train_path, test_path])
    cache_args = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    dataset = CIFAR10_2(split=split, **cache_args)

    def fail_download(*args, **kwargs):
        pytest.fail("Reopening the processed cache must not download again")

    monkeypatch.setattr(utils, "bulk_download", fail_download)
    reopened = CIFAR10_2(split=None, **cache_args)
    assert set(reopened) == {Split.TRAIN, Split.TEST}
    assert len(dataset) == (4 if split == "train" else 2)
    for name, images, labels in [("train", train_images, train_labels), ("test", test_images, test_labels)]:
        assert len(reopened[name]) == len(labels)
        for index, label in enumerate(labels):
            np.testing.assert_array_equal(np.asarray(reopened[name][index]["image"]), images[index])
            assert reopened[name][index]["label"] == int(label)


@pytest.mark.parametrize("missing", ["images", "labels"])
def test_cifar10_2_rejects_missing_required_arrays(tmp_path: Path, missing: str):
    arrays = {"images": np.zeros((1, 32, 32, 3), dtype=np.uint8), "labels": np.array([0])}
    del arrays[missing]
    path = tmp_path / "invalid.npz"
    np.savez(path, **arrays)

    instance = object.__new__(CIFAR10_2)
    with pytest.raises(ValueError, match="must contain 'images' and 'labels'"):
        list(instance._generate_examples(path, "test"))


@pytest.mark.parametrize(
    "images, labels, message",
    [
        (np.zeros((2, 32, 32, 3), dtype=np.float32), np.array([0, 1]), "uint8"),
        (np.zeros((2, 32, 32, 3), dtype=np.uint8), np.array([0, 10]), "labels must be in"),
    ],
    ids=["pixel-dtype", "label-range"],
)
def test_cifar10_2_rejects_invalid_arrays(tmp_path: Path, images: np.ndarray, labels: np.ndarray, message: str):
    path = tmp_path / "invalid.npz"
    np.savez(path, images=images, labels=labels)

    instance = object.__new__(CIFAR10_2)
    with pytest.raises(ValueError, match=message):
        list(instance._generate_examples(path, "test"))
