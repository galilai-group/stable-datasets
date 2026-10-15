from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets import utils
from stable_datasets.images import CIFAR10_1, CIFAR10_2
from stable_datasets.images import cifar10_1 as cifar10_1_module
from stable_datasets.splits import Split


def _images(count: int) -> np.ndarray:
    """Return deterministic RGB images that exercise the image encoding path."""
    return np.arange(count * 32 * 32 * 3, dtype=np.uint8).reshape(count, 32, 32, 3)


def _assert_dataset_contract(dataset, expected_count: int) -> None:
    assert len(dataset) == expected_count
    sample = dataset[0]
    assert set(sample) == {"image", "label"}
    assert isinstance(sample["image"], PILImage.Image)
    assert sample["image"].size == (32, 32)
    assert sample["label"] == 0


def test_cifar10_1_builds_and_reopens_test_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images_path = tmp_path / "cifar10.1_v6_data.npy"
    labels_path = tmp_path / "cifar10.1_v6_labels.npy"
    np.save(images_path, _images(3))
    np.save(labels_path, np.array([0, 4, 9], dtype=np.int64))

    calls = []

    def fake_download(urls, dest_folder):
        calls.append((list(urls), Path(dest_folder)))
        return [images_path, labels_path]

    monkeypatch.setattr(cifar10_1_module, "bulk_download", fake_download)
    processed_dir = tmp_path / "processed"
    download_dir = tmp_path / "downloads"

    dataset = CIFAR10_1(split="test", processed_cache_dir=processed_dir, download_dir=download_dir)
    _assert_dataset_contract(dataset, expected_count=3)
    assert len(calls) == 1

    reopened = CIFAR10_1(split="test", processed_cache_dir=processed_dir, download_dir=download_dir)
    _assert_dataset_contract(reopened, expected_count=3)
    assert len(calls) == 1


@pytest.mark.parametrize("split, count", [("train", 4), ("test", 2)])
def test_cifar10_2_builds_all_splits_and_reopens_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, count: int
):
    train_path = tmp_path / "cifar102_train.npz"
    test_path = tmp_path / "cifar102_test.npz"
    np.savez(train_path, images=_images(4), labels=np.array([0, 1, 2, 3], dtype=np.int64))
    np.savez(test_path, images=_images(2), labels=np.array([0, 9], dtype=np.int64))

    calls = []

    def fake_download(urls, dest_folder):
        calls.append((list(urls), Path(dest_folder)))
        return [train_path, test_path]

    monkeypatch.setattr(utils, "bulk_download", fake_download)
    processed_dir = tmp_path / "processed"
    dataset = CIFAR10_2(split=split, processed_cache_dir=processed_dir, download_dir=tmp_path / "downloads")
    _assert_dataset_contract(dataset, expected_count=count)
    assert len(calls) == 1

    all_splits = CIFAR10_2(split=None, processed_cache_dir=processed_dir, download_dir=tmp_path / "downloads")
    assert set(all_splits) == {Split.TRAIN, Split.TEST}
    _assert_dataset_contract(all_splits[Split.TRAIN], expected_count=4)
    _assert_dataset_contract(all_splits[Split.TEST], expected_count=2)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "builder, images, labels, message",
    [
        (CIFAR10_1, np.zeros((2, 31, 32, 3), dtype=np.uint8), np.array([0, 1]), "shape"),
        (CIFAR10_1, _images(2), np.array([0, 10]), "labels must be in"),
    ],
)
def test_cifar10_1_rejects_invalid_arrays(
    tmp_path: Path, builder, images: np.ndarray, labels: np.ndarray, message: str
):
    images_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(images_path, images)
    np.save(labels_path, labels)

    instance = object.__new__(builder)
    with pytest.raises(ValueError, match=message):
        list(instance._generate_examples(images_path, labels_path))


def test_cifar10_2_rejects_missing_required_arrays(tmp_path: Path):
    path = tmp_path / "invalid.npz"
    np.savez(path, pixels=_images(1))

    instance = object.__new__(CIFAR10_2)
    with pytest.raises(ValueError, match="must contain 'images' and 'labels'"):
        list(instance._generate_examples(path, "test"))
