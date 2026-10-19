import io
import os
import pickle
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from stable_datasets.images import MNISTM


# Offline tests: synthetic tar.gz files with the same layout as the real liyxi/mnist-m release.


class _NotATensor:
    """Arbitrary object that ``torch.load(weights_only=True)`` must refuse to unpickle."""


def _write_fake_release(path, split, payload, legacy=True):
    buf = io.BytesIO()
    # The real release uses torch's legacy (non-zip) serialization.
    torch.save(payload, buf, _use_new_zipfile_serialization=not legacy)
    data = buf.getvalue()
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(f"mnist_m_{split}.pt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def _fake_tensors(n, seed):
    rng = np.random.default_rng(seed)
    images = torch.from_numpy(rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8))
    labels = torch.from_numpy(rng.integers(0, 10, size=(n,), dtype=np.int64))
    return images, labels


def _patch_download(monkeypatch, files_by_name):
    """Serve local files keyed by URL basename instead of downloading; return the requested DownloadInfos."""
    requested = []

    def fake_bulk_download(infos, dest_folder):
        infos = list(infos)
        requested.extend(infos)
        return [files_by_name[os.path.basename(info.url)] for info in infos]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    return requested


def _build(tmp_path, monkeypatch, payloads, legacy=True, split=None):
    files = {}
    for name, payload in payloads.items():
        files[f"mnist_m_{name}.pt.tar.gz"] = tmp_path / f"mnist_m_{name}.pt.tar.gz"
        _write_fake_release(files[f"mnist_m_{name}.pt.tar.gz"], name, payload, legacy=legacy)
    requested = _patch_download(monkeypatch, files)
    ds = MNISTM(split=split, download_dir=tmp_path / "downloads", processed_cache_dir=tmp_path / "processed")
    return ds, requested


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy-format", "zip-format"])
def test_fake_round_trip(tmp_path, monkeypatch, legacy):
    payloads = {"train": _fake_tensors(12, seed=0), "test": _fake_tensors(5, seed=1)}
    ds, _ = _build(tmp_path, monkeypatch, payloads, legacy=legacy)

    for split, (images, labels) in payloads.items():
        assert len(ds[split]) == len(labels)
        for i in range(len(labels)):
            sample = ds[split][i]
            assert set(sample) == {"image", "label"}
            assert sample["image"].mode == "RGB" and sample["image"].size == (28, 28)
            np.testing.assert_array_equal(np.asarray(sample["image"]), images[i].numpy())
            assert sample["label"] == int(labels[i])


def test_fake_requests_both_release_files_with_checksums(tmp_path, monkeypatch):
    payloads = {"train": _fake_tensors(2, seed=0), "test": _fake_tensors(2, seed=1)}
    _, requested = _build(tmp_path, monkeypatch, payloads)

    by_name = {os.path.basename(info.url): info for info in requested}
    assert set(by_name) == {"mnist_m_train.pt.tar.gz", "mnist_m_test.pt.tar.gz"}
    assert all(info.url.startswith("https://github.com/liyxi/mnist-m/releases/download/data/") for info in requested)
    assert by_name["mnist_m_train.pt.tar.gz"].checksum == (
        "sha256:69c983f2d2f9b0280efc691f8252576b18bf6e5dee5279dbf43109a023d151de"
    )
    assert by_name["mnist_m_test.pt.tar.gz"].checksum == (
        "sha256:85958398c49815ac2f38aa309eb6d7d3d54c5bf9e0e3946cb00259b58a8b7cf7"
    )


def test_fake_torch_format(tmp_path, monkeypatch):
    payloads = {"train": _fake_tensors(3, seed=0), "test": _fake_tensors(3, seed=1)}
    ds, _ = _build(tmp_path, monkeypatch, payloads, split="test")
    sample = ds.with_format("torch")[0]
    assert tuple(sample["image"].shape) == (3, 28, 28)
    assert int(sample["label"]) == int(payloads["test"][1][0])


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((4, 28, 28), dtype=torch.uint8),
        torch.zeros((4, 3, 28, 28), dtype=torch.uint8),
        torch.zeros((4, 28, 28, 3), dtype=torch.float32),
    ],
    ids=["grayscale", "channels-first", "float"],
)
def test_fake_rejects_unexpected_image_tensors(tmp_path, images):
    path = tmp_path / "mnist_m_test.pt.tar.gz"
    _write_fake_release(path, "test", (images, torch.zeros(4, dtype=torch.int64)))
    with pytest.raises(ValueError, match="Expected uint8 images"):
        MNISTM._load_tensors(path, "test")


def test_fake_rejects_mismatched_labels(tmp_path):
    path = tmp_path / "mnist_m_test.pt.tar.gz"
    images, _ = _fake_tensors(4, seed=0)
    _write_fake_release(path, "test", (images, torch.zeros(3, dtype=torch.int64)))
    with pytest.raises(ValueError, match="Expected 4 labels"):
        MNISTM._load_tensors(path, "test")


def test_fake_refuses_arbitrary_pickled_objects(tmp_path):
    path = tmp_path / "mnist_m_test.pt.tar.gz"
    images, labels = _fake_tensors(2, seed=0)
    _write_fake_release(path, "test", (images, labels, _NotATensor()))
    with pytest.raises(pickle.UnpicklingError):
        MNISTM._load_tensors(path, "test")


def test_metadata():
    builder = object.__new__(MNISTM)
    builder.__init__()
    assert builder.info.features["label"].names == [str(i) for i in range(10)]
    assert builder.info.supervised_keys == ("image", "label")
    assert set(MNISTM.SOURCE["assets"]) == {"train", "test"}


# Real download (network, ~130 MB).


@pytest.mark.large
def test_mnist_m_dataset():
    mnistm_train = MNISTM(split="train")

    # Test 1: Check number of training samples
    expected_num_train_samples = 60000
    assert len(mnistm_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(mnistm_train)}."
    )

    # Test 2: Check sample keys
    sample = mnistm_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type, shape and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (28, 28, 3), f"MNIST-M images should be 28x28x3, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 10, f"Label should be in range [0, 9], got {label}."
    assert len(mnistm_train.features["label"].names) == 10

    # Test 5: Load and validate test split
    mnistm_test = MNISTM(split="test")
    expected_num_test_samples = 10000
    assert len(mnistm_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(mnistm_test)}."
    )

    # Test 6: All 10 digits appear in both splits
    for ds in (mnistm_train, mnistm_test):
        assert set(ds.table.column("label").to_pylist()) == set(range(10))

    # Test 7: Torch formatting yields CHW tensors
    torch_image = mnistm_train.with_format("torch")[0]["image"]
    assert tuple(torch_image.shape) == (3, 28, 28), f"Expected (3, 28, 28), got {tuple(torch_image.shape)}."
