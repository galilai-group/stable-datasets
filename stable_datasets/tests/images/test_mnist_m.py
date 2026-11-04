import io
import tarfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import MNISTM


# Offline tests: synthetic mnist_m.tar.gz files with the same layout as the original release.


def _png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _fake_split(n, seed):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, 10, size=(n,)).tolist()
    return images, labels


def _add_file(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_dir(tar, name):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    tar.addfile(info)


def _write_fake_release(path, splits, labels_text=None, extra_files=None):
    """Write a fake mnist_m.tar.gz from ``{split: (images, labels)}``.

    ``labels_text`` overrides a split's labels file and ``extra_files`` adds raw ``{name: bytes}`` entries.
    """
    labels_text = labels_text or {}
    with tarfile.open(path, "w:gz") as tar:
        _add_dir(tar, "mnist_m/")
        for split, (images, labels) in splits.items():
            names = [f"{i:08d}.png" for i in range(len(labels))]
            text = labels_text.get(split, "".join(f"{n} {y}\n" for n, y in zip(names, labels)))
            _add_file(tar, f"mnist_m/mnist_m_{split}_labels.txt", text.encode())
            _add_dir(tar, f"mnist_m/mnist_m_{split}/")
            for name, image in zip(names, images):
                _add_file(tar, f"mnist_m/mnist_m_{split}/{name}", _png(image))
        for name, data in (extra_files or {}).items():
            _add_file(tar, name, data)


def _patch_download(monkeypatch, archive_path):
    """Serve ``archive_path`` instead of downloading; return the list of requested DownloadInfos."""
    requested = []

    def fake_bulk_download(infos, dest_folder):
        infos = list(infos)
        requested.extend(infos)
        return [archive_path] * len(infos)

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    return requested


def _build(tmp_path, monkeypatch, splits, split=None, **kwargs):
    archive_path = tmp_path / "mnist_m.tar.gz"
    _write_fake_release(archive_path, splits, **kwargs)
    requested = _patch_download(monkeypatch, archive_path)
    ds = MNISTM(split=split, download_dir=tmp_path / "downloads", processed_cache_dir=tmp_path / "processed")
    return ds, requested


def test_fake_round_trip(tmp_path, monkeypatch):
    splits = {"train": _fake_split(12, seed=0), "test": _fake_split(5, seed=1)}
    # Junk entries that the builder must skip.
    extra = {"mnist_m/mnist_m_train/._00000000.png": b"resource fork", "mnist_m/README": b"not an image"}
    ds, _ = _build(tmp_path, monkeypatch, splits, extra_files=extra)

    for split, (images, labels) in splits.items():
        assert len(ds[split]) == len(labels)
        for i in range(len(labels)):
            sample = ds[split][i]
            assert set(sample) == {"image", "label"}
            assert sample["image"].mode == "RGB" and sample["image"].size == (32, 32)
            np.testing.assert_array_equal(np.asarray(sample["image"]), images[i])
            assert sample["label"] == labels[i]


def test_fake_shared_archive_is_requested_once_with_checksum(tmp_path, monkeypatch):
    splits = {"train": _fake_split(2, seed=0), "test": _fake_split(2, seed=1)}
    _, requested = _build(tmp_path, monkeypatch, splits)

    (info,) = requested
    assert info.filename == "mnist_m.tar.gz"
    assert info.url.startswith("https://drive.usercontent.google.com/download?id=0B_tExHiYS-0veklUZHFYT19KYjg")
    assert info.checksum == "sha256:022d6584383f38fce6f53ca1387dcac9a51986cee5e4d231544546ec103501a6"


def test_fake_torch_format(tmp_path, monkeypatch):
    splits = {"train": _fake_split(3, seed=0), "test": _fake_split(3, seed=1)}
    ds, _ = _build(tmp_path, monkeypatch, splits, split="test")
    sample = ds.with_format("torch")[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert int(sample["label"]) == splits["test"][1][0]


def _read(tmp_path, splits, split="test", **kwargs):
    path = tmp_path / "mnist_m.tar.gz"
    _write_fake_release(path, splits, **kwargs)
    return MNISTM._read_split(path, split)


@pytest.mark.parametrize(
    "labels_text",
    ["00000000.png 0\n00000001.png 1\n", "00000000.png 0\n00000001.png 1\n00000002.png 2\n00000003.png 3\n"],
    ids=["unlabeled-image", "missing-image"],
)
def test_fake_rejects_labels_not_matching_images(tmp_path, labels_text):
    with pytest.raises(ValueError, match="must match one to one"):
        _read(tmp_path, {"test": _fake_split(3, seed=0)}, labels_text={"test": labels_text})


@pytest.mark.parametrize("label", ["10", "-1", "x"])
def test_fake_rejects_invalid_labels(tmp_path, label):
    with pytest.raises(ValueError, match="Invalid label"):
        _read(tmp_path, {"test": _fake_split(1, seed=0)}, labels_text={"test": f"00000000.png {label}\n"})


def test_fake_rejects_missing_labels_file(tmp_path):
    with pytest.raises(ValueError, match="mnist_m_test_labels.txt not found"):
        _read(tmp_path, {"train": _fake_split(1, seed=0)})


@pytest.mark.parametrize(
    "image", [np.zeros((32, 32), dtype=np.uint8), np.zeros((28, 28, 3), dtype=np.uint8)], ids=["grayscale", "28x28"]
)
def test_fake_rejects_unexpected_images(tmp_path, monkeypatch, image):
    splits = {"train": ([image], [0]), "test": ([image], [0])}
    with pytest.raises(ValueError, match="Expected a 32x32 RGB image"):
        _build(tmp_path, monkeypatch, splits)


def test_metadata():
    builder = object.__new__(MNISTM)
    builder.__init__()
    assert builder.info.features["label"].names == [str(i) for i in range(10)]
    assert builder.info.supervised_keys == ("image", "label")

    assets = MNISTM.SOURCE["assets"]
    assert set(assets) == {"train", "test"}
    assert assets["train"] == assets["test"]


# Real download (network, ~134 MB).

EXPECTED_NUM_TRAIN = 59001
EXPECTED_NUM_TEST = 9001


def test_mnist_m_dataset():
    mnistm_train = MNISTM(split="train")

    # Test 1: Check number of training samples
    assert len(mnistm_train) == EXPECTED_NUM_TRAIN, (
        f"Expected {EXPECTED_NUM_TRAIN} training samples, got {len(mnistm_train)}."
    )

    # Test 2: Check sample keys
    sample = mnistm_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type, shape and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.shape == (32, 32, 3), f"MNIST-M images should be 32x32x3, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 10, f"Label should be in range [0, 9], got {label}."
    assert len(mnistm_train.features["label"].names) == 10

    # Test 5: Load and validate test split
    mnistm_test = MNISTM(split="test")
    assert len(mnistm_test) == EXPECTED_NUM_TEST, f"Expected {EXPECTED_NUM_TEST} test samples, got {len(mnistm_test)}."

    # Test 6: Per-class counts of the original release
    assert Counter(mnistm_train.table.column("label").to_pylist()) == {
        0: 5825, 1: 6640, 2: 5853, 3: 6028, 4: 5746, 5: 5331, 6: 5812, 7: 6158, 8: 5757, 9: 5851
    }  # fmt: skip
    assert Counter(mnistm_test.table.column("label").to_pylist()) == {
        0: 878, 1: 1016, 2: 933, 3: 908, 4: 890, 5: 807, 6: 856, 7: 914, 8: 880, 9: 919
    }  # fmt: skip

    # Test 7: Torch formatting yields CHW tensors
    torch_image = mnistm_train.with_format("torch")[0]["image"]
    assert tuple(torch_image.shape) == (3, 32, 32), f"Expected (3, 32, 32), got {tuple(torch_image.shape)}."
