import io
import tarfile

import numpy as np
from PIL import Image

import stable_datasets.utils as utils
from stable_datasets.images import CINIC10


def _png_bytes(mode, color):
    buf = io.BytesIO()
    Image.new(mode, (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_synthetic_archive(path):
    """Write a tiny archive with the CINIC-10 layout, including a top-level file the builder must skip."""
    members = {
        "README.md": b"not an image",
        "train/airplane/cifar10-train-1.png": _png_bytes("RGB", (255, 0, 0)),
        "train/airplane/n02691156_1.png": _png_bytes("RGB", (0, 255, 0)),
        "train/truck/n04467665_2.png": _png_bytes("L", 128),  # grayscale ImageNet image
        "valid/cat/cifar10-test-7.png": _png_bytes("RGB", (0, 0, 255)),
        "test/ship/n04147183_3.png": _png_bytes("RGB", (9, 9, 9)),
        "test/ship/cifar10-test-8.png": _png_bytes("RGB", (8, 8, 8)),
    }
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_cinic10_parses_archive_layout(tmp_path, monkeypatch):
    archive = tmp_path / "CINIC-10.tar.gz"
    _write_synthetic_archive(archive)
    monkeypatch.setattr(utils, "bulk_download", lambda infos, dest_folder: [archive for _ in infos])

    ds = CINIC10(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "raw")
    assert {name: len(split) for name, split in ds.items()} == {"train": 3, "validation": 1, "test": 2}

    train = ds["train"]
    names = train.features["label"].names
    rows = sorted((names[train[i]["label"]], train[i]["origin"], train[i]["image"].mode) for i in range(len(train)))
    assert rows == [("airplane", 0, "RGB"), ("airplane", 1, "RGB"), ("truck", 1, "RGB")]

    # The grayscale source image is stored as a 3-channel image with the same gray value.
    truck = next(train[i] for i in range(len(train)) if names[train[i]["label"]] == "truck")
    assert np.array(truck["image"]).shape == (32, 32, 3)
    assert np.unique(np.array(truck["image"])).tolist() == [128]

    validation = ds["validation"]
    assert validation.features["label"].names[validation[0]["label"]] == "cat"
    assert validation[0]["origin"] == 0
    assert sorted(ds["test"][i]["origin"] for i in range(2)) == [0, 1]


def test_cinic10_dataset():
    train = CINIC10(split="train")
    assert len(train) == 90000, f"Expected 90000 training samples, got {len(train)}."

    sample = train[0]
    assert set(sample.keys()) == {"image", "label", "origin"}
    assert isinstance(sample["image"], Image.Image)
    image_np = np.array(sample["image"])
    assert image_np.shape == (32, 32, 3), f"Image should have shape (32, 32, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8
    assert train.features["origin"].names == ["cifar10", "imagenet"]

    # Every split has 9,000 images per class, 20,000 from CIFAR-10 and 70,000 from ImageNet.
    for split in ("train", "validation", "test"):
        table = CINIC10(split=split).table
        assert table.num_rows == 90000, f"{split}: expected 90000 rows, got {table.num_rows}"
        labels = table.column("label").to_numpy()
        origins = table.column("origin").to_numpy()
        assert np.bincount(labels, minlength=10).tolist() == [9000] * 10, split
        assert np.bincount(origins, minlength=2).tolist() == [20000, 70000], split

    assert tuple(train.with_format("torch")[0]["image"].shape) == (3, 32, 32)
