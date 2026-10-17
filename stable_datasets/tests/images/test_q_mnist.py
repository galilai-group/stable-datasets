import gzip
import struct

import numpy as np
import pytest

from stable_datasets.images import QMNIST


def _idx_pair(tmp_path, count=3, prefix="tiny"):
    images = np.zeros((count, 28, 28), dtype=np.uint8)
    images[:, 0, 0] = np.arange(count, dtype=np.int64) % 256
    labels = np.zeros((count, 8), dtype=np.int32)
    labels[:, 0] = np.arange(count) % 10
    labels[:, 1] = 4
    labels[:, 2] = 2105
    labels[:, 3] = np.arange(count) % 150
    labels[:, 4] = 30 + labels[:, 0]
    labels[:, 5] = np.arange(count) + 100000
    image_path, label_path = tmp_path / f"{prefix}_images.gz", tmp_path / f"{prefix}_labels.gz"
    image_path.write_bytes(gzip.compress(struct.pack(">IIII", 2051, count, 28, 28) + images.tobytes()))
    label_path.write_bytes(gzip.compress(struct.pack(">III", 3074, count, 8) + labels.astype(">i4").tobytes()))
    return image_path, label_path, images, labels


def _rows(images_path, labels_path, split="train"):
    return object.__new__(QMNIST)._generate_examples(images_path, labels_path, split)


def test_extended_labels_and_pixels_are_preserved(tmp_path):
    ip, lp, images, labels = _idx_pair(tmp_path)
    rows = list(_rows(ip, lp))
    assert [key for key, _ in rows] == [0, 1, 2]
    for index, row in rows:
        np.testing.assert_array_equal(row["image"], images[index])
        assert row["label"] == labels[index, 0]
        assert row["nist_writer_id"] == 2105  # catches wrong int32 endianness
        assert row["nist_hsf_series"] == 4
        assert row["nist_class_code"] == 30 + index
        assert row["nist_global_index"] == 100000 + index
        assert row["duplicate"] == row["unused"] == 0


def test_official_test_subset_boundary_and_order(tmp_path):
    ip, lp, _, _ = _idx_pair(tmp_path, count=10002)
    first = list(_rows(ip, lp, "test10k"))
    remaining = list(_rows(ip, lp, "test50k"))
    assert len(first) == 10000
    assert first[-1][0] == 9999
    assert [key for key, _ in remaining] == [10000, 10001]
    assert remaining[0][1]["nist_global_index"] == 110000
    assert remaining[0][1]["label"] == 0
    assert remaining[0][1]["image"][0, 0] == 16


@pytest.mark.parametrize(
    ("target", "offset", "replacement", "message"),
    [
        ("images", 0, 0, "unsigned-byte"),
        ("images", 8, 27, "shape"),
        ("labels", 0, 2049, "eight columns"),
        ("labels", 8, 1, "eight columns"),
        ("labels", 12, 10, "digit labels"),
    ],
)
def test_invalid_idx_rejected(tmp_path, target, offset, replacement, message):
    ip, lp, _, _ = _idx_pair(tmp_path)
    path = ip if target == "images" else lp
    data = bytearray(gzip.decompress(path.read_bytes()))
    data[offset : offset + 4] = struct.pack(">I", replacement)
    path.write_bytes(gzip.compress(data))
    with pytest.raises(ValueError, match=message):
        list(_rows(ip, lp))


@pytest.mark.parametrize("target", ["images", "labels"])
def test_truncated_payload_rejected(tmp_path, target):
    ip, lp, _, _ = _idx_pair(tmp_path)
    path = ip if target == "images" else lp
    path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes())[:-1]))
    with pytest.raises(ValueError, match="payload length"):
        list(_rows(ip, lp))


def test_mismatched_counts_rejected(tmp_path):
    ip, _, _, _ = _idx_pair(tmp_path, count=3, prefix="a")
    _, lp, _, _ = _idx_pair(tmp_path, count=2, prefix="b")
    with pytest.raises(ValueError, match="counts differ"):
        list(_rows(ip, lp))


def test_public_api_and_cache_reopen(tmp_path, monkeypatch):
    train_ip, train_lp, _, _ = _idx_pair(tmp_path, count=3, prefix="train")
    test_ip, test_lp, _, _ = _idx_pair(tmp_path, count=10002, prefix="test")
    paths = dict(zip((asset.url for asset in QMNIST.SOURCE.assets.values()), [train_ip, train_lp, test_ip, test_lp]))
    monkeypatch.setattr("stable_datasets.images.q_mnist.download", lambda asset, **kwargs: paths[asset.url])
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "processed"}
    datasets = QMNIST(**kwargs)
    assert {split: len(ds) for split, ds in datasets.items()} == {
        "train": 3,
        "test": 10002,
        "test10k": 10000,
        "test50k": 2,
    }
    assert datasets["train"][2]["image"].mode == "L"
    assert datasets["train"].features["label"].names == [str(i) for i in range(10)]
    tensor = datasets["train"].with_format("torch")[2]["image"]
    assert tuple(tensor.shape) == (1, 28, 28)
    assert float(tensor[0, 0, 0]) == pytest.approx(2 / 255)

    def no_download(*args, **kwargs):
        pytest.fail("Reopening a complete processed cache must not download")

    monkeypatch.setattr("stable_datasets.images.q_mnist.download", no_download)
    reopened = QMNIST(split="test50k", **kwargs)
    assert reopened[0]["nist_global_index"] == 110000


@pytest.mark.large
def test_qmnist_real_download(tmp_path):
    datasets = QMNIST(download_dir=tmp_path / "raw", processed_cache_dir=tmp_path / "processed")
    assert {split: len(ds) for split, ds in datasets.items()} == {
        "train": 60000,
        "test": 60000,
        "test10k": 10000,
        "test50k": 50000,
    }
    for split in ("train", "test"):
        assert datasets[split][0]["image"].size == (28, 28)
        assert datasets[split][0]["image"].mode == "L"
    for sub, offset in (("test10k", 0), ("test50k", 10000)):
        for index in (0, len(datasets[sub]) - 1):
            original, subset = datasets["test"][offset + index], datasets[sub][index]
            np.testing.assert_array_equal(original.pop("image"), subset.pop("image"))
            assert original == subset
