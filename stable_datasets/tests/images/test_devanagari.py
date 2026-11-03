import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Devanagari


def _archive(tmp_path, extra=None):
    path = tmp_path / "dhcd.zip"
    buffer = io.BytesIO()
    Image.new("L", (32, 32), 83).save(buffer, format="PNG")
    prefix = "DevanagariHandwrittenCharacterDataset/"
    with ZipFile(path, "w") as archive:
        for split in ("Train", "Test"):
            # Write in reverse order to check deterministic enumeration.
            for name in ("digit_9", "character_10_yna", "character_2_kha", "character_1_ka"):
                archive.writestr(f"{prefix}{split}/{name}/sample.png", buffer.getvalue())
        archive.writestr(f"{prefix}Train/README.txt", "not an image")
        if extra:
            archive.writestr(f"{prefix}Train/{extra}/sample.png", buffer.getvalue())
    return path


def test_original_class_order_and_split_isolation(tmp_path):
    path = _archive(tmp_path)
    builder = object.__new__(Devanagari)
    train = list(builder._generate_examples(path, "train"))
    test = list(builder._generate_examples(path, "test"))
    assert len(train) == len(test) == 4
    assert not {key for key, _ in train} & {key for key, _ in test}
    assert [key for key, _ in train] == sorted(key for key, _ in train)
    expected = {"character_1_ka": 0, "character_2_kha": 1, "character_10_yna": 9, "digit_9": 45}
    for key, row in train:
        assert row["label"] == expected[key.split("/")[-2]]
        assert np.all(np.array(Image.open(io.BytesIO(row["image"]))) == 83)


def test_unknown_class_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unexpected"):
        list(object.__new__(Devanagari)._generate_examples(_archive(tmp_path, "unknown"), "train"))


def test_empty_split_is_rejected(tmp_path):
    path = tmp_path / "empty.zip"
    with ZipFile(path, "w"):
        pass
    with pytest.raises(ValueError, match="No images"):
        list(object.__new__(Devanagari)._generate_examples(path, "test"))


def test_cache_roundtrip(tmp_path, monkeypatch):
    path = _archive(tmp_path)
    calls = []

    def download(specs, dest_folder):
        calls.append(list(specs))
        assert dest_folder == tmp_path / "raw"
        return [path]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "cache"}
    ds = Devanagari(**kwargs)
    assert set(ds) == {"train", "test"}
    assert len(calls) == 1 and len(calls[0]) == 1  # shared archive downloaded once
    assert ds["train"].features["label"].num_classes == 46
    assert ds["train"][0]["image"].mode == "L"
    assert ds["train"].with_format("torch")[0]["image"].shape == (1, 32, 32)

    def no_download(*args, **kwargs):
        raise AssertionError("Cache reload must not download")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    cached = Devanagari(split="test", **kwargs)
    assert len(cached) == 4
    assert np.all(np.array(cached[0]["image"]) == 83)


def test_official_splits():
    ds = Devanagari()
    for split, size, per_class in [("train", 78200, 1700), ("test", 13800, 300)]:
        assert len(ds[split]) == size
        assert np.array_equal(
            np.bincount([sample["label"] for sample in ds[split].with_format("raw")], minlength=46),
            np.full(46, per_class),
        )
        assert ds[split][0]["image"].size == (32, 32)
        assert ds[split][0]["image"].mode == "L"


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown Devanagari split"):
        list(object.__new__(Devanagari)._generate_examples(_archive(tmp_path), "validation"))
