"""Real-download integration and offline parsing tests for Scene8."""

import io
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Scene8


def test_scene8_real_dataset(tmp_path):
    """Download the published archive and verify the original collection."""
    kwargs = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    data = Scene8(**kwargs)
    assert set(data) == {"train"}
    ds = data["train"]
    assert len(ds) == 2688
    assert ds.features["label"].names == list(Scene8.CLASS_NAMES)
    labels = ds.table.column("label").to_pylist()
    assert Counter(labels) == dict(enumerate([360, 328, 260, 308, 374, 410, 292, 356]))
    ids = ds.table.column("image_id").to_pylist()
    assert len(set(ids)) == len(ds) and ids == sorted(ids)
    # One real image from every class checks archive parsing and decoding.
    for label in range(8):
        sample = ds[labels.index(label)]
        assert set(sample) == {"image", "label", "image_id"}
        assert sample["image_id"].startswith(Scene8.CLASS_NAMES[label] + "_")
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB" and sample["image"].size == (256, 256)
        assert np.asarray(sample["image"]).dtype == np.uint8
    assert ds.with_format("torch")[0]["image"].shape == (3, 256, 256)
    reopened = Scene8(split="train", **kwargs)
    assert len(reopened) == 2688
    assert reopened.with_format("raw")[0] == ds.with_format("raw")[0]
    with pytest.raises(ValueError, match="Split 'test' not found"):
        Scene8(split="test", **kwargs)


def _make_archive(path, names=("forest_a", "coast_b")):
    payloads = {}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.txt", "fixture")
        for i, name in enumerate(names):
            buffer = io.BytesIO()
            Image.new("RGB", (7 + i, 5), (20 * i, 70, 90)).save(buffer, format="JPEG")
            payloads[name] = buffer.getvalue()
            archive.writestr(f"original/{name}.jpg", payloads[name])
    return payloads


def test_scene8_parser_preserves_bytes(tmp_path):
    path = tmp_path / "source.zip"
    payloads = _make_archive(path)
    rows = [row for _, row in object.__new__(Scene8)._generate_examples(path, "train")]
    assert [r["image_id"] for r in rows] == ["coast_b", "forest_a"]
    assert [r["label"] for r in rows] == [0, 1]
    for row in rows:
        assert row["image"] == payloads[row["image_id"]]


@pytest.mark.parametrize("names, message", [(("unknown_a",), "Unknown Scene8"), ((), "no JPEG")])
def test_scene8_rejects_invalid_collection(tmp_path, names, message):
    path = tmp_path / "bad.zip"
    _make_archive(path, names)
    with pytest.raises(ValueError, match=message):
        list(object.__new__(Scene8)._generate_examples(path, "train"))


def test_scene8_rejects_repeated_ids(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/coast_a.jpg", b"first")
        archive.writestr("b/coast_a.jpg", b"second")
    with pytest.raises(ValueError, match="Duplicate Scene8 image ID"):
        list(object.__new__(Scene8)._generate_examples(path, "train"))
