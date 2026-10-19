"""Offline parser and real Arrow cache checks for Scene8."""

import io
import zipfile

import pytest
from PIL import Image

from stable_datasets.images import Scene8


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


def test_scene8_cold_warm_cache_and_verbatim_images(tmp_path, monkeypatch):
    archive = tmp_path / "scene8.zip"
    payloads = _make_archive(archive)
    kwargs = {"archive_path": archive, "processed_cache_dir": tmp_path / "processed"}
    data = Scene8(**kwargs)
    assert set(data) == {"train"}
    ds = data["train"]
    assert ds.features["label"].names == list(Scene8.CLASS_NAMES)
    assert [row["image_id"] for row in ds] == ["coast_b", "forest_a"]
    for row in ds.with_format("raw"):
        assert row["image"] == payloads[row["image_id"]]
    assert [row["label"] for row in ds] == [0, 1]
    assert ds[0]["image"].size == (8, 5)
    assert ds.with_format("torch")[0]["image"].shape == (3, 5, 8)
    archive.unlink()
    monkeypatch.setattr("stable_datasets.images.scene8.download", lambda *a, **k: pytest.fail("network on cache hit"))
    reopened = Scene8(split="train", **kwargs)
    assert [row["image_id"] for row in reopened] == ["coast_b", "forest_a"]
    assert reopened.with_format("raw")[0]["image"] == payloads["coast_b"]


@pytest.mark.parametrize("names, message", [(("unknown_a",), "Unknown Scene8"), ((), "no JPEG")])
def test_scene8_rejects_invalid_collection(tmp_path, names, message):
    path = tmp_path / "bad.zip"
    _make_archive(path, names)
    builder = object.__new__(Scene8)
    with pytest.raises(ValueError, match=message):
        list(builder._generate_examples(path, "train"))


def test_scene8_rejects_repeated_ids(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/coast_a.jpg", b"first")
        archive.writestr("b/coast_a.jpg", b"second")
    with pytest.raises(ValueError, match="Duplicate Scene8 image ID"):
        list(object.__new__(Scene8)._generate_examples(path, "train"))


def test_scene8_download_route_and_invalid_split(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    _make_archive(path)
    calls = []

    def download(info, **kwargs):
        calls.append(info)
        return path

    monkeypatch.setattr("stable_datasets.images.scene8.download", download)
    ds = Scene8(split="train", processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 2
    assert calls == [Scene8.SOURCE.assets["train"]]
    with pytest.raises(ValueError, match="Split 'test' not found"):
        Scene8(split="test", processed_cache_dir=tmp_path / "processed")
