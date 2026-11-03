"""Tests for the ImageNet-Sketch builder.

All tests except the last run offline on a tiny zip with the layout of the real
ImageNet-Sketch.zip. The last test downloads the real 7.6 GB archive and checks it.
"""

import io
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import ImageNet1K, ImageNetSketch
from stable_datasets.images._imagenet_wnids import WNID_TO_IDX


def _encode(shape, fmt, seed):
    pixels = np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


# One RGB JPEG that must pass through unchanged, and two files that need conversion.
RGB_JPEG = _encode((12, 16, 3), "JPEG", seed=0)
RGBA_PNG = _encode((14, 10, 4), "PNG", seed=1)
GRAY_JPEG = _encode((8, 12), "JPEG", seed=2)

# Written out of name order, to check that rows come out sorted by entry name.
FAKE_ENTRIES = {
    "sketch/n15075141/c.JPEG": GRAY_JPEG,
    "sketch/": None,
    "sketch/n01440764/": None,
    "sketch/n01440764/a.JPEG": RGB_JPEG,
    "sketch/n01443537/b.JPEG": RGBA_PNG,
}
EXPECTED_ROWS = [
    ("sketch/n01440764/a.JPEG", "n01440764", RGB_JPEG),
    ("sketch/n01443537/b.JPEG", "n01443537", RGBA_PNG),
    ("sketch/n15075141/c.JPEG", "n15075141", GRAY_JPEG),
]


def _write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, b"" if data is None else data)
    return path


def _patch_download(monkeypatch, zip_path):
    """Serve ``zip_path`` instead of downloading and return the list of requested DownloadInfos."""
    requested = []

    def fake_bulk_download(infos, dest_folder):
        infos = list(infos)
        requested.extend(infos)
        return [zip_path] * len(infos)

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    return requested


def _metadata_only(builder_cls):
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder)
    return builder


@pytest.fixture(scope="module")
def fake_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("imagenet_sketch")
    zip_path = _write_zip(root / "ImageNet-Sketch.zip", FAKE_ENTRIES)
    with pytest.MonkeyPatch.context() as mp:
        requested = _patch_download(mp, zip_path)
        ds = ImageNetSketch(split="test", download_dir=root / "downloads", processed_cache_dir=root / "processed")
    return {"ds": ds, "requested": requested, "root": root, "zip_path": zip_path}


def test_label_names_match_imagenet_1k():
    names = _metadata_only(ImageNetSketch).info.features["label"].names
    assert names == _metadata_only(ImageNet1K).info.features["label"].names


def test_downloads_the_huggingface_zip_once(fake_build):
    (info,) = fake_build["requested"]
    assert list(ImageNetSketch.SOURCE["assets"]) == ["test"]
    assert info.url == "https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip"


def test_directory_entries_are_skipped(fake_build):
    assert len(fake_build["ds"]) == len(EXPECTED_ROWS)


def test_labels_come_from_wnid_folders(fake_build):
    ds = fake_build["ds"]
    names = ds.features["label"].names
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert labels == [WNID_TO_IDX[wnid] for _, wnid, _ in EXPECTED_ROWS]
    assert [names[label] for label in labels] == [wnid for _, wnid, _ in EXPECTED_ROWS]


def test_rgb_jpeg_is_stored_byte_identical(fake_build):
    assert fake_build["ds"].with_format("raw")[0]["image"] == RGB_JPEG


def test_other_images_are_converted_to_rgb_losslessly(fake_build):
    ds = fake_build["ds"]
    raw = ds.with_format("raw")
    for i in (1, 2):
        assert Image.open(io.BytesIO(raw[i]["image"])).format == "PNG"
        image = ds[i]["image"]
        assert image.mode == "RGB"
        expected = Image.open(io.BytesIO(EXPECTED_ROWS[i][2])).convert("RGB")
        np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))


def test_torch_format(fake_build):
    image = fake_build["ds"].with_format("torch")[2]["image"]
    assert image.dtype.is_floating_point
    assert tuple(image.shape) == (3, 8, 12)


def test_second_load_uses_cache(fake_build, monkeypatch):
    requested = _patch_download(monkeypatch, fake_build["zip_path"])
    root = fake_build["root"]
    reopened = ImageNetSketch(split="test", download_dir=root / "downloads", processed_cache_dir=root / "processed")
    assert requested == []
    assert len(reopened) == len(EXPECTED_ROWS)


def test_order_does_not_depend_on_archive_order(tmp_path):
    builder = _metadata_only(ImageNetSketch)
    forward = _write_zip(tmp_path / "forward.zip", FAKE_ENTRIES)
    backward = _write_zip(tmp_path / "backward.zip", dict(reversed(FAKE_ENTRIES.items())))

    keys = [key for key, _ in builder._generate_examples(data_path=forward, split="test")]
    assert keys == [entry for entry, _, _ in EXPECTED_ROWS]
    assert [key for key, _ in builder._generate_examples(data_path=backward, split="test")] == keys


def test_unknown_class_folder_raises(tmp_path):
    zip_path = _write_zip(tmp_path / "bad.zip", {"sketch/n99999999/a.JPEG": RGB_JPEG})
    builder = _metadata_only(ImageNetSketch)
    with pytest.raises(ValueError, match="sketch/n99999999/a.JPEG"):
        list(builder._generate_examples(data_path=zip_path, split="test"))


def test_imagenet_sketch_real_dataset():
    ds = ImageNetSketch(split="test")

    # The README says 50,000 images; the released archive has 50,889.
    assert len(ds) == 50889

    # Every class has 50 images, except a few with 51.
    counts = Counter(ds.table.column("label").to_pylist())
    assert set(counts) == set(range(1000))
    assert set(counts.values()) == {50, 51}

    assert all(ds[i]["image"].mode == "RGB" for i in range(0, len(ds), 500))
