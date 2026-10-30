"""Tests for the ImageNetV2 builder.

The offline tests build a tiny fake copy of the real archive layout, so they need no
download. The last test downloads the three real 1.2 GB archives and checks them.
"""

import io
import tarfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import ImageNet1K, ImageNetV2


BASE_URL = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main"
TOP_DIR = "imagenetv2-matched-frequency-format-val"


def _jpeg(mode, size, seed):
    channels = {"L": 1, "RGB": 3, "CMYK": 4}[mode]
    pixels = np.random.default_rng(seed).bytes(size[0] * size[1] * channels)
    buf = io.BytesIO()
    Image.frombytes(mode, size, pixels).save(buf, format="JPEG")
    return buf.getvalue()


# One image per class folder, so the label identifies the source file. Folder names are
# ImageNet-1K class indices, as in the real archive.
SOURCE_IMAGES = {
    0: _jpeg("RGB", (16, 12), seed=0),
    1: _jpeg("L", (10, 14), seed=1),
    999: _jpeg("CMYK", (9, 7), seed=2),
}


def _write_tar(path, entries):
    """Write an uncompressed tar from ``{name: bytes}``; a ``None`` value adds a folder entry."""
    with tarfile.open(path, "w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def _write_fake_archive(path):
    entries = {f"{TOP_DIR}/": None}
    for label, data in SOURCE_IMAGES.items():
        entries[f"{TOP_DIR}/{label}/"] = None
        entries[f"{TOP_DIR}/{label}/{label:04d}abcdef.jpeg"] = data
    _write_tar(path, entries)


def _patch_download(monkeypatch, archive_path):
    """Serve ``archive_path`` instead of downloading; return the list of requested DownloadInfos."""
    requested = []

    def fake_bulk_download(infos, dest_folder):
        infos = list(infos)
        requested.extend(infos)
        return [archive_path] * len(infos)

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    return requested


def _metadata_only_builder(builder_cls=ImageNetV2, **kwargs):
    """Build an instance without downloading or caching anything."""
    builder = object.__new__(builder_cls)
    builder_cls.__init__(builder, **kwargs)
    return builder


@pytest.fixture(scope="module")
def fake_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("imagenet_v2")
    # Like the real files: an uncompressed tar behind a .tar.gz name.
    archive_path = root / "imagenetv2-matched-frequency.tar.gz"
    _write_fake_archive(archive_path)
    with pytest.MonkeyPatch.context() as mp:
        requested = _patch_download(mp, archive_path)
        ds = ImageNetV2(split="test", download_dir=root / "downloads", processed_cache_dir=root / "processed")
    return {"ds": ds, "root": root, "archive_path": archive_path, "requested": requested}


def test_labels_are_folder_indices(fake_build):
    assert sorted(fake_build["ds"].table.column("label").to_pylist()) == [0, 1, 999]


def test_rgb_jpeg_is_stored_unchanged(fake_build):
    raw = fake_build["ds"].with_format("raw")
    stored = {row["label"]: row["image"] for row in (raw[i] for i in range(len(raw)))}
    assert stored[0] == SOURCE_IMAGES[0]


def test_images_decode_to_source_rgb_pixels(fake_build):
    ds = fake_build["ds"]
    for i in range(len(ds)):
        sample = ds[i]
        data = SOURCE_IMAGES[sample["label"]]
        assert sample["image"].mode == "RGB"
        np.testing.assert_array_equal(
            np.asarray(sample["image"]), np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        )


def test_torch_format(fake_build):
    ds = fake_build["ds"].with_format("torch")
    for i in range(len(ds)):
        sample = ds[i]
        width, height = Image.open(io.BytesIO(SOURCE_IMAGES[int(sample["label"])])).size
        assert sample["image"].dtype.is_floating_point
        assert tuple(sample["image"].shape) == (3, height, width)


def test_reopens_from_cache(fake_build, monkeypatch):
    assert len(fake_build["requested"]) == 1

    requested = _patch_download(monkeypatch, fake_build["archive_path"])
    root = fake_build["root"]
    reopened = ImageNetV2(split="test", download_dir=root / "downloads", processed_cache_dir=root / "processed")
    assert len(reopened) == len(SOURCE_IMAGES)
    assert requested == []


@pytest.mark.parametrize("folder", ["n01440764", "1000", "-1"])
def test_unexpected_class_folder_raises(tmp_path, folder):
    archive_path = tmp_path / "bad.tar.gz"
    _write_tar(archive_path, {f"{TOP_DIR}/{folder}/img.jpeg": SOURCE_IMAGES[0]})

    builder = _metadata_only_builder()
    with pytest.raises(ValueError, match="class folder"):
        list(builder._generate_examples(data_path=archive_path, split="test"))


@pytest.mark.parametrize(
    "config_name, archive",
    [
        ("matched-frequency", "imagenetv2-matched-frequency.tar.gz"),
        ("threshold-0.7", "imagenetv2-threshold0.7.tar.gz"),
        ("top-images", "imagenetv2-top-images.tar.gz"),
    ],
)
def test_config_selects_archive(config_name, archive):
    assets = _metadata_only_builder(config_name=config_name)._source()["assets"]
    assert list(assets) == ["test"]
    assert assets["test"].url == f"{BASE_URL}/{archive}"
    assert assets["test"].checksum.startswith("sha256:")


def test_default_config():
    assert _metadata_only_builder().config.name == "matched-frequency"


def test_unknown_config_raises():
    with pytest.raises(ValueError, match="Unknown config"):
        _metadata_only_builder(config_name="matched_frequency")


def test_features_match_imagenet_1k():
    features = _metadata_only_builder().info.features
    imagenet_1k_features = _metadata_only_builder(ImageNet1K).info.features
    assert features["label"].names == imagenet_1k_features["label"].names
    assert repr(features) == repr(imagenet_1k_features)


@pytest.mark.parametrize("config_name", ["matched-frequency", "threshold-0.7", "top-images"])
def test_imagenet_v2_real_dataset(config_name):
    ds = ImageNetV2(split="test", config_name=config_name)
    assert Counter(ds.table.column("label").to_pylist()) == dict.fromkeys(range(1000), 10)
    assert all(sample["image"].mode == "RGB" for sample in ds)
