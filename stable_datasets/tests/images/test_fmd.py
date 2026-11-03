"""Native FMD cache tests and public-download integration coverage."""

import hashlib
import io
import stat
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets import StableDataset, StableDatasetDict, utils
from stable_datasets.images import FMD, fmd
from stable_datasets.images.fmd import FMD_CLASS_NAMES


def _encoded(pixels, *, image_format="JPEG"):
    buffer = io.BytesIO()
    PILImage.fromarray(pixels).save(buffer, format=image_format, quality=93, subsampling=0)
    data = buffer.getvalue()
    with PILImage.open(io.BytesIO(data)) as decoded:
        return data, (decoded.mode, np.array(decoded))


def _write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            if isinstance(name, str):
                member = zipfile.ZipInfo(name)
                # The Windows writer normalizes backslashes in ZipInfo.__init__;
                # preserve the malicious fixture spelling in the actual ZIP.
                member.filename = name
                member.orig_filename = name
            else:
                member = name
            archive.writestr(member, data)


@pytest.fixture
def release(tmp_path):
    ids = ["fabric/Same.jpg", "foliage/Same.jpg", "water/space name.JPG", "wood/Binary.jpg"]
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 5
    gray = np.arange(15, dtype=np.uint8).reshape(3, 5) * 13
    masks = [gray, pixels, np.full((3, 5, 3), 255, dtype=np.uint8), np.full((3, 5), 255, dtype=np.uint8)]
    entries = [("image/", b""), ("mask/", b""), ("image/fabric/", b"")]
    expected = {}
    for index, image_id in enumerate(ids):
        photo, image_expected = _encoded(pixels + index * 7)
        mask, mask_expected = _encoded(masks[index])
        expected[image_id] = {"image": image_expected, "mask": mask_expected}
        entries.extend([("image/" + image_id, photo), ("mask/" + image_id, mask)])
    # Invalid image bytes here must never be decoded or counted as photos.
    entries.extend(
        (name, b"ancillary")
        for name in (
            "image/fabric/normalizeImage.m",
            "image/fabric/normalizeImage.asv",
            "image/foliage/Thumbs.db",
            "image/.DS_Store",
            "__MACOSX/image/fabric/._Same.jpg",
            "thumbnails/fabric/Same.jpg",
        )
    )
    path = tmp_path / "FMD.zip"
    _write_zip(path, entries)
    return path, entries, expected


def _mock_download(monkeypatch, tmp_path, archive_path):
    def local_download(specs, dest_folder):
        assert Path(dest_folder) == tmp_path / "downloads"
        assert list(specs) == [FMD.SOURCE.assets["train"]]
        return [archive_path]

    monkeypatch.setattr(utils, "bulk_download", local_download)


def _load(monkeypatch, tmp_path, archive_path, *, split="train"):
    _mock_download(monkeypatch, tmp_path, archive_path)
    return FMD(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", split=split)


def _forbid_raw_assets(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("warm FMD cache must not download or reopen raw assets")

    # The inherited split generator resolves both download names in utils.
    monkeypatch.setattr(utils, "bulk_download", forbidden)
    monkeypatch.setattr(utils, "download", forbidden)
    monkeypatch.setattr(utils.requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(fmd.zipfile, "ZipFile", forbidden)


def test_metadata_and_export_do_not_download(monkeypatch):
    _forbid_raw_assets(monkeypatch)
    builder = object.__new__(FMD)
    builder.__init__()
    assert str(builder.VERSION) == "1.0.0"
    assert builder._candidate_splits() == ["train"]
    assert FMD_CLASS_NAMES == [
        "fabric",
        "foliage",
        "glass",
        "leather",
        "metal",
        "paper",
        "plastic",
        "stone",
        "water",
        "wood",
    ]
    assert builder.info.features["label"].names == FMD_CLASS_NAMES
    assert set(builder.info.features) == {"image", "label", "image_id", "mask"}
    assert builder.info.supervised_keys == ("image", "label")
    assert builder.info.features["image"].encode_format == "PNG"
    assert builder.info.features["mask"].encode_format == "PNG"
    assert "exceptions" in builder.info.license
    assert "10.1167/14.9.12" in builder.info.citation
    assert set(builder.SOURCE.assets) == {"train"}
    assert builder.SOURCE.assets["train"].url == "https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip"


@pytest.mark.parametrize("kwargs", [{}, {"split": None}, {"split": "train"}])
def test_constructor_split_contract(monkeypatch, tmp_path, release, kwargs):
    archive_path, _, expected = release
    _mock_download(monkeypatch, tmp_path, archive_path)
    result = FMD(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", **kwargs)
    if kwargs.get("split") is None:
        assert isinstance(result, StableDatasetDict)
        assert set(result) == {"train"}
        result = result["train"]
    assert isinstance(result, StableDataset)
    assert len(result) == len(expected)


def test_native_cache_preserves_original_pixels_and_modes(monkeypatch, tmp_path, release):
    import torch

    archive_path, _, expected = release
    dataset = _load(monkeypatch, tmp_path, archive_path)
    assert len(dataset) == len(expected)
    assert [row["image_id"] for row in dataset] == sorted(expected)
    for index, row in enumerate(dataset):
        assert set(row) == {"image", "label", "image_id", "mask"}
        assert row["label"] == FMD_CLASS_NAMES.index(row["image_id"].split("/")[0])
        for field in ("image", "mask"):
            mode, pixels = expected[row["image_id"]][field]
            assert isinstance(row[field], PILImage.Image)
            assert row[field].mode == mode
            assert row[field].size == (5, 3)
            assert row[field].format == "PNG"
            np.testing.assert_array_equal(np.array(row[field]), pixels)
            raw = dataset.with_format("raw")[index][field]
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
            array = dataset.with_format("numpy")[index][field]
            assert array.dtype == np.uint8
            np.testing.assert_array_equal(array, pixels)
            channels = pixels[:, :, None] if pixels.ndim == 2 else pixels
            tensor = dataset.with_format("torch")[index][field]
            assert tensor.dtype == torch.float32
            np.testing.assert_array_equal(tensor.numpy(), channels.transpose(2, 0, 1).astype(np.float32) / 255)
    assert dataset[0]["mask"].mode == "L"
    assert not set(np.unique(np.array(dataset[0]["mask"]))).issubset({0, 255})
    rgb_mask = np.array(dataset[1]["mask"])
    assert dataset[1]["mask"].mode == "RGB"
    assert np.any(rgb_mask[..., 0] != rgb_mask[..., 1])

    cache_files = {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (tmp_path / "processed").rglob("*") if p.is_file()
    }
    assert cache_files
    _forbid_raw_assets(monkeypatch)
    for kwargs in ({}, {"split": "train"}, {"split": None}):
        warm = FMD(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", **kwargs)
        if kwargs.get("split") is None:
            assert isinstance(warm, StableDatasetDict)
            assert set(warm) == {"train"}
            warm = warm["train"]
        assert isinstance(warm, StableDataset)
        assert len(warm) == len(expected)
        for row in warm:
            for field in ("image", "mask"):
                mode, pixels = expected[row["image_id"]][field]
                assert row[field].mode == mode
                np.testing.assert_array_equal(np.array(row[field]), pixels)
    assert cache_files == {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in cache_files}


@pytest.mark.parametrize("split", ["test", "validation", "archive"])
def test_unsupported_split_uses_base_error_with_warm_cache(monkeypatch, tmp_path, release, split):
    _load(monkeypatch, tmp_path, release[0])
    _forbid_raw_assets(monkeypatch)
    with pytest.raises(ValueError, match=f"Split '{split}' not found. Available: \\['train'\\]"):
        FMD(split=split, processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads")


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../image/fabric/extra.jpg", "unsafe path"),
        ("/image/fabric/extra.jpg", "unsafe path"),
        ("image\\fabric\\extra.jpg", "unsafe path"),
        ("image/fabric/../extra.jpg", "unsafe path"),
        ("image//fabric/extra.jpg", "unsafe path"),
        ("image/fabric/C:extra.jpg", "unsafe path"),
        ("image/fabric/./extra.jpg", "unsafe path"),
        ("__MACOSX/../extra.jpg", "unsafe path"),
        ("image/unknown/extra.jpg", "unknown class"),
        ("image/Fabric/extra.jpg", "unknown class"),
        ("mask/fabric/nested/extra.jpg", "expected mask/class/image.jpg"),
        ("image/fabric/extra.png", "expected image/class/image.jpg"),
    ],
)
def test_bad_archive_paths(monkeypatch, tmp_path, release, name, message):
    archive_path, entries, _ = release
    _write_zip(archive_path, entries + [(name, b"invalid")])
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


@pytest.mark.parametrize("root", ["image", "mask"])
def test_duplicate_archive_member_fails(monkeypatch, tmp_path, release, root):
    archive_path, entries, _ = release
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(archive_path, entries + [(root + "/fabric/Same.jpg", b"duplicate")])
    with pytest.raises(ValueError, match="duplicate member"):
        _load(monkeypatch, tmp_path, archive_path)


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO])
def test_links_and_special_zip_entries_fail(monkeypatch, tmp_path, release, kind):
    archive_path, entries, _ = release
    member = zipfile.ZipInfo("image/fabric/link.jpg")
    member.create_system = 3
    member.external_attr = (kind | 0o644) << 16
    _write_zip(archive_path, entries + [(member, b"somewhere")])
    with pytest.raises(ValueError, match="links and special entries"):
        _load(monkeypatch, tmp_path, archive_path)


def test_nul_in_zip_member_name_fails(monkeypatch, tmp_path, release):
    archive_path, entries, _ = release
    _write_zip(archive_path, entries + [("image/fabric/Bad.jpg", b"invalid")])
    archive_path.write_bytes(archive_path.read_bytes().replace(b"Bad.jpg", b"B\x00d.jpg"))
    with pytest.raises(ValueError, match="unsafe path"):
        _load(monkeypatch, tmp_path, archive_path)


@pytest.mark.parametrize("missing_root", ["image", "mask"])
def test_pairing_requires_same_full_case_sensitive_id(monkeypatch, tmp_path, release, missing_root):
    archive_path, entries, _ = release
    _write_zip(archive_path, [(name, data) for name, data in entries if name != missing_root + "/fabric/Same.jpg"])
    error, message = (
        (FileNotFoundError, "missing masks") if missing_root == "mask" else (ValueError, "matching photos")
    )
    with pytest.raises(error, match=message):
        _load(monkeypatch, tmp_path, archive_path)


def test_mask_filename_case_is_not_normalized(monkeypatch, tmp_path, release):
    archive_path, entries, _ = release
    _write_zip(
        archive_path,
        [("mask/fabric/same.jpg" if name == "mask/fabric/Same.jpg" else name, data) for name, data in entries],
    )
    with pytest.raises(FileNotFoundError, match="fabric/Same.jpg"):
        _load(monkeypatch, tmp_path, archive_path)


@pytest.mark.parametrize("root", ["image", "mask"])
def test_corrupt_photo_or_mask_fails(monkeypatch, tmp_path, release, root):
    archive_path, entries, _ = release
    _write_zip(
        archive_path, [(name, b"broken" if name == root + "/fabric/Same.jpg" else data) for name, data in entries]
    )
    with pytest.raises(ValueError, match=f"cannot decode {root}"):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_mask_size_mismatch_fails_without_resize(monkeypatch, tmp_path, release):
    archive_path, entries, _ = release
    wrong_size, _ = _encoded(np.zeros((4, 5), dtype=np.uint8))
    _write_zip(
        archive_path, [(name, wrong_size if name == "mask/fabric/Same.jpg" else data) for name, data in entries]
    )
    with pytest.raises(ValueError, match="image/mask size mismatch"):
        _load(monkeypatch, tmp_path, archive_path)


@pytest.mark.parametrize("mode", ["P", "RGBA", "1"])
def test_unsupported_mask_modes_fail_without_conversion(monkeypatch, tmp_path, release, mode):
    archive_path, entries, _ = release
    buffer = io.BytesIO()
    PILImage.new(mode, (5, 3)).save(buffer, format="PNG")
    _write_zip(
        archive_path,
        [(name, buffer.getvalue() if name == "mask/fabric/Same.jpg" else data) for name, data in entries],
    )
    with pytest.raises(ValueError, match="unsupported mask mode"):
        _load(monkeypatch, tmp_path, archive_path)


def test_grayscale_photo_converts_to_rgb_but_mask_stays_l(monkeypatch, tmp_path, release):
    archive_path, entries, _ = release
    photo, (_, pixels) = _encoded(np.arange(15, dtype=np.uint8).reshape(3, 5) * 13)
    _write_zip(archive_path, [(name, photo if name == "image/fabric/Same.jpg" else data) for name, data in entries])
    dataset = _load(monkeypatch, tmp_path, archive_path)
    assert dataset[0]["image"].mode == "RGB"
    assert dataset[0]["mask"].mode == "L"
    np.testing.assert_array_equal(np.array(dataset[0]["image"]), np.repeat(pixels[:, :, None], 3, axis=2))


def test_empty_archive_has_no_silent_empty_dataset(monkeypatch, tmp_path):
    archive_path = tmp_path / "FMD.zip"
    _write_zip(archive_path, [("thumbnails/example.jpg", b"ignored")])
    with pytest.raises(ValueError, match="contains no photos"):
        _load(monkeypatch, tmp_path, archive_path)


def test_fmd_public_release(monkeypatch, tmp_path):
    """Download the public release through the builder and check every record."""
    download_dir = tmp_path / "downloads"
    processed_cache_dir = tmp_path / "processed"
    assert not download_dir.exists() and not processed_cache_dir.exists()
    datasets = FMD(processed_cache_dir=processed_cache_dir, download_dir=download_dir)
    assert isinstance(datasets, StableDatasetDict)
    assert set(datasets) == {"train"}
    dataset = datasets["train"]
    assert isinstance(dataset, StableDataset)
    assert len(dataset) == 1000
    archives = list(download_dir.glob("*.zip"))
    assert len(archives) == 1
    archive_path = archives[0]
    with archive_path.open("rb") as stream:
        fingerprint = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(block)
    assert "sha256:" + fingerprint.hexdigest() == FMD.SOURCE.assets["train"].checksum
    labels, modes = Counter(), Counter()
    nonbinary, unequal_channels = 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        photos = {
            name.removeprefix("image/")
            for name in archive.namelist()
            if name.startswith("image/") and name.endswith(".jpg")
        }
        actual = set()
        for index, row in enumerate(dataset):
            assert set(row) == {"image", "label", "image_id", "mask"}
            image_id = row["image_id"]
            assert isinstance(image_id, str)
            assert image_id not in actual
            actual.add(image_id)
            labels[image_id.split("/")[0]] += 1
            assert row["label"] == FMD_CLASS_NAMES.index(image_id.split("/")[0])
            assert isinstance(row["label"], int) and 0 <= row["label"] < 10
            assert row["image"].mode == "RGB"
            assert row["image"].size == row["mask"].size == (512, 384)
            for field, root in (("image", "image"), ("mask", "mask")):
                assert isinstance(row[field], PILImage.Image)
                with PILImage.open(io.BytesIO(archive.read(root + "/" + image_id))) as source:
                    original = source.convert("RGB") if field == "image" else source
                    assert row[field].mode == original.mode
                    np.testing.assert_array_equal(np.array(row[field]), np.array(original))
                assert dataset.with_format("raw")[index][field].startswith(b"\x89PNG\r\n\x1a\n")
            mask = np.array(row["mask"])
            modes[row["mask"].mode] += 1
            nonbinary += int(np.any((mask != 0) & (mask != 255)))
            unequal_channels += int(mask.ndim == 3 and np.any(mask != mask[..., :1]))
        assert actual == photos
    assert labels == dict.fromkeys(FMD_CLASS_NAMES, 100)
    assert modes == {"RGB": 743, "L": 257}
    assert nonbinary == 653
    assert unequal_channels == 142
    _forbid_raw_assets(monkeypatch)
    warm = FMD(split="train", processed_cache_dir=processed_cache_dir, download_dir=download_dir)
    assert len(warm) == 1000
    for before, after in zip(dataset.with_format("raw"), warm.with_format("raw")):
        assert before == after
