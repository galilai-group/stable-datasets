"""Native TensorFlow Flowers tests and public-download integration coverage."""

import hashlib
import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from stable_datasets import StableDataset, StableDatasetDict, utils
from stable_datasets.images import tf_flowers
from stable_datasets.images.tf_flowers import TFFLOWERS_CLASS_NAMES, TFFlowers


LICENSE_HEADER = (
    "All images in this archive are licensed under the Creative Commons By-Attribution License, available at:\n"
    "https://creativecommons.org/licenses/by/2.0/\n"
    "The photographers are listed below, thanks to all of them for making their work available, "
    "and please be sure to credit them for any use as per the license."
)
ARCHIVE_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jpeg(pixels):
    buffer = io.BytesIO()
    PILImage.fromarray(pixels).save(buffer, format="JPEG", quality=93, subsampling=0)
    content = buffer.getvalue()
    with PILImage.open(io.BytesIO(content)) as image:
        return content, np.array(image.convert("RGB"))


def _license(credits):
    return (LICENSE_HEADER + "\n\n" + "\n".join(credits) + "\n").encode("utf-8")


def _write_tar(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = "flower_photos/roses/15922772266.jpg"
                archive.addfile(member)


@pytest.fixture
def release(tmp_path):
    # Preserve a conflicting duplicate under both original class labels, along
    # with spaces/case and the profile-link attribution format in nine real rows.
    image_ids = ["roses/15922772266.jpg", "tulips/15922772266.jpg", "daisy/Space Name.JPG", "dandelion/123_m.jpg"]
    pixels = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 5
    gray = np.arange(15, dtype=np.uint8).reshape(3, 5) * 13
    expected, credits = {}, []
    entries = [("flower_photos", b"", tarfile.DIRTYPE), ("flower_photos/roses/", b"", tarfile.DIRTYPE)]
    for index, image_id in enumerate(image_ids):
        content, decoded = _jpeg(gray if index == 3 else pixels)
        credit = (
            f"{image_id} CC-BY https://www.flickr.com/photos/example/ - by Photographer  Heart \u2665"
            if index == 3
            else f"{image_id} CC-BY by Photographer  Name - https://www.flickr.com/photos/example/{index + 1}/"
        )
        expected[image_id] = decoded
        credits.append(credit)
        entries.append(("flower_photos/" + image_id, content, tarfile.REGTYPE))
    # The real LICENSE is also in the middle, not before every photograph.
    entries.insert(4, ("flower_photos/LICENSE.txt", _license(credits), tarfile.REGTYPE))
    entries.append(("flower_photos/README.txt", b"not an image", tarfile.REGTYPE))
    archive_path = tmp_path / "flower_photos.tgz"
    _write_tar(archive_path, entries)
    return archive_path, entries, expected, dict(zip(image_ids, credits))


def _mock_download(monkeypatch, tmp_path, archive_path, *, checksum=None):
    # Only the raw download boundary and expected checksum change for tiny
    # fixtures. The real parser, generator, encoding and cache all run.
    monkeypatch.setattr(TFFlowers.SOURCE.assets["train"], "checksum", "sha256:" + (checksum or _sha256(archive_path)))

    def local_download(specs, dest_folder):
        assert Path(dest_folder) == tmp_path / "downloads"
        assert list(specs) == [TFFlowers.SOURCE.assets["train"]]
        return [archive_path]

    monkeypatch.setattr(utils, "bulk_download", local_download)


def _load(monkeypatch, tmp_path, archive_path, *, checksum=None, split="train"):
    _mock_download(monkeypatch, tmp_path, archive_path, checksum=checksum)
    return TFFlowers(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", split=split)


def _forbid_raw_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("warm TFFlowers cache must not download, hash, or reopen raw assets")

    monkeypatch.setattr(utils, "bulk_download", forbidden)
    monkeypatch.setattr(utils, "download", forbidden)
    monkeypatch.setattr(utils.requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(tf_flowers, "_verify_archive_checksum", forbidden)
    monkeypatch.setattr(tf_flowers.tarfile, "open", forbidden)
    monkeypatch.setattr(tf_flowers.gzip, "open", forbidden)


def test_metadata_does_not_access_raw_assets(monkeypatch):
    _forbid_raw_access(monkeypatch)
    builder = object.__new__(TFFlowers)
    builder.__init__()
    assert str(builder.VERSION) == "1.0.0"
    assert builder._candidate_splits() == ["train"]
    assert TFFLOWERS_CLASS_NAMES == ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
    assert builder.info.features["label"].names == TFFLOWERS_CLASS_NAMES
    assert set(builder.info.features) == {"image", "label", "image_id", "attribution"}
    assert builder.info.features["image"].encode_format == "PNG"
    assert builder.info.supervised_keys == ("image", "label")
    assert builder.info.license == LICENSE_HEADER
    assert set(builder.SOURCE.assets) == {"train"}
    asset = builder.SOURCE.assets["train"]
    assert asset.url == "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    assert asset.checksum == "sha256:" + ARCHIVE_SHA256


@pytest.mark.parametrize("kwargs", [{}, {"split": None}, {"split": "train"}])
def test_constructor_split_contract(monkeypatch, tmp_path, release, kwargs):
    archive_path, _, expected, _ = release
    _mock_download(monkeypatch, tmp_path, archive_path)
    result = TFFlowers(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", **kwargs)
    if kwargs.get("split") is None:
        assert isinstance(result, StableDatasetDict)
        assert set(result) == {"train"}
        result = result["train"]
    assert isinstance(result, StableDataset)
    assert len(result) == len(expected)


def test_pixels_attribution_original_labels_and_warm_cache(monkeypatch, tmp_path, release):
    import torch

    archive_path, _, expected, credits = release
    real_open, archive_modes = tarfile.open, []

    def counted_open(*args, **kwargs):
        archive_modes.append(kwargs.get("mode"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(tf_flowers.tarfile, "open", counted_open)
    datasets = _load(monkeypatch, tmp_path, archive_path, split=None)
    assert set(datasets) == {"train"}
    dataset = datasets["train"]
    assert len(dataset) == len(expected)
    assert archive_modes == ["r|", "r|gz"]
    assert [row["image_id"] for row in dataset] == list(expected)
    for index, row in enumerate(dataset):
        assert set(row) == {"image", "label", "image_id", "attribution"}
        assert row["label"] == TFFLOWERS_CLASS_NAMES.index(row["image_id"].split("/")[0])
        assert row["attribution"] == credits[row["image_id"]]
        assert row["image"].mode == "RGB"
        assert row["image"].size == (5, 3)
        assert row["image"].format == "PNG"
        pixels = expected[row["image_id"]]
        np.testing.assert_array_equal(np.array(row["image"]), pixels)
        assert dataset.with_format("raw")[index]["image"].startswith(b"\x89PNG\r\n\x1a\n")
        array = dataset.with_format("numpy")[index]["image"]
        assert array.dtype == np.uint8
        np.testing.assert_array_equal(array, pixels)
        tensor = dataset.with_format("torch")[index]["image"]
        assert tensor.dtype == torch.float32
        np.testing.assert_array_equal(tensor.numpy(), pixels.transpose(2, 0, 1).astype(np.float32) / 255)
    # Byte/pixel duplicates with different labels remain two distinct records.
    assert dataset[0]["label"] == 4 and dataset[1]["label"] == 2
    np.testing.assert_array_equal(np.array(dataset[0]["image"]), np.array(dataset[1]["image"]))
    assert "/example/ - by Photographer  Heart \u2665" in dataset[3]["attribution"]
    assert dataset.info.license == LICENSE_HEADER
    cache_files = {
        p: (p.stat().st_mtime_ns, p.stat().st_size) for p in (tmp_path / "processed").rglob("*") if p.is_file()
    }
    assert cache_files
    _forbid_raw_access(monkeypatch)
    for kwargs in ({}, {"split": "train"}, {"split": None}):
        warm = TFFlowers(processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads", **kwargs)
        if kwargs.get("split") is None:
            assert isinstance(warm, StableDatasetDict)
            assert set(warm) == {"train"}
            warm = warm["train"]
        assert isinstance(warm, StableDataset)
        assert list(warm.with_format("raw")) == list(dataset.with_format("raw"))
        assert warm.info.license == LICENSE_HEADER
    assert cache_files == {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in cache_files}


def test_single_optional_dot_archive_prefix(monkeypatch, tmp_path, release):
    archive_path, entries, expected, _ = release
    _write_tar(archive_path, [("./" + name, data, kind) for name, data, kind in entries])
    dataset = _load(monkeypatch, tmp_path, archive_path)
    assert [row["image_id"] for row in dataset] == list(expected)


@pytest.mark.parametrize("split", ["test", "validation", "archive"])
def test_unknown_split_uses_base_error_with_warm_cache(monkeypatch, tmp_path, release, split):
    _load(monkeypatch, tmp_path, release[0])
    _forbid_raw_access(monkeypatch)
    with pytest.raises(ValueError, match=f"Split '{split}' not found. Available: \\['train'\\]"):
        TFFlowers(split=split, processed_cache_dir=tmp_path / "processed", download_dir=tmp_path / "downloads")


def test_completed_raw_file_is_rehashed_before_parsing(monkeypatch, tmp_path, release):
    archive_path, _, _, _ = release
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        _load(monkeypatch, tmp_path, archive_path, checksum="0" * 64)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_public_source_uses_real_downloader_checksum(monkeypatch, tmp_path, release, valid_checksum):
    """Only HTTP and process scheduling are replaced; download and builder run."""
    archive_path, _, expected, credits = release
    content = archive_path.read_bytes()
    asset = TFFlowers.SOURCE.assets["train"]
    checksum = hashlib.sha256(content).hexdigest() if valid_checksum else "0" * 64
    monkeypatch.setattr(asset, "checksum", "sha256:" + checksum)
    requests_seen = []

    def http_response(session, method, url, **kwargs):
        requests_seen.append((method, url))
        assert url == asset.url
        response = utils.requests.Response()
        response.status_code = 200
        response.headers["content-length"] = str(len(content))
        response._content = content
        response._content_consumed = True
        response.url = url
        return response

    def sequential_download(specs, dest_folder):
        # A child process would not inherit the offline HTTP patch on Windows.
        assert list(specs) == [asset]
        return [utils.download(spec, dest_folder=dest_folder, progress_bar=False) for spec in specs]

    monkeypatch.setattr(utils.requests.sessions.Session, "request", http_response)
    monkeypatch.setattr(utils, "bulk_download", sequential_download)
    kwargs = {"split": "train", "download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    if valid_checksum:
        dataset = TFFlowers(**kwargs)
        assert len(dataset) == len(expected)
        for row in dataset:
            np.testing.assert_array_equal(np.array(row["image"]), expected[row["image_id"]])
            assert row["attribution"] == credits[row["image_id"]]
        completed = list(kwargs["download_dir"].glob("flower_photos.*.tgz"))
        assert len(completed) == 1 and completed[0].read_bytes() == content
    else:
        with pytest.raises(ValueError, match="Checksum mismatch"):
            TFFlowers(**kwargs)
        assert not list(kwargs["processed_cache_dir"].rglob("_metadata.json"))
        assert not list(kwargs["download_dir"].glob("*.tgz"))
        assert not list(kwargs["download_dir"].glob("*.tmp"))
    assert requests_seen == [("GET", asset.url)]


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("/flower_photos/roses/extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("flower_photos/../roses/extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("flower_photos/roses\\extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("flower_photos//roses/extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("flower_photos/roses/./extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("flower_photos/roses/C:extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("././flower_photos/roses/extra.jpg", tarfile.REGTYPE, "unsafe path"),
        ("other/roses/extra.jpg", tarfile.REGTYPE, "expected flower_photos archive prefix"),
        ("flower_photos/lily/extra.jpg", tarfile.REGTYPE, "unknown class"),
        ("flower_photos/Roses/extra.jpg", tarfile.REGTYPE, "unknown class"),
        ("flower_photos/roses/nested/extra.jpg", tarfile.REGTYPE, "expected class/image.jpg"),
        ("flower_photos/roses/extra.png", tarfile.REGTYPE, "expected class/image.jpg"),
        ("flower_photos/roses/15922772266.jpg", tarfile.REGTYPE, "duplicate canonical member"),
        ("./flower_photos/roses/15922772266.jpg", tarfile.REGTYPE, "duplicate canonical member"),
        ("flower_photos/LICENSE.txt", tarfile.REGTYPE, "duplicate canonical member"),
        ("flower_photos/roses/link.jpg", tarfile.SYMTYPE, "archive links"),
        ("flower_photos/roses/link.jpg", tarfile.LNKTYPE, "archive links"),
        ("flower_photos/roses/device.jpg", tarfile.CHRTYPE, "unsupported archive entry"),
        ("flower_photos/roses/nested", tarfile.DIRTYPE, "unsupported archive directory"),
    ],
)
def test_invalid_members_fail_before_cache_publication(monkeypatch, tmp_path, release, name, kind, message):
    archive_path, entries, _, _ = release
    _write_tar(archive_path, entries + [(name, b"invalid", kind)])
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_missing_license_is_an_error(monkeypatch, tmp_path, release):
    archive_path, entries, _, _ = release
    _write_tar(archive_path, [row for row in entries if row[0] != "flower_photos/LICENSE.txt"])
    with pytest.raises(FileNotFoundError, match="missing flower_photos/LICENSE.txt"):
        _load(monkeypatch, tmp_path, archive_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "missing attribution"),
        ("orphan", "attribution without a photo"),
        ("duplicate", "duplicate attribution"),
        ("wrong_case", "missing attribution"),
        ("malformed", "invalid photographer attribution"),
        ("bad_path", "unsafe path"),
        ("header", "unexpected license declaration"),
        ("encoding", "not UTF-8"),
    ],
)
def test_invalid_attribution_fails(monkeypatch, tmp_path, release, change, message):
    archive_path, entries, _, credits = release
    lines = list(credits.values())
    if change == "missing":
        lines.pop()
    elif change == "orphan":
        lines.append("daisy/missing.jpg CC-BY by Author - https://www.flickr.com/photos/example/99/")
    elif change == "duplicate":
        lines.append(lines[0])
    elif change == "wrong_case":
        lines[2] = lines[2].replace("Space Name.JPG", "space name.JPG")
    elif change == "malformed":
        lines[0] = "roses/15922772266.jpg CC-BY nobody"
    elif change == "bad_path":
        lines[0] = "../" + lines[0]
    content = _license(lines)
    if change == "header":
        content = content.replace(b"by/2.0", b"by/4.0")
    elif change == "encoding":
        content += b"\xff"
    _write_tar(
        archive_path,
        [(name, content if name == "flower_photos/LICENSE.txt" else data, kind) for name, data, kind in entries],
    )
    with pytest.raises(ValueError, match=message):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_corrupt_photo_does_not_silently_shrink_collection(monkeypatch, tmp_path, release):
    archive_path, entries, _, _ = release
    _write_tar(
        archive_path,
        [(name, b"broken" if name.endswith("daisy/Space Name.JPG") else data, kind) for name, data, kind in entries],
    )
    with pytest.raises(ValueError, match="cannot decode image 'daisy/Space Name.JPG'"):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_gzip_trailer_is_checked_after_tar_end(monkeypatch, tmp_path, release):
    archive_path, _, _, _ = release
    archive_path.write_bytes(archive_path.read_bytes()[:-4])
    # The fixture checksum is updated so this exercises decompression integrity.
    with pytest.raises((EOFError, OSError, tarfile.TarError)):
        _load(monkeypatch, tmp_path, archive_path)
    assert not list((tmp_path / "processed").rglob("_metadata.json"))


def test_empty_collection_fails(monkeypatch, tmp_path):
    archive_path = tmp_path / "flower_photos.tgz"
    _write_tar(archive_path, [("flower_photos/LICENSE.txt", _license([]), tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="contains no photos"):
        _load(monkeypatch, tmp_path, archive_path)


def test_tf_flowers_public_release(monkeypatch, tmp_path):
    """Download the public release through the builder and check every record."""
    download_dir = tmp_path / "downloads"
    processed_cache_dir = tmp_path / "processed"
    assert not download_dir.exists() and not processed_cache_dir.exists()
    datasets = TFFlowers(processed_cache_dir=processed_cache_dir, download_dir=download_dir)
    assert isinstance(datasets, StableDatasetDict)
    assert set(datasets) == {"train"}
    dataset = datasets["train"]
    assert isinstance(dataset, StableDataset)
    assert len(dataset) == 3670
    archives = list(download_dir.glob("*.tgz"))
    assert len(archives) == 1
    archive_path = archives[0]
    assert archive_path.is_file() and archive_path.stat().st_size == 228813984
    assert _sha256(archive_path) == ARCHIVE_SHA256
    rows = {row["image_id"]: index for index, row in enumerate(dataset.with_format("raw"))}
    assert len(rows) == 3670
    source_ids, counts, credits = [], Counter(), {}
    # Independent source comparison: no builder parsing helper supplies the
    # expected pixels, original labels, or attribution text in this audit.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name == "flower_photos/LICENSE.txt":
                lines = archive.extractfile(member).read().decode("utf-8").splitlines()
                assert "\n".join(lines[:3]) == LICENSE_HEADER
                credits = {line.split(" CC-BY ", 1)[0]: line for line in lines[3:] if line}
            elif member.isfile() and member.name.endswith(".jpg"):
                image_id = member.name.removeprefix("flower_photos/")
                source_ids.append(image_id)
                row = dataset[rows[image_id]]
                assert set(row) == {"image", "label", "image_id", "attribution"}
                assert isinstance(row["image_id"], str)
                assert isinstance(row["attribution"], str)
                class_name = image_id.split("/")[0]
                counts[class_name] += 1
                assert row["label"] == TFFLOWERS_CLASS_NAMES.index(class_name)
                assert isinstance(row["label"], int) and 0 <= row["label"] < 5
                assert isinstance(row["image"], PILImage.Image)
                assert row["image"].mode == "RGB"
                with PILImage.open(io.BytesIO(archive.extractfile(member).read())) as source:
                    assert row["image"].size == source.size
                    np.testing.assert_array_equal(np.array(row["image"]), np.array(source.convert("RGB")))
                assert dataset.with_format("raw")[rows[image_id]]["image"].startswith(b"\x89PNG\r\n\x1a\n")
    assert source_ids == list(rows)
    assert set(credits) == set(rows)
    assert counts == dict(zip(TFFLOWERS_CLASS_NAMES, [898, 633, 799, 699, 641]))
    assert sum(" CC-BY https://" in line for line in credits.values()) == 9
    for row in dataset.with_format("raw"):
        assert row["attribution"] == credits[row["image_id"]]
    conflict = [dataset[rows[f"{cls}/15922772266_1167a06620.jpg"]] for cls in ("roses", "tulips")]
    assert [row["label"] for row in conflict] == [4, 2]
    np.testing.assert_array_equal(np.array(conflict[0]["image"]), np.array(conflict[1]["image"]))
    _forbid_raw_access(monkeypatch)
    warm = TFFlowers(split="train", processed_cache_dir=processed_cache_dir, download_dir=download_dir)
    assert len(warm) == 3670
    for original, cached in zip(dataset.with_format("raw"), warm.with_format("raw")):
        assert original == cached
