import hashlib
import io
import tarfile
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Caltech101
from stable_datasets.images.caltech101 import BACKGROUND_CLASS, TRAIN_IMAGES_PER_CLASS


# Offline tests: a tiny synthetic archive with the same layout as the real caltech-101.zip.

FAKE_CLASS_SIZES = {"Faces": 31, "Leopards": 33, "accordion": 35}
FAKE_BACKGROUND_SIZE = 4
GRAYSCALE_FILE = ("accordion", "image_0007.jpg")


def _pixel_hash(image):
    arr = np.asarray(image)
    return hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _jpeg(rng, mode):
    shape = (12, 16, 3) if mode == "RGB" else (12, 16)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8), mode=mode).save(buf, format="JPEG")
    return buf.getvalue()


def _add_tar_file(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_fake_archive(path):
    """Write a fake caltech-101.zip and return ``{(class, file): rgb_pixel_hash}`` for every image in it."""
    rng = np.random.default_rng(0)
    sizes = {**FAKE_CLASS_SIZES, BACKGROUND_CLASS: FAKE_BACKGROUND_SIZE}
    hashes = {}

    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as tar:
        for class_name, size in sizes.items():
            for i in range(1, size + 1):
                file_name = f"image_{i:04d}.jpg"
                data = _jpeg(rng, "L" if (class_name, file_name) == GRAYSCALE_FILE else "RGB")
                _add_tar_file(tar, f"101_ObjectCategories/{class_name}/{file_name}", data)
                hashes[(class_name, file_name)] = _pixel_hash(Image.open(io.BytesIO(data)).convert("RGB"))
        # Junk present in (or typical of) the real archive that the builder must skip.
        _add_tar_file(tar, f"101_ObjectCategories/{BACKGROUND_CLASS}/tmp", b"not an image")
        _add_tar_file(tar, "101_ObjectCategories/Faces/._image_0001.jpg", b"resource fork")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("caltech-101/", b"")
        zf.writestr("__MACOSX/._caltech-101", b"junk")
        zf.writestr("caltech-101/101_ObjectCategories.tar.gz", inner.getvalue())
        # Same suffix as the real inner tar; the builder must not pick this one.
        zf.writestr("__MACOSX/caltech-101/._101_ObjectCategories.tar.gz", b"junk")
        zf.writestr("caltech-101/Annotations.tar", b"")
    return hashes


def _patch_download(monkeypatch, archive_path):
    """Serve ``archive_path`` instead of downloading; return the list of requested DownloadInfos."""
    requested = []

    def fake_bulk_download(infos, dest_folder):
        infos = list(infos)
        requested.extend(infos)
        return [archive_path] * len(infos)

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    monkeypatch.setattr(Caltech101, "_labels", staticmethod(lambda: sorted(FAKE_CLASS_SIZES)))
    return requested


def _rows(ds):
    """Return ``[(pixel_hash, class_name, mode, size)]`` for every row of a split."""
    names = ds.features["label"].names
    rows = []
    for i in range(len(ds)):
        sample = ds[i]
        image = sample["image"]
        rows.append((_pixel_hash(image), names[sample["label"]], image.mode, image.size))
    return rows


@pytest.fixture(scope="module")
def fake_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("caltech101")
    archive_path = root / "caltech-101.zip"
    hashes = _write_fake_archive(archive_path)
    with pytest.MonkeyPatch.context() as mp:
        requested = _patch_download(mp, archive_path)
        ds = Caltech101(split=None, download_dir=root / "downloads", processed_cache_dir=root / "processed")
        rows = {split: _rows(ds[split]) for split in ("train", "test")}
    return {"ds": ds, "rows": rows, "hashes": hashes, "requested": requested, "archive_path": archive_path}


def test_fake_split_sizes(fake_build):
    rows = fake_build["rows"]
    assert len(rows["train"]) == TRAIN_IMAGES_PER_CLASS * len(FAKE_CLASS_SIZES)
    assert len(rows["test"]) == sum(FAKE_CLASS_SIZES.values()) - TRAIN_IMAGES_PER_CLASS * len(FAKE_CLASS_SIZES)


def test_fake_train_has_30_per_class_and_test_covers_every_class(fake_build):
    train_counts = Counter(name for _, name, _, _ in fake_build["rows"]["train"])
    test_counts = Counter(name for _, name, _, _ in fake_build["rows"]["test"])
    assert train_counts == dict.fromkeys(FAKE_CLASS_SIZES, TRAIN_IMAGES_PER_CLASS)
    assert test_counts == {name: size - TRAIN_IMAGES_PER_CLASS for name, size in FAKE_CLASS_SIZES.items()}


def test_fake_background_class_is_excluded(fake_build):
    background = {h for (class_name, _), h in fake_build["hashes"].items() if class_name == BACKGROUND_CLASS}
    seen = {h for split_rows in fake_build["rows"].values() for h, _, _, _ in split_rows}
    assert len(background) == FAKE_BACKGROUND_SIZE
    assert not background & seen


def test_fake_splits_are_disjoint_and_cover_all_images(fake_build):
    train = [h for h, _, _, _ in fake_build["rows"]["train"]]
    test = [h for h, _, _, _ in fake_build["rows"]["test"]]
    expected = {h for (class_name, _), h in fake_build["hashes"].items() if class_name != BACKGROUND_CLASS}
    assert len(set(train)) == len(train) and len(set(test)) == len(test)
    assert not set(train) & set(test)
    assert set(train) | set(test) == expected


def test_fake_labels_match_source_class(fake_build):
    class_of_hash = {h: class_name for (class_name, _), h in fake_build["hashes"].items()}
    for split_rows in fake_build["rows"].values():
        for h, name, _, _ in split_rows:
            assert class_of_hash[h] == name


def test_fake_images_are_rgb_including_grayscale_source(fake_build):
    grayscale_hash = fake_build["hashes"][GRAYSCALE_FILE]
    all_rows = [row for split_rows in fake_build["rows"].values() for row in split_rows]
    assert all(mode == "RGB" and size == (16, 12) for _, _, mode, size in all_rows)
    assert grayscale_hash in {h for h, _, _, _ in all_rows}


def test_fake_split_matches_seeded_protocol(fake_build):
    expected = Caltech101._split_files(k for k in fake_build["hashes"] if k[0] != BACKGROUND_CLASS)
    for split in ("train", "test"):
        expected_hashes = [fake_build["hashes"][key] for key in expected[split]]
        assert [h for h, _, _, _ in fake_build["rows"][split]] == expected_hashes


def test_fake_build_is_deterministic(fake_build, tmp_path, monkeypatch):
    _patch_download(monkeypatch, fake_build["archive_path"])
    rebuilt = Caltech101(split=None, download_dir=tmp_path / "downloads", processed_cache_dir=tmp_path / "processed")
    for split in ("train", "test"):
        assert _rows(rebuilt[split]) == fake_build["rows"][split]


def test_fake_shared_archive_is_requested_once_with_checksum(fake_build):
    (info,) = fake_build["requested"]
    assert info.filename == "caltech-101.zip"
    assert info.url.startswith("https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip")
    assert info.checksum == "sha256:331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91"


def test_fake_torch_format(fake_build):
    image = fake_build["ds"]["train"].with_format("torch")[0]["image"]
    assert tuple(image.shape) == (3, 12, 16)


# Split protocol, without any archive.


def _keys(sizes):
    return [(c, f"image_{i:04d}.jpg") for c, n in sizes.items() for i in range(1, n + 1)]


def test_split_files_is_independent_of_key_order():
    keys = _keys({"b": 40, "a": 31, "c": 100})
    assert Caltech101._split_files(keys) == Caltech101._split_files(reversed(keys))


def test_split_files_is_pinned():
    # Guards against silently changing the published split (seed, sort order, or shuffle procedure).
    splits = Caltech101._split_files(_keys({"a": 35}))
    assert splits["test"] == [("a", f"image_{i:04d}.jpg") for i in (3, 17, 25, 27, 31)]


def test_split_files_rejects_classes_without_test_images():
    with pytest.raises(ValueError, match="need more than 30"):
        Caltech101._split_files(_keys({"a": 31, "b": TRAIN_IMAGES_PER_CLASS}))


def test_inner_tar_must_exist(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as zf:
        zf.writestr("__MACOSX/caltech-101/._101_ObjectCategories.tar.gz", b"junk")
    with zipfile.ZipFile(tmp_path / "bad.zip") as zf, pytest.raises(ValueError, match="exactly one"):
        Caltech101._inner_tar_name(zf)


def test_metadata():
    labels = Caltech101._labels()
    assert len(labels) == 101
    assert labels == sorted(set(labels))
    assert BACKGROUND_CLASS not in labels
    assert {"Faces", "Faces_easy"} <= set(labels)

    assets = Caltech101.SOURCE["assets"]
    assert set(assets) == {"train", "test"}
    assert assets["train"] == assets["test"]


# Real download (network, ~140 MB).

EXPECTED_NUM_TRAIN = 3030
EXPECTED_NUM_TEST = 5647


@pytest.mark.large
def test_caltech101_dataset():
    caltech_train = Caltech101(split="train")

    # Test 1: Check number of training samples
    assert len(caltech_train) == EXPECTED_NUM_TRAIN, (
        f"Expected {EXPECTED_NUM_TRAIN} training samples, got {len(caltech_train)}."
    )

    # Test 2: Check sample keys
    sample = caltech_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type, dtype and channels
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Caltech-101 images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Caltech-101 images should have 3 channels, got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 101, f"Label should be in range [0, 100], got {label}."
    assert len(caltech_train.features["label"].names) == 101

    # Test 5: Every class has exactly 30 training images
    train_counts = Counter(caltech_train.table.column("label").to_pylist())
    assert train_counts == dict.fromkeys(range(101), TRAIN_IMAGES_PER_CLASS)

    # Test 6: Load and validate test split; every class keeps at least one test image
    caltech_test = Caltech101(split="test")
    assert len(caltech_test) == EXPECTED_NUM_TEST, (
        f"Expected {EXPECTED_NUM_TEST} test samples, got {len(caltech_test)}."
    )
    assert set(caltech_test.table.column("label").to_pylist()) == set(range(101))

    # Test 7: Grayscale sources are converted, so every image decodes to RGB
    assert all(caltech_test[i]["image"].mode == "RGB" for i in range(0, len(caltech_test), 50))

    # Test 8: Torch formatting yields CHW tensors
    torch_image = caltech_train.with_format("torch")[0]["image"]
    assert torch_image.ndim == 3 and torch_image.shape[0] == 3, f"Expected (3, H, W), got {tuple(torch_image.shape)}."
