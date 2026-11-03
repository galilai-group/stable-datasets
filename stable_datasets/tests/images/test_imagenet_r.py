"""Tests for the ImageNet-R builder.

The first four tests run in CI with no download: they check the class lists, the
builder metadata, and the archive-reading logic on a tiny fake copy of the real
tar layout. The last test is marked ``large`` and checks the real 2.2 GB dataset.
"""

import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import ImageNetR
from stable_datasets.images._imagenet_wnids import IN1K_CLASSES, WNID_TO_IDX
from stable_datasets.images.imagenet_r import IMAGENET_R_CLASS_NAMES, IMAGENET_R_TO_IN1K, IMAGENET_R_WNIDS


def _jpeg_bytes(mode="RGB", icc_profile=None):
    """Return a tiny 8x8 JPEG image as bytes, optionally with an embedded color profile."""
    buff = io.BytesIO()
    Image.new(mode, (8, 8)).save(buff, format="JPEG", icc_profile=icc_profile)
    return buff.getvalue()


def _create_fake_imagenet_r_tar(path, entries):
    """Write a tar with the real ImageNet-R layout. ``entries`` maps name -> bytes (None = folder)."""
    with tarfile.open(path, "w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def _metadata_only_builder():
    """Build an ImageNetR instance without downloading or caching anything."""
    builder = object.__new__(ImageNetR)
    ImageNetR.__init__(builder)
    return builder


def test_class_lists_match_imagenet_1k():
    # 200 classes, each with a unique WordNet ID and a unique readable name
    assert len(IMAGENET_R_WNIDS) == 200
    assert len(set(IMAGENET_R_WNIDS)) == 200
    assert len(IMAGENET_R_CLASS_NAMES) == 200
    assert len(set(IMAGENET_R_CLASS_NAMES)) == 200

    # Same classes, in the same order, as the canonical ImageNet-1K list
    imagenet_r_wnids = set(IMAGENET_R_WNIDS)
    assert IMAGENET_R_WNIDS == [wnid for wnid in IN1K_CLASSES if wnid in imagenet_r_wnids]

    # Label i maps to the ImageNet-1K index of class i
    assert IMAGENET_R_TO_IN1K == [WNID_TO_IDX[wnid] for wnid in IMAGENET_R_WNIDS]
    assert (IMAGENET_R_CLASS_NAMES[0], IMAGENET_R_TO_IN1K[0]) == ("goldfish", 1)
    assert (IMAGENET_R_CLASS_NAMES[-1], IMAGENET_R_TO_IN1K[-1]) == ("acorn", 988)


def test_builder_metadata():
    builder = _metadata_only_builder()
    info = builder.info

    assert set(info.features.keys()) == {"image", "label"}
    assert info.features["label"].names == IMAGENET_R_CLASS_NAMES
    assert info.supervised_keys == ("image", "label")
    assert info.homepage == "https://github.com/hendrycks/imagenet-r"
    assert info.license == "MIT"
    assert "The Many Faces of Robustness" in info.citation

    # Test-only dataset: a single "test" asset, pinned by a SHA-256 checksum
    assets = ImageNetR.SOURCE["assets"]
    assert list(assets.keys()) == ["test"]
    assert assets["test"].checksum.startswith("sha256:")


def test_generate_examples_on_fake_archive(tmp_path, monkeypatch):
    tar_path = tmp_path / "imagenet-r.tar"
    _create_fake_imagenet_r_tar(
        tar_path,
        {
            "imagenet-r/": None,
            "imagenet-r/README.txt": b"not an image",
            # File names start with the search query used to collect the image (a style hint, not the class)
            "imagenet-r/n01443537/": None,
            "imagenet-r/n01443537/cartoon_12.jpg": _jpeg_bytes(),
            "imagenet-r/n01443537/sketch_0.jpg": _jpeg_bytes(mode="L"),
            "imagenet-r/n12267677/": None,
            "imagenet-r/n12267677/deviantart_22.jpg": _jpeg_bytes(),
            # Two real CMYK images carry a 1.8 MB color profile, over PIL's 1 MB limit for reading PNGs
            "imagenet-r/n12267677/misc_13.jpg": _jpeg_bytes(mode="CMYK", icc_profile=bytes(2 * 1024 * 1024)),
            # Not in the current release (all lowercase .jpg), but the builder accepts any case
            "imagenet-r/n12267677/toy_7.JPEG": _jpeg_bytes(),
        },
    )

    # The default split logic downloads through stable_datasets.utils.bulk_download
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda specs, *a, **k: [tar_path for _ in specs])

    # A separate cache folder, so the real ImageNet-R cache (same cache name) is never reused here
    ds = ImageNetR(split="test", download_dir=tmp_path / "downloads", processed_cache_dir=tmp_path / "processed")

    # Folders and README.txt are skipped; every image is kept
    assert len(ds) == 5

    samples = list(ds)
    for sample in samples:
        assert set(sample.keys()) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"

    # Labels come from the folder: n01443537 -> 0 (goldfish), n12267677 -> 199 (acorn)
    assert sorted(sample["label"] for sample in samples) == [0, 0, 199, 199, 199]

    # RGB images keep their original JPEG bytes; grayscale and CMYK are converted to RGB
    assert sorted(sample["image"].format for sample in samples) == ["JPEG", "JPEG", "JPEG", "PNG", "PNG"]

    # Converted images drop the old color profile (it describes the grayscale/CMYK colors, not RGB)
    assert all("icc_profile" not in sample["image"].info for sample in samples)


def test_unknown_class_folder_fails_loudly(tmp_path):
    tar_path = tmp_path / "imagenet-r.tar"
    _create_fake_imagenet_r_tar(tar_path, {"imagenet-r/n99999999/art_0.jpg": _jpeg_bytes()})

    builder = _metadata_only_builder()
    with pytest.raises(KeyError):
        list(builder._generate_examples(data_path=tar_path, split="test"))


def test_imagenet_r_real_dataset():
    # Downloads (2.2 GB) on first use, then loads from the processed cache
    ds = ImageNetR(split="test")

    # Test 1: the published size
    assert len(ds) == 30000, f"Expected 30000 samples, got {len(ds)}."

    # Test 2: one sample has the expected keys, an HxWx3 uint8 image and an integer label
    sample = ds[0]
    assert set(sample.keys()) == {"image", "label"}
    image_np = np.array(sample["image"])
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Expected an HxWx3 image, got {image_np.shape}."
    assert image_np.dtype == np.uint8
    assert isinstance(sample["label"], int)

    # Test 3: every image is RGB (about 4% are converted from grayscale or CMYK) and all 200 classes appear
    labels = []
    for sample in ds:
        assert sample["image"].mode == "RGB"
        labels.append(sample["label"])
    assert len(labels) == 30000
    assert set(labels) == set(range(200)), f"Expected all 200 classes, got {len(set(labels))}."

    # Test 4: loading without a split returns the single "test" split
    all_splits = ImageNetR(split=None)
    assert list(all_splits.keys()) == ["test"]
    assert len(all_splits["test"]) == 30000
