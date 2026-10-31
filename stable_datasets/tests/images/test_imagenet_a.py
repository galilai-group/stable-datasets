"""Tests for the ImageNet-A builder.

The first four tests run in CI with no download: they check the class lists, the
builder metadata, and the archive-reading logic on a tiny fake copy of the real
tar layout. The last test is marked ``large`` and checks the real 0.7 GB dataset.
"""

import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import ImageNetA
from stable_datasets.images._imagenet_wnids import IN1K_CLASSES, WNID_TO_IDX
from stable_datasets.images.imagenet_a import IMAGENET_A_CLASS_NAMES, IMAGENET_A_TO_IN1K, IMAGENET_A_WNIDS


def _jpeg_bytes(mode="RGB"):
    """Return a tiny 8x8 JPEG image as bytes."""
    buff = io.BytesIO()
    Image.new(mode, (8, 8)).save(buff, format="JPEG")
    return buff.getvalue()


def _create_fake_imagenet_a_tar(path, entries):
    """Write a tar with the real ImageNet-A layout. ``entries`` maps name -> bytes (None = folder)."""
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
    """Build an ImageNetA instance without downloading or caching anything."""
    builder = object.__new__(ImageNetA)
    ImageNetA.__init__(builder)
    return builder


def test_class_lists_match_imagenet_1k():
    # 200 classes, each with a unique WordNet ID and a unique readable name
    assert len(IMAGENET_A_WNIDS) == 200
    assert len(set(IMAGENET_A_WNIDS)) == 200
    assert len(IMAGENET_A_CLASS_NAMES) == 200
    assert len(set(IMAGENET_A_CLASS_NAMES)) == 200

    # Same classes, in the same order, as the canonical ImageNet-1K list
    imagenet_a_wnids = set(IMAGENET_A_WNIDS)
    assert IMAGENET_A_WNIDS == [wnid for wnid in IN1K_CLASSES if wnid in imagenet_a_wnids]

    # Label i maps to the ImageNet-1K index of class i
    assert IMAGENET_A_TO_IN1K == [WNID_TO_IDX[wnid] for wnid in IMAGENET_A_WNIDS]
    assert (IMAGENET_A_CLASS_NAMES[0], IMAGENET_A_TO_IN1K[0]) == ("stingray", 6)
    assert (IMAGENET_A_CLASS_NAMES[-1], IMAGENET_A_TO_IN1K[-1]) == ("acorn", 988)


def test_builder_metadata():
    builder = _metadata_only_builder()
    info = builder.info

    assert set(info.features.keys()) == {"image", "label"}
    assert info.features["label"].names == IMAGENET_A_CLASS_NAMES
    assert info.supervised_keys == ("image", "label")
    assert info.homepage == "https://github.com/hendrycks/natural-adv-examples"
    assert info.license == "MIT"
    assert "Natural Adversarial Examples" in info.citation

    # Test-only dataset: a single "test" asset, pinned by a SHA-256 checksum
    assets = ImageNetA.SOURCE["assets"]
    assert list(assets.keys()) == ["test"]
    assert assets["test"].checksum.startswith("sha256:")


def test_generate_examples_on_fake_archive(tmp_path, monkeypatch):
    tar_path = tmp_path / "imagenet-a.tar"
    _create_fake_imagenet_a_tar(
        tar_path,
        {
            "imagenet-a/": None,
            "imagenet-a/README.txt": b"not an image",
            "imagenet-a/n01498041/": None,
            # The file name says "balloon", but the folder (stingray) is the true class
            "imagenet-a/n01498041/0.003001_balloon _ balloon_0.4695473.jpg": _jpeg_bytes(),
            "imagenet-a/n01498041/0.004515_bear _ bear_0.8238403.JPEG": _jpeg_bytes(),
            "imagenet-a/n12267677/": None,
            "imagenet-a/n12267677/0.001987_unicycle _ unicycle_0.34870127.jpeg": _jpeg_bytes(),
            "imagenet-a/n12267677/0.002000_grayscale _ acorn_0.5.jpg": _jpeg_bytes(mode="L"),
        },
    )

    # The default split logic downloads through stable_datasets.utils.bulk_download
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda specs, *a, **k: [tar_path for _ in specs])

    # A separate cache folder, so the real ImageNet-A cache (same cache name) is never reused here
    ds = ImageNetA(split="test", download_dir=tmp_path / "downloads", processed_cache_dir=tmp_path / "processed")

    # Folders and README.txt are skipped; .jpg, .JPEG and .jpeg are all kept
    assert len(ds) == 4

    samples = list(ds)
    for sample in samples:
        assert set(sample.keys()) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"

    # Labels come from the folder: n01498041 -> 0 (stingray), n12267677 -> 199 (acorn)
    assert sorted(sample["label"] for sample in samples) == [0, 0, 199, 199]

    # RGB images keep their original JPEG bytes; the grayscale one is converted to RGB
    assert sorted(sample["image"].format for sample in samples) == ["JPEG", "JPEG", "JPEG", "PNG"]


def test_unknown_class_folder_fails_loudly(tmp_path):
    tar_path = tmp_path / "imagenet-a.tar"
    _create_fake_imagenet_a_tar(tar_path, {"imagenet-a/n99999999/image.jpg": _jpeg_bytes()})

    builder = _metadata_only_builder()
    with pytest.raises(KeyError):
        list(builder._generate_examples(data_path=tar_path, split="test"))


def test_imagenet_a_real_dataset():
    # Downloads (0.7 GB) on first use, then loads from the processed cache
    ds = ImageNetA(split="test")

    # Test 1: the published size
    assert len(ds) == 7500, f"Expected 7500 samples, got {len(ds)}."

    # Test 2: one sample has the expected keys, an HxWx3 uint8 image and an integer label
    sample = ds[0]
    assert set(sample.keys()) == {"image", "label"}
    image_np = np.array(sample["image"])
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Expected an HxWx3 image, got {image_np.shape}."
    assert image_np.dtype == np.uint8
    assert isinstance(sample["label"], int)

    # Test 3: every image is RGB and every one of the 200 classes appears
    labels = []
    for sample in ds:
        assert sample["image"].mode == "RGB"
        labels.append(sample["label"])
    assert len(labels) == 7500
    assert set(labels) == set(range(200)), f"Expected all 200 classes, got {len(set(labels))}."

    # Test 4: loading without a split returns the single "test" split
    all_splits = ImageNetA(split=None)
    assert list(all_splits.keys()) == ["test"]
    assert len(all_splits["test"]) == 7500
