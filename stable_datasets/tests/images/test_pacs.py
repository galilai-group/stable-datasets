"""Tests for PACS.

The offline tests build a small fake release (an image zip and the official split files)
to check the split logic. The other tests download the real data (~200 MB).
"""

import io
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

import stable_datasets.images.pacs as pacs_module
from stable_datasets.images import PACS


DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
EXPECTED_PER_DOMAIN = {"photo": 1670, "art_painting": 2048, "cartoon": 2344, "sketch": 3929}
TOTAL = sum(EXPECTED_PER_DOMAIN.values())  # 9,991


def _counts(ds, column):
    raw = ds.with_format("raw")
    return Counter(raw[i][column] for i in range(len(raw)))


# ---- offline tests on a fake release --------------------------------------------------

FAKE_CLASSES = 3  # dog, elephant, giraffe
FAKE_TRAIN, FAKE_VAL = 4, 1  # images per (domain, class) in the official train / crossval files


def _png(seed: int, mode: str) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").convert(mode).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_pacs(tmp_path, monkeypatch):
    """Write a fake PACS.zip and split files and point the builder's downloads at them."""
    files = {}
    with zipfile.ZipFile(tmp_path / "PACS.zip", "w") as archive:
        for d, domain in enumerate(DOMAINS):
            lines = {"train": [], "crossval": []}
            for c in range(FAKE_CLASSES):
                class_name = PACS.CLASS_NAMES[c]
                for k in range(FAKE_TRAIN + FAKE_VAL):
                    # Sketches are transparent PNGs, like in the real release.
                    ext, mode = (".png", "RGBA") if domain == "sketch" else (".jpg", "RGB")
                    rel = f"{domain}/{class_name}/pic_{k:03d}{ext}"
                    archive.writestr(f"kfold/{rel}", _png(1000 * d + 10 * c + k, mode))
                    lines["train" if k < FAKE_TRAIN else "crossval"].append(f"{rel} {c + 1}")
            lines["test"] = lines["train"] + lines["crossval"]
            for part, part_lines in lines.items():
                path = tmp_path / f"pacs_{domain}_{part}_kfold.txt"
                path.write_text("\n".join(part_lines) + "\n")
                files[path.name] = path
    files["PACS.zip"] = tmp_path / "PACS.zip"
    monkeypatch.setattr(pacs_module, "bulk_download", lambda infos, dest_folder: [files[i.filename] for i in infos])

    def load(split, config_name=None):
        return PACS(split=split, config_name=config_name, processed_cache_dir=str(tmp_path / "cache"))

    return load


def test_configs_and_sources():
    assert [c.name for c in PACS.BUILDER_CONFIGS] == ["all", *DOMAINS]
    assert PACS.DEFAULT_CONFIG_NAME == "all"
    assets = PACS.SOURCE["assets"]
    assert len(assets) == 1 + 3 * len(DOMAINS)  # the images and 3 split files per domain
    assert all(a.url.startswith("https://drive.usercontent.google.com/") for a in assets.values())


def test_all_config_uses_official_splits(fake_pacs):
    train, val = fake_pacs("train"), fake_pacs("validation")
    per_domain_classes = len(DOMAINS) * FAKE_CLASSES
    assert len(train) == per_domain_classes * FAKE_TRAIN
    assert len(val) == per_domain_classes * FAKE_VAL
    assert set(_counts(train, "domain")) == set(range(len(DOMAINS)))
    assert set(_counts(val, "label")) == set(range(FAKE_CLASSES))
    with pytest.raises(ValueError):
        fake_pacs("test")  # no test split without a held-out domain

    sample = train[0]
    assert set(sample) == {"image", "label", "domain"}
    assert isinstance(sample["image"], Image.Image) and sample["image"].mode == "RGB"


def test_transparent_sketches_become_white_rgb(fake_pacs):
    train = fake_pacs("train")
    sketch = DOMAINS.index("sketch")
    sample = next(train[i] for i in range(len(train)) if train[i]["domain"] == sketch)
    assert sample["image"].mode == "RGB"


@pytest.mark.parametrize("held_out", DOMAINS)
def test_leave_one_domain_out(fake_pacs, held_out):
    h = DOMAINS.index(held_out)
    train = fake_pacs("train", held_out)
    val = fake_pacs("validation", held_out)
    test = fake_pacs("test", held_out)

    assert len(train) == 3 * FAKE_CLASSES * FAKE_TRAIN
    assert len(val) == 3 * FAKE_CLASSES * FAKE_VAL  # official crossval images of the source domains
    assert len(test) == FAKE_CLASSES * (FAKE_TRAIN + FAKE_VAL)  # every image of the held-out domain
    assert h not in _counts(train, "domain") and h not in _counts(val, "domain")
    assert set(_counts(test, "domain")) == {h}


def test_mislabeled_split_file_raises(tmp_path, fake_pacs):
    path = tmp_path / "pacs_photo_train_kfold.txt"
    path.write_text(path.read_text().replace(" 1\n", " 2\n", 1))
    with pytest.raises(ValueError, match="labels"):
        fake_pacs("train")


# ---- real data -------------------------------------------------------------------------


def test_pacs_all_domains():
    train, val = PACS(split="train"), PACS(split="validation")
    assert len(train) + len(val) == TOTAL

    names = train.features["domain"].names
    per_domain = _counts(train, "domain") + _counts(val, "domain")
    assert {names[d]: n for d, n in per_domain.items()} == EXPECTED_PER_DOMAIN
    art = names.index("art_painting")
    assert (_counts(train, "domain")[art], _counts(val, "domain")[art]) == (1840, 208)

    assert set(_counts(train, "label")) == set(range(7))
    sample = train[0]
    assert set(sample) == {"image", "label", "domain"}
    assert isinstance(sample["image"], Image.Image) and sample["image"].mode == "RGB"
    assert sample["image"].size == (227, 227)


def test_pacs_leave_out_sketch():
    train = PACS(config_name="sketch", split="train")
    val = PACS(config_name="sketch", split="validation")
    test = PACS(config_name="sketch", split="test")
    assert len(test) == EXPECTED_PER_DOMAIN["sketch"]
    assert len(train) + len(val) == TOTAL - EXPECTED_PER_DOMAIN["sketch"]
    sketch = test.features["domain"].names.index("sketch")
    assert set(_counts(test, "domain")) == {sketch}
    assert sketch not in _counts(train, "domain")
