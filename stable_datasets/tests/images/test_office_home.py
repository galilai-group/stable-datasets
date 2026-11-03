"""Tests for Office-Home.

The offline tests build a small fake release archive to check the split logic. The other
tests download the real data (~1.1 GB).
"""

import io
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

import stable_datasets.images.office_home as office_home_module
from stable_datasets.images import OfficeHome
from stable_datasets.images._multi_domain import num_test_examples


DOMAINS = ["art", "clipart", "product", "real_world"]
FOLDERS = ["Art", "Clipart", "Product", "Real World"]  # as named in the release archive
EXPECTED_PER_DOMAIN = {"art": 2427, "clipart": 4365, "product": 4439, "real_world": 4357}
TOTAL = sum(EXPECTED_PER_DOMAIN.values())  # 15,588


def _counts(ds, column):
    raw = ds.with_format("raw")
    return Counter(raw[i][column] for i in range(len(raw)))


def _pixels(ds):
    return Counter(np.asarray(ds[i]["image"]).tobytes() for i in range(len(ds)))


# ---- offline tests on a fake release --------------------------------------------------

FAKE_FOLDERS = ["Alarm_Clock", "Backpack", "ToothBrush"]  # class folders as spelled in the archive
FAKE_LABELS = [0, 1, 61]
PER_CELL = 6


def _jpeg(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _write_archive(path, order_seed=None):
    entries = [
        (f"OfficeHomeDataset_10072016/{folder}/{cls}/{k:05d}.jpg", 1000 * d + 100 * c + k)
        for d, folder in enumerate(FOLDERS)
        for c, cls in enumerate(FAKE_FOLDERS)
        for k in range(PER_CELL)
    ]
    if order_seed is not None:
        entries = [entries[i] for i in np.random.default_rng(order_seed).permutation(len(entries))]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("OfficeHomeDataset_10072016/ImageInfo.csv", "image,artist\n")
        for name, seed in entries:
            archive.writestr(name, _jpeg(seed))
    return path


@pytest.fixture
def fake_office_home(tmp_path, monkeypatch):
    """Return ``load(split, config_name=None, order_seed=None)`` backed by a fake release archive."""

    def load(split, config_name=None, order_seed=None):
        archive = _write_archive(tmp_path / f"office_home_{order_seed}.zip", order_seed)
        monkeypatch.setattr(office_home_module, "bulk_download", lambda infos, dest_folder: [archive])
        cache = tmp_path / f"cache_{config_name}_{order_seed}"
        return OfficeHome(split=split, config_name=config_name, processed_cache_dir=str(cache))

    return load


def test_configs_and_source():
    assert [c.name for c in OfficeHome.BUILDER_CONFIGS] == ["random", *DOMAINS]
    assert OfficeHome.DEFAULT_CONFIG_NAME == "random"
    assert len(OfficeHome.CLASS_NAMES) == len(set(OfficeHome.CLASS_NAMES)) == 65
    assert OfficeHome.SOURCE["assets"]["archive"].url.startswith("https://drive.usercontent.google.com/")


def test_random_split_is_stratified(fake_office_home):
    train, test = fake_office_home("train"), fake_office_home("test")
    n_test = num_test_examples(PER_CELL, 0.2)
    cells = len(DOMAINS) * len(FAKE_FOLDERS)
    assert len(test) == cells * n_test
    assert len(train) == cells * (PER_CELL - n_test)

    raw = test.with_format("raw")
    per_cell = Counter((raw[i]["domain"], raw[i]["label"]) for i in range(len(raw)))
    assert set(per_cell.values()) == {n_test}
    assert {label for _, label in per_cell} == set(FAKE_LABELS)  # folder names map to the right classes
    assert not set(_pixels(train)) & set(_pixels(test))

    sample = train[0]
    assert set(sample) == {"image", "label", "domain"}
    assert isinstance(sample["image"], Image.Image) and sample["image"].mode == "RGB"


def test_random_split_ignores_archive_order(fake_office_home):
    assert _pixels(fake_office_home("test", order_seed=1)) == _pixels(fake_office_home("test", order_seed=2))


@pytest.mark.parametrize("held_out", DOMAINS)
def test_leave_one_domain_out(fake_office_home, held_out):
    h = DOMAINS.index(held_out)
    train, test = fake_office_home("train", held_out), fake_office_home("test", held_out)
    assert len(test) == len(FAKE_FOLDERS) * PER_CELL
    assert len(train) == 3 * len(FAKE_FOLDERS) * PER_CELL
    assert set(_counts(test, "domain")) == {h}
    assert h not in _counts(train, "domain")


# ---- real data -------------------------------------------------------------------------


def test_office_home_random_split():
    train, test = OfficeHome(split="train"), OfficeHome(split="test")
    assert len(train) + len(test) == TOTAL
    assert 0.19 < len(test) / TOTAL < 0.21

    names = train.features["domain"].names
    per_domain = _counts(train, "domain") + _counts(test, "domain")
    assert {names[d]: n for d, n in per_domain.items()} == EXPECTED_PER_DOMAIN
    assert set(_counts(train, "label")) == set(range(65))
    assert set(_counts(test, "label")) == set(range(65))

    sample = train[0]
    assert set(sample) == {"image", "label", "domain"}
    assert isinstance(sample["image"], Image.Image) and sample["image"].mode == "RGB"


def test_office_home_leave_out_clipart():
    train = OfficeHome(config_name="clipart", split="train")
    test = OfficeHome(config_name="clipart", split="test")
    assert len(test) == EXPECTED_PER_DOMAIN["clipart"]
    assert len(train) == TOTAL - EXPECTED_PER_DOMAIN["clipart"]
