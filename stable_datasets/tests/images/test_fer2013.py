import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import FER2013
from stable_datasets.images import fer2013 as fer_module


# ---------------------------------------------------------------------------
# Offline unit tests (synthetic Kaggle-format CSVs, no network)
# ---------------------------------------------------------------------------

_USAGES = {"Training": 12, "PublicTest": 5, "PrivateTest": 4}


def _rows(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for usage, n in _USAGES.items():
        for _ in range(n):
            pixels = rng.integers(0, 256, 48 * 48, dtype=np.uint8)
            rows.append((int(rng.integers(7)), " ".join(map(str, pixels)), usage, pixels.reshape(48, 48)))
    return rows


def _write_csv(path: Path, rows, icml=False):
    with open(path, "w", newline="") as f:
        if icml:  # icml_face_data.csv: different column order and spaces in the header
            f.write("emotion, Usage, pixels\n")
            f.writelines(f"{e},{u},{p}\n" for e, p, u, _ in rows)
        else:
            f.write("emotion,pixels,Usage\n")
            f.writelines(f"{e},{p},{u}\n" for e, p, u, _ in rows)


def _accept(monkeypatch, path: Path):
    """Make the builder accept the synthetic file by registering its MD5."""
    monkeypatch.setitem(fer_module._CSV_MD5, path.name, fer_module._md5(path))


def _check_build(tmp_path, rows):
    ds = FER2013(download_dir=tmp_path / "dl", processed_cache_dir=tmp_path / "proc")
    assert {k: len(v) for k, v in ds.items()} == {"train": 12, "validation": 5, "test": 4}
    assert ds["train"].info.features["label"].names == [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]
    expected = {"train": rows[:12], "validation": rows[12:17], "test": rows[17:]}
    for split, split_rows in expected.items():
        for i, (emotion, _, _, pixels) in enumerate(split_rows):
            sample = ds[split][i]
            assert sample["label"] == emotion
            assert isinstance(sample["image"], Image.Image)
            assert np.array_equal(np.array(sample["image"].convert("L")), pixels)


def test_fer2013_csv(tmp_path, monkeypatch):
    rows = _rows()
    (tmp_path / "dl").mkdir()
    _write_csv(tmp_path / "dl" / "fer2013.csv", rows)
    _accept(monkeypatch, tmp_path / "dl" / "fer2013.csv")
    _check_build(tmp_path, rows)


def test_icml_face_data_csv(tmp_path, monkeypatch):
    rows = _rows(seed=1)
    (tmp_path / "dl").mkdir()
    _write_csv(tmp_path / "dl" / "icml_face_data.csv", rows, icml=True)
    _accept(monkeypatch, tmp_path / "dl" / "icml_face_data.csv")
    _check_build(tmp_path, rows)


def test_fer2013_tarball_is_extracted(tmp_path, monkeypatch):
    rows = _rows(seed=2)
    src = tmp_path / "fer2013.csv"
    _write_csv(src, rows)
    _accept(monkeypatch, src)
    (tmp_path / "dl").mkdir()
    with tarfile.open(tmp_path / "dl" / "fer2013.tar.gz", "w:gz") as archive:
        archive.add(src, arcname="fer2013/fer2013.csv")
    _check_build(tmp_path, rows)


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        fer_module._find_csv(tmp_path)
    (tmp_path / "fer2013.csv").write_bytes(b"emotion,pixels,Usage\n")
    with pytest.raises(RuntimeError, match="checksum"):
        fer_module._find_csv(tmp_path)


# ---------------------------------------------------------------------------
# Real data (manual Kaggle download), excluded from CI like other large datasets
# ---------------------------------------------------------------------------


@pytest.mark.large
def test_fer2013_dataset():
    ds = FER2013()
    assert {k: len(v) for k, v in ds.items()} == {"train": 28709, "validation": 3589, "test": 3589}

    sample = ds["train"][0]
    assert np.array(sample["image"].convert("L")).shape == (48, 48)
    assert 0 <= sample["label"] < 7
