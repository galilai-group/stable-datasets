"""Tests for the Sewer-ML builder.

The real dataset is ~340 GB behind a password-protected share, so the behavioural
tests run against a synthetic fixture with the same on-disk shape: annotation
CSVs plus image archives, with each split's images spread across several zips.
"""

import csv
import io
import zipfile

import pytest
from PIL import Image as PILImage

from stable_datasets.images import SewerML
from stable_datasets.images.sewer_ml import _BASE_URL, _SPLIT_LAYOUT, SEWER_ML_DEFECT_CLASSES, _is_forbidden


def _png_bytes(seed: int) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), (seed % 255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sewer_ml_tree(tmp_path):
    """Build a miniature Sewer-ML directory mirroring the real layout."""
    root = tmp_path / "sewer-ml"
    root.mkdir()

    for split, (csv_name, archive_names) in _SPLIT_LAYOUT.items():
        labeled = split != "test"
        filenames = [f"{split}_{i:04d}.png" for i in range(6)]

        # Spread images across the split's archives so multi-archive indexing is exercised.
        for archive_index, archive_name in enumerate(archive_names):
            with zipfile.ZipFile(root / archive_name, "w") as archive:
                for file_index, filename in enumerate(filenames):
                    if file_index % len(archive_names) == archive_index:
                        archive.writestr(filename, _png_bytes(file_index))

        header = ["Filename"]
        if labeled:
            header += ["WaterLevel", "VA", *SEWER_ML_DEFECT_CLASSES, "ND", "Defect"]

        with open(root / csv_name, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for file_index, filename in enumerate(filenames):
                if not labeled:
                    writer.writerow([filename])
                    continue
                # Every third image carries an RB defect.
                defects = [1 if (file_index % 3 == 0 and code == "RB") else 0 for code in SEWER_ML_DEFECT_CLASSES]
                writer.writerow([filename, file_index % 4, 0, *defects, 0, int(any(defects))])

    return root


def _load(sewer_ml_tree, tmp_path, split=None):
    return SewerML(
        split=split,
        data_dir=str(sewer_ml_tree),
        download_dir=str(tmp_path / "downloads"),
        processed_cache_dir=str(tmp_path / "processed"),
    )


def test_sewer_ml_exposes_all_three_splits(sewer_ml_tree, tmp_path):
    splits = _load(sewer_ml_tree, tmp_path)
    assert sorted(splits.keys()) == ["test", "train", "validation"]
    for split in splits.keys():
        assert len(splits[split]) == 6, f"{split} should have 6 rows."


def test_sewer_ml_labeled_split_schema(sewer_ml_tree, tmp_path):
    ds = _load(sewer_ml_tree, tmp_path, split="train")
    sample = ds[0]

    assert set(sample.keys()) == {"image", "label", "defects", "water_level", "filename"}
    assert isinstance(sample["image"], PILImage.Image)
    assert sample["image"].size == (8, 8)
    assert sample["label"] in (0, 1)
    assert len(sample["defects"]) == len(SEWER_ML_DEFECT_CLASSES)
    assert set(sample["defects"]).issubset({0, 1})
    assert sample["filename"] == "train_0000.png"


def test_sewer_ml_defect_flag_matches_multi_hot(sewer_ml_tree, tmp_path):
    ds = _load(sewer_ml_tree, tmp_path, split="train")
    for index in range(len(ds)):
        sample = ds[index]
        assert sample["label"] == int(any(sample["defects"])), (
            f"Row {index}: binary label must agree with the multi-hot defect vector."
        )


def test_sewer_ml_reads_images_from_every_archive(sewer_ml_tree, tmp_path):
    """Train spreads 6 images over 14 archives, so every row must still resolve."""
    ds = _load(sewer_ml_tree, tmp_path, split="train")
    filenames = {ds[i]["filename"] for i in range(len(ds))}
    assert filenames == {f"train_{i:04d}.png" for i in range(6)}


def test_sewer_ml_single_split_needs_only_its_own_files(sewer_ml_tree, tmp_path):
    """split="validation" must not touch train/test files, so it works as a cheap access check."""
    csv_name, archive_names = _SPLIT_LAYOUT["val"]
    val_only = tmp_path / "val-only"
    val_only.mkdir()
    for name in (csv_name, *archive_names):
        (val_only / name).write_bytes((sewer_ml_tree / name).read_bytes())

    ds = _load(val_only, tmp_path, split="validation")
    assert len(ds) == 6


def test_sewer_ml_test_split_is_unlabeled(sewer_ml_tree, tmp_path):
    """The benchmark withholds test annotations; the CSV carries only Filename."""
    ds = _load(sewer_ml_tree, tmp_path, split="test")
    for index in range(len(ds)):
        sample = ds[index]
        assert sample["label"] is None
        assert sample["defects"] is None
        assert sample["water_level"] is None
        assert isinstance(sample["image"], PILImage.Image)


def test_sewer_ml_missing_data_dir_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="vap.aau.dk/sewer-ml"):
        SewerML(
            split="train",
            data_dir=str(tmp_path / "empty"),
            download_dir=str(tmp_path / "downloads"),
            processed_cache_dir=str(tmp_path / "processed"),
        )


def test_is_forbidden_walks_the_cause_chain():
    """download() hides the HTTP status inside __cause__, so the check must recurse."""
    http_error = Exception("403 Client Error: Forbidden for url: https://sciencedata.dk/...")
    wrapped = RuntimeError("Failed to download from all candidate URLs: https://sciencedata.dk/...")
    wrapped.__cause__ = http_error

    assert _is_forbidden(wrapped)
    assert not _is_forbidden(RuntimeError("Connection reset by peer"))


def test_is_forbidden_survives_a_cycle():
    first = RuntimeError("boom")
    second = RuntimeError("bang")
    first.__cause__ = second
    second.__cause__ = first
    assert not _is_forbidden(first)


def test_sewer_ml_source_declares_every_file():
    """3 annotation CSVs + 18 image archives."""
    assets = SewerML.SOURCE["assets"]
    assert len(assets) == 21
    for csv_name, archive_names in _SPLIT_LAYOUT.values():
        assert csv_name in assets
        for archive_name in archive_names:
            assert archive_name in assets


def _has_sewer_ml_credentials() -> bool:
    """True when requests would find a password for the host.

    ``requests`` resolves ~/.netrc itself, so setup is just a netrc stanza holding
    the share password; this mirrors that lookup so the gated tests skip instead
    of failing for contributors who have no access.
    """
    try:
        from requests.utils import get_netrc_auth

        return get_netrc_auth(f"{_BASE_URL}/SewerML_Val.csv") is not None
    except Exception:
        return False


requires_credentials = pytest.mark.skipif(
    not _has_sewer_ml_credentials(),
    reason="No Sewer-ML password: add a 'machine sciencedata.dk' stanza with a password to ~/.netrc.",
)


@requires_credentials
def test_sewer_ml_annotations_are_reachable_and_unchanged(tmp_path):
    """Cheap contract check against the live host: CSVs only, no image archives.

    Catches credential breakage, link rot and upstream schema drift without
    touching the ~340 GB of archives, so it is safe to run on a schedule.
    """
    from stable_datasets.utils import download

    train_csv = download(f"{_BASE_URL}/SewerML_Train.csv", tmp_path, progress_bar=False)
    with open(train_csv, newline="") as handle:
        train_columns = next(csv.reader(handle))

    assert "Defect" in train_columns, "The binary Defect column backs the anomaly label."
    for code in SEWER_ML_DEFECT_CLASSES:
        assert code in train_columns, f"Defect class {code} vanished from the annotations."

    test_csv = download(f"{_BASE_URL}/SewerML_Test.csv", tmp_path, progress_bar=False)
    with open(test_csv, newline="") as handle:
        test_columns = next(csv.reader(handle))

    # If the benchmark ever publishes test labels, the builder should stop
    # emitting None for them -- fail loudly so we notice.
    assert test_columns == ["Filename"], (
        f"Sewer-ML test annotations changed: {test_columns}. The builder assumes they are withheld."
    )


@pytest.mark.large
@requires_credentials
def test_sewer_ml_real_download(tmp_path):
    """Full build of the validation split: ~38 GB of archives. Not for hosted CI."""
    ds = SewerML(split="validation", processed_cache_dir=str(tmp_path / "processed"))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample["image"], PILImage.Image)
    assert len(sample["defects"]) == len(SEWER_ML_DEFECT_CLASSES)
