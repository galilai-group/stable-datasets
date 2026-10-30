"""Offline ISIC2016 label joins, error handling and real cache tests."""

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import ISIC2016


@pytest.fixture
def original_assets(tmp_path):
    payloads = {}
    for prefix, ids in (("Training", (2, 1)), ("Test", (4, 3))):
        with zipfile.ZipFile(tmp_path / f"ISBI2016_ISIC_Part3_{prefix}_Data.zip", "w") as archive:
            for i in ids:
                image_id = f"ISIC_{i:07d}"
                buffer = io.BytesIO()
                Image.new("RGB", (8, 6), (30 * i, 10, 90)).save(buffer, format="JPEG")
                payloads[image_id] = buffer.getvalue()
                archive.writestr(f"{prefix}/{image_id}.jpg", payloads[image_id])
        csv_path = tmp_path / f"ISBI2016_ISIC_Part3_{prefix}_GroundTruth.csv"
        # Reverse archive order so an incorrect positional join is observable.
        rows = (
            "ISIC_0000001,malignant\nISIC_0000002,benign\n"
            if prefix == "Training"
            else ("ISIC_0000003,0.0\nISIC_0000004,1.0\n")
        )
        csv_path.write_text(rows, encoding="utf-8")
    return tmp_path, payloads


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("image_id,label\n", "Invalid"),
        ("ISIC_0000001,unknown\n", "Invalid"),
        ("not_an_id,benign\n", "Invalid"),
        ("ISIC_0000001,benign,extra\n", "exactly two"),
        ("ISIC_0000001,benign\nISIC_0000001,malignant\n", "Duplicate"),
    ],
)
def test_isic2016_rejects_bad_labels(tmp_path, content, message):
    path = tmp_path / "labels.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        ISIC2016._read_labels(path, "train")


def test_isic2016_rejects_image_label_mismatch(original_assets):
    root, _ = original_assets
    labels = root / "labels.csv"
    labels.write_text("ISIC_0000001,benign\nISIC_9999999,malignant\n")
    with pytest.raises(ValueError, match="missing images=.*9999999.*unlabelled images=.*0000002"):
        list(
            object.__new__(ISIC2016)._generate_examples(
                root / "ISBI2016_ISIC_Part3_Training_Data.zip", labels, "train"
            )
        )


@pytest.fixture(scope="module")
def real_isic2016(tmp_path_factory):
    """Use the public download path, not local assets or mocks."""
    root = tmp_path_factory.mktemp("real_isic2016")
    return ISIC2016(download_dir=root / "downloads", processed_cache_dir=root / "processed"), root


@pytest.mark.parametrize(
    "split,size,counts,prefix",
    [
        ("train", 900, [727, 173], "Training"),
        ("test", 379, [304, 75], "Test"),
    ],
)
def test_isic2016_real_dataset(real_isic2016, split, size, counts, prefix):
    datasets, root = real_isic2016
    assert set(datasets) == {"train", "test"}
    ds = datasets[split]
    assert len(ds) == size
    assert ds.features["label"].names == ["benign", "malignant"]
    labels = ds.table.column("label").to_pylist()
    ids = ds.table.column("image_id").to_pylist()
    assert Counter(labels) == dict(enumerate(counts))
    assert len(set(ids)) == size and ids == sorted(ids)
    # The common downloader inserts a URL hash into each cached filename.
    csv_paths = list((root / "downloads").glob(f"ISBI2016_ISIC_Part3_{prefix}_GroundTruth.*.csv"))
    assert len(csv_paths) == 1
    path = csv_paths[0]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        source = list(csv.reader(stream))
    encoding = {"benign": 0, "malignant": 1} if split == "train" else {"0.0": 0, "1.0": 1}
    assert len(source) == size and {r[1] for r in source} == set(encoding)
    expected = {r[0]: encoding[r[1]] for r in source}
    assert dict(zip(ids, labels)) == expected
    archive_paths = list((root / "downloads").glob(f"ISBI2016_ISIC_Part3_{prefix}_Data.*.zip"))
    assert len(archive_paths) == 1
    archive_path = archive_paths[0]
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(n).stem: n for n in archive.namelist() if n.lower().endswith(".jpg")}
        assert set(members) == set(ids)
        for label in (0, 1):
            index = labels.index(label)
            sample = ds[index]
            assert set(sample) == {"image", "label", "image_id"}
            assert isinstance(sample["image"], Image.Image) and sample["image"].mode == "RGB"
            assert np.asarray(sample["image"]).dtype == np.uint8
            assert min(sample["image"].size) > 0
            assert ds.with_format("raw")[index]["image"] == archive.read(members[sample["image_id"]])
    assert not set(datasets["train"].table.column("image_id").to_pylist()) & set(
        datasets["test"].table.column("image_id").to_pylist()
    )
    reopened = ISIC2016(split=split, download_dir=root / "downloads", processed_cache_dir=root / "processed")
    assert reopened.with_format("raw")[0] == ds.with_format("raw")[0]
