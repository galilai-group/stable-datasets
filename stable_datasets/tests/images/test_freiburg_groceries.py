"""Offline tests for author splits, duplicate handling and cache identity."""

import io
import tarfile

import pytest
from PIL import Image
from torch.utils.data import DataLoader

from stable_datasets.images import FreiburgGroceries


def add_bytes(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.fixture
def groceries_archives(tmp_path):
    images = tmp_path / "images.tar.gz"
    annotations = tmp_path / "annotations.tar.gz"
    prefix = f"freiburg_groceries_dataset-{FreiburgGroceries.SPLITS_REVISION}"
    with tarfile.open(images, "w") as archive:
        for label, name in enumerate(FreiburgGroceries.LABELS):
            for index in range(6):
                data = io.BytesIO()
                Image.new("RGB", (8, 8), (label, index, 123)).save(data, format="PNG")
                add_bytes(archive, f"images/{name}/{name}{index:04d}.png", data.getvalue())
    with tarfile.open(annotations, "w:gz") as archive:
        labels = "\n".join(f"{name} {label}" for label, name in enumerate(FreiburgGroceries.LABELS)) + "\n"
        add_bytes(archive, f"{prefix}/classid.txt", labels.encode())
        for fold in range(5):
            for split, index in (("train", fold), ("test", fold + 1)):
                rows = [
                    f"{name}/{name}{index:04d}.png {label}\n" for label, name in enumerate(FreiburgGroceries.LABELS)
                ]
                if split == "train":
                    rows *= 2
                add_bytes(archive, f"{prefix}/splits/{split}{fold}.txt", "".join(rows).encode())
    return images, annotations


def test_freiburg_all_configs(groceries_archives, tmp_path, monkeypatch):
    calls = []

    def download(specs, dest_folder):
        calls.append(list(specs))
        assert dest_folder == tmp_path / "downloads"
        return groceries_archives

    monkeypatch.setattr("stable_datasets.images.freiburg_groceries.bulk_download", download)
    kwargs = {"processed_cache_dir": tmp_path / "cache", "download_dir": tmp_path / "downloads"}
    datasets = []
    for fold in range(5):
        ds = FreiburgGroceries(config_name=f"split{fold}", **kwargs)
        assert set(ds) == {"train", "test"}
        assert len(ds["train"]) == 50 and len(ds["test"]) == 25
        assert ds["train"].features["label"].names == list(FreiburgGroceries.LABELS)
        for split, index in (("train", fold), ("test", fold + 1)):
            for i, row in enumerate(ds[split]):
                label = i % 25
                name = FreiburgGroceries.LABELS[label]
                assert set(row) == {"image", "label", "image_id"}
                assert row["label"] == label
                assert row["image_id"] == f"{name}/{name}{index:04d}.png"
                assert row["image"].getpixel((0, 0)) == (label, index, 123)
                assert row["image"].mode == "RGB"
        train_ids = [row["image_id"] for row in ds["train"]]
        test_ids = {row["image_id"] for row in ds["test"]}
        assert len(set(train_ids)) == 25
        assert train_ids[:25] == train_ids[25:]
        assert not set(train_ids) & test_ids
        datasets.append(ds)
    assert len(calls) == 5

    def no_download(*args, **kwargs):
        pytest.fail("Cached splits must reopen without downloading")

    monkeypatch.setattr("stable_datasets.images.freiburg_groceries.bulk_download", no_download)
    for fold, original in enumerate(datasets):
        reopened = FreiburgGroceries(config_name=f"split{fold}", **kwargs)
        for split in original:
            assert reopened[split].table.equals(original[split].table)
    default = FreiburgGroceries(split="train", **kwargs)
    assert default.table.equals(datasets[0]["train"].table)
    batch = next(iter(DataLoader(default.with_format("torch"), batch_size=8)))
    assert batch["image"].shape == (8, 3, 8, 8)
    assert batch["label"].tolist() == list(range(8))
    with pytest.raises(ValueError, match="Unknown config"):
        FreiburgGroceries(config_name="split5", **kwargs)
    with pytest.raises(ValueError, match="Split"):
        FreiburgGroceries(split="validation", **kwargs)


@pytest.mark.parametrize(
    "text",
    ["", "CANDY/a.png", "../CANDY0000.png 0", "CANDY/CANDY0000.png 25", "TEA/TEA0000.png 0", "CANDY//CANDY0000.png 0"],
)
def test_freiburg_bad_manifest(text):
    with pytest.raises(ValueError):
        FreiburgGroceries._parse_manifest(text)


@pytest.mark.parametrize("problem", ["labels", "overlap", "duplicate"])
def test_freiburg_bad_partitions(tmp_path, problem):
    builder = object.__new__(FreiburgGroceries)
    builder.__init__()
    path = tmp_path / "bad.tar"
    prefix = f"freiburg_groceries_dataset-{FreiburgGroceries.SPLITS_REVISION}"
    with tarfile.open(path, "w") as archive:
        labels = "\n".join(f"{name} {i}" for i, name in enumerate(FreiburgGroceries.LABELS))
        add_bytes(archive, f"{prefix}/classid.txt", ("wrong" if problem == "labels" else labels).encode())
        add_bytes(archive, f"{prefix}/splits/train0.txt", b"CANDY/CANDY0000.png 0\n")
        test = "CANDY/CANDY0000.png 0\n" if problem == "overlap" else "CANDY/CANDY0001.png 0\n"
        if problem == "duplicate":
            test *= 2
        add_bytes(archive, f"{prefix}/splits/test0.txt", test.encode())
    with pytest.raises(
        ValueError, match={"labels": "mapping", "overlap": "overlap", "duplicate": "Duplicate"}[problem]
    ):
        builder._read_partitions(path)


def test_freiburg_missing_image(groceries_archives):
    builder = object.__new__(FreiburgGroceries)
    with pytest.raises(ValueError, match="Missing regular"):
        list(builder._generate_examples(groceries_archives[0], [("CANDY/CANDY9999.png", 0)]))
