"""ETH-80 parsing, instance isolation, cache and real-data loading tests."""

import io
import tarfile
from collections import Counter
from itertools import combinations

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from stable_datasets.images import ETH80


def add_bytes(archive, name, content):
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


@pytest.fixture
def eth80_archive(tmp_path):
    path = tmp_path / "eth80.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for label, category in enumerate(ETH80.LABELS):
            for instance in ETH80.INSTANCE_IDS[category]:
                for view in range(41):
                    name = f"{category}{instance}-{view:03d}-000.png"
                    data = io.BytesIO()
                    Image.new("RGBA", (8, 8), (label, instance, view, 255)).save(data, format="PNG")
                    add_bytes(archive, f"eth80-cropped256/{category}{instance}/{name}", data.getvalue())
        add_bytes(archive, "eth80-cropped256/apple1/maps/apple1-000-000-map.png", b"not an image")
    return path


def test_eth80_end_to_end(eth80_archive, tmp_path, monkeypatch):
    calls = []

    def download(specs, dest_folder):
        calls.append(specs)
        return [eth80_archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    kwargs = {"processed_cache_dir": tmp_path / "cache", "download_dir": tmp_path / "raw"}
    datasets = ETH80(**kwargs)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert {split: len(ds) for split, ds in datasets.items()} == {"train": 1968, "validation": 656, "test": 656}
    ids, objects = {}, {}
    for split, ds in datasets.items():
        assert ds.features["label"].names == list(ETH80.LABELS)
        ids[split], objects[split] = set(), set()
        counts = Counter()
        for row in ds:
            assert set(row) == {"image", "label", "object_id", "instance_id", "view_id", "image_id"}
            assert row["image"].mode == "RGB"
            assert row["instance_id"] in ETH80.INSTANCE_IDS[ETH80.LABELS[row["label"]]][ETH80.SPLIT_SLICES[split]]
            assert row["object_id"] == f"{ETH80.LABELS[row['label']]}{row['instance_id']}"
            assert row["image_id"] == f"{row['object_id']}-{row['view_id']}.png"
            view = int(row["view_id"].split("-")[0])
            assert row["image"].getpixel((0, 0)) == (row["label"], row["instance_id"], view)
            ids[split].add(row["image_id"])
            objects[split].add(row["object_id"])
            counts[row["object_id"]] += 1
        assert set(counts.values()) == {41}
    for a, b in combinations(datasets, 2):
        assert not ids[a] & ids[b]
        assert not objects[a] & objects[b]
    assert sum(map(len, ids.values())) == 3280
    assert {x for x in objects["test"] if x.startswith("car")} == {"car12", "car14"}

    def no_download(*args, **kwargs):
        pytest.fail("Reopening a processed cache must not access the network")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    reopened = ETH80(**kwargs)
    for split in datasets:
        assert reopened[split].with_format("raw").table.equals(datasets[split].with_format("raw").table)

    def resize(row):
        row["image"] = np.asarray(row["image"].resize((16, 16))).transpose(2, 0, 1).copy() / np.float32(255)
        return row

    batch = next(iter(DataLoader(reopened["train"].with_transform(resize), batch_size=7)))
    assert batch["image"].shape == (7, 3, 16, 16)
    assert batch["label"].shape == (7,)
    assert next(iter(DataLoader(reopened["test"].with_format("torch"), batch_size=4)))["image"].shape == (4, 3, 8, 8)
    with pytest.raises(ValueError, match="Split"):
        ETH80(split="invalid", **kwargs)


def test_eth80_real_dataset(tmp_path):
    kwargs = {"processed_cache_dir": tmp_path / "processed"}
    datasets = ETH80(**kwargs)
    labels = ["apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato"]
    split_sizes = {"train": 1968, "validation": 656, "test": 656}
    instance_slices = {"train": slice(0, 6), "validation": slice(6, 8), "test": slice(8, 10)}
    object_counts = {"train": 48, "validation": 16, "test": 16}
    assert {split: len(ds) for split, ds in datasets.items()} == split_sizes

    image_ids, objects = {}, {}
    for split, ds in datasets.items():
        assert ds.features["label"].names == labels
        image_ids[split], objects[split] = set(), set()
        views_per_object, class_counts = Counter(), Counter()
        for row in ds:
            assert set(row) == {"image", "label", "object_id", "instance_id", "view_id", "image_id"}
            assert isinstance(row["image"], Image.Image)
            assert row["image"].mode == "RGB"
            pixels = np.asarray(row["image"])
            assert pixels.shape == (256, 256, 3)
            assert pixels.dtype == np.uint8
            assert isinstance(row["label"], int) and 0 <= row["label"] < 8
            assert isinstance(row["instance_id"], int)
            assert isinstance(row["object_id"], str)
            assert isinstance(row["view_id"], str)
            assert row["object_id"] == f"{labels[row['label']]}{row['instance_id']}"
            assert row["image_id"] == f"{row['object_id']}-{row['view_id']}.png"
            image_ids[split].add(row["image_id"])
            objects[split].add(row["object_id"])
            views_per_object[row["object_id"]] += 1
            class_counts[row["label"]] += 1

        expected_objects = set()
        for category in labels:
            instances = (1, 2, 3, 5, 6, 7, 9, 11, 12, 14) if category == "car" else tuple(range(1, 11))
            expected_objects.update(f"{category}{instance}" for instance in instances[instance_slices[split]])
        assert objects[split] == expected_objects
        assert len(objects[split]) == object_counts[split]
        assert len(image_ids[split]) == len(ds)
        assert set(views_per_object.values()) == {41}
        assert class_counts == dict.fromkeys(range(8), split_sizes[split] // 8)

    for first, second in combinations(datasets, 2):
        assert not image_ids[first] & image_ids[second]
        assert not objects[first] & objects[second]
    assert sum(map(len, image_ids.values())) == 3280

    reopened = ETH80(**kwargs)
    for split in datasets:
        assert reopened[split].table.equals(datasets[split].table)
    batch = next(iter(DataLoader(reopened["train"].with_format("torch"), batch_size=8)))
    assert batch["image"].shape == (8, 3, 256, 256)
    assert batch["label"].shape == (8,)


@pytest.mark.parametrize(
    "names,error",
    [
        (["eth80-cropped256/apple1/apple1-000-000.png"], "Incomplete"),
        (["eth80-cropped256/apple1/broken.png"], "Malformed"),
        (["eth80-cropped256/apple2/apple1-000-000.png"], "Unexpected"),
        (["eth80-cropped256/apple1/apple1-000-000.png"] * 2, "Duplicate"),
    ],
)
def test_eth80_rejects_invalid_archives(tmp_path, names, error):
    path = tmp_path / "bad.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            add_bytes(archive, name, b"unused")
    builder = object.__new__(ETH80)
    with pytest.raises(ValueError, match=error):
        list(builder._generate_examples(path, "train"))
