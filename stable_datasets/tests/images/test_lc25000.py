import csv
import hashlib
import io
import zipfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import LC25000, lc25000
from stable_datasets.images.lc25000 import split_groups


EXPECTED_LABEL_COUNTS = {
    "train": {0: 3993, 1: 3980, 2: 3973, 3: 4020, 4: 3974},
    "test": {0: 1007, 1: 1020, 2: 1027, 3: 980, 4: 1026},
}
EXPECTED_GROUP_COUNTS = {"train": 996, "test": 250}
# SHA-256 of the sorted test group ids joined by commas, so any change to the split fails the test.
EXPECTED_TEST_GROUPS_SHA256 = "ed51e8fdc7795a7d12abde4b14f32edcfd5000a1db7522fff8f56e16a97d6540"
FAKE_LABELS = ["colon_aca", "lung_n"]


def synthetic_group_rows():
    """Two classes with 10 groups each and 3 copies per group. The hundreds digit of a group id is its class."""
    group_rows = []
    for label_index, label in enumerate(FAKE_LABELS):
        for group in range(10):
            for copy in range(3):
                image_name = f"{label}{group}_{copy}.jpeg"
                group_rows.append({"filename": image_name, "label": label, "group_id": str(label_index * 100 + group)})
    return group_rows


def jpeg_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    """Run the builder's generator on a fake image zip and group zip. Return its examples and the source bytes."""
    group_rows = synthetic_group_rows()
    source_bytes = {}
    images_path = tmp_path / "lc25000.zip"
    with zipfile.ZipFile(images_path, "w") as archive:
        for index, row in enumerate(group_rows):
            tissue = row["label"].split("_")[0]
            file_name = f"lung_colon_image_set/{tissue}_image_sets/{row['label']}/{row['filename']}"
            source_bytes[file_name] = jpeg_bytes((index * 4 % 256, index * 7 % 256, index * 11 % 256))
            archive.writestr(file_name, source_bytes[file_name])
        archive.writestr("lung_colon_image_set/README.txt", b"not an image")
        archive.writestr("lung_colon_image_set/colon_image_sets/colon_aca/Thumbs.db", b"not an image")

    groups_csv = io.StringIO()
    writer = csv.DictWriter(groups_csv, fieldnames=["filename", "label", "group_id"])
    writer.writeheader()
    writer.writerows(group_rows)
    groups_path = tmp_path / "lc25000_clean_groups.zip"
    with zipfile.ZipFile(groups_path, "w") as archive:
        archive.writestr(lc25000.GROUPS_FILE, groups_csv.getvalue())
    monkeypatch.setattr(lc25000, "download", lambda *args, **kwargs: groups_path)

    builder = object.__new__(LC25000)
    examples = {split: dict(builder._generate_examples(images_path, split)) for split in ["train", "test"]}
    return examples, source_bytes


def test_split_groups_keeps_each_group_in_one_split():
    group_split = split_groups(synthetic_group_rows())

    assert len(group_split) == 20
    test_groups_per_label = Counter(group_id // 100 for group_id, split in group_split.items() if split == "test")
    assert test_groups_per_label == {0: 2, 1: 2}
    assert split_groups(list(reversed(synthetic_group_rows()))) == group_split


def test_split_groups_is_pinned():
    group_split = split_groups(synthetic_group_rows())
    assert sorted(group_id for group_id, split in group_split.items() if split == "test") == [7, 8, 107, 108]


def test_fake_labels_match_source_folder(fake_build):
    examples, _ = fake_build
    for split_examples in examples.values():
        for file_name, example in split_examples.items():
            assert example["label"] == file_name.split("/")[-2]


def test_fake_skips_non_image_files(fake_build):
    examples, source_bytes = fake_build
    yielded_files = [file_name for split_examples in examples.values() for file_name in split_examples]
    assert all(file_name.endswith(".jpeg") for file_name in yielded_files)
    assert len(yielded_files) == len(source_bytes)


def test_fake_splits_are_disjoint_and_cover_all_images(fake_build):
    examples, source_bytes = fake_build
    assert set(examples["train"]).isdisjoint(examples["test"])
    assert set(examples["train"]) | set(examples["test"]) == set(source_bytes)

    train_groups = {example["group_id"] for example in examples["train"].values()}
    test_groups = {example["group_id"] for example in examples["test"].values()}
    assert train_groups.isdisjoint(test_groups)
    assert test_groups == {7, 8, 107, 108}


def test_fake_image_bytes_are_unchanged(fake_build):
    examples, source_bytes = fake_build
    for split_examples in examples.values():
        for file_name, example in split_examples.items():
            assert example["image"] == source_bytes[file_name]


@pytest.mark.large
@pytest.mark.parametrize("split", ["train", "test"])
def test_lc25000_dataset(split):
    dataset = LC25000(split=split)

    expected_counts = EXPECTED_LABEL_COUNTS[split]
    assert len(dataset) == sum(expected_counts.values())

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label", "group_id"}

    image = sample["image"]
    assert isinstance(image, Image.Image)
    image_array = np.array(image)
    assert image_array.shape == (768, 768, 3)
    assert image_array.dtype == np.uint8

    assert isinstance(sample["label"], int)
    assert isinstance(sample["group_id"], int)
    label_counts = Counter(dataset.table.column("label").to_pylist())
    assert dict(label_counts) == expected_counts
    assert len(set(dataset.table.column("group_id").to_pylist())) == EXPECTED_GROUP_COUNTS[split]


@pytest.mark.large
def test_lc25000_splits_are_disjoint_complete_and_pinned():
    train_groups = LC25000(split="train").table.column("group_id").to_pylist()
    test_groups = LC25000(split="test").table.column("group_id").to_pylist()

    assert set(train_groups).isdisjoint(test_groups)
    assert len(train_groups) + len(test_groups) == 25000
    test_group_ids = ",".join(str(group_id) for group_id in sorted(set(test_groups)))
    assert hashlib.sha256(test_group_ids.encode()).hexdigest() == EXPECTED_TEST_GROUPS_SHA256


@pytest.mark.large
def test_lc25000_metadata():
    dataset = LC25000(split="test")
    label_names = dataset.info.features["label"].names
    assert label_names == ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
