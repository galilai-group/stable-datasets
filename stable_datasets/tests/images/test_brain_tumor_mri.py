import io
import zipfile
from collections import Counter

import pytest
from PIL import Image

from stable_datasets.images import BrainTumorMRI


IMAGES_PER_CLASS = {"train": 1400, "test": 400}
LABELS = ["glioma", "meningioma", "notumor", "pituitary"]


def jpeg_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def fake_build(tmp_path):
    """Run the builder's generator on a fake zip with the real folder layout. Return its examples and the source bytes."""
    source_bytes = {}
    archive_path = tmp_path / "brain_tumor_mri.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for folder_index, folder in enumerate(["Training", "Testing"]):
            for label_index, label in enumerate(LABELS):
                for image_index in range(2):
                    # The builder matches the extension case insensitively, so include one upper case name.
                    extension = "JPG" if image_index == 1 else "jpg"
                    file_name = f"{folder}/{label}/{folder[:2]}-{label[:2]}_{image_index}.{extension}"
                    source_bytes[file_name] = jpeg_bytes((folder_index * 120, label_index * 60, image_index * 200))
                    archive.writestr(file_name, source_bytes[file_name])
        archive.writestr("README.md", b"not an image")
        archive.writestr("Training/glioma/notes.txt", b"not an image")

    builder = object.__new__(BrainTumorMRI)
    examples = {split: dict(builder._generate_examples(archive_path, split)) for split in ["train", "test"]}
    return examples, source_bytes


def test_fake_labels_match_source_folder(fake_build):
    examples, _ = fake_build
    for split_examples in examples.values():
        for file_name, example in split_examples.items():
            assert example["label"] == file_name.split("/")[1]


def test_fake_skips_non_image_files(fake_build):
    examples, source_bytes = fake_build
    yielded_files = [file_name for split_examples in examples.values() for file_name in split_examples]
    assert all(file_name.lower().endswith(".jpg") for file_name in yielded_files)
    assert len(yielded_files) == len(source_bytes)


def test_fake_splits_follow_published_folders(fake_build):
    examples, source_bytes = fake_build
    assert all(file_name.startswith("Training/") for file_name in examples["train"])
    assert all(file_name.startswith("Testing/") for file_name in examples["test"])
    assert set(examples["train"]).isdisjoint(examples["test"])
    assert set(examples["train"]) | set(examples["test"]) == set(source_bytes)


def test_fake_image_bytes_are_unchanged(fake_build):
    examples, source_bytes = fake_build
    for split_examples in examples.values():
        for file_name, example in split_examples.items():
            assert example["image"] == source_bytes[file_name]


@pytest.mark.parametrize("split", ["train", "test"])
def test_brain_tumor_mri_dataset(split):
    dataset = BrainTumorMRI(split=split)

    images_per_class = IMAGES_PER_CLASS[split]
    assert len(dataset) == 4 * images_per_class

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label"}

    # Images keep their original size and color mode, so only check that they decode.
    image = sample["image"]
    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0

    assert isinstance(sample["label"], int)
    label_counts = Counter(dataset[i]["label"] for i in range(len(dataset)))
    assert dict(label_counts) == dict.fromkeys(range(4), images_per_class)


def test_brain_tumor_mri_metadata():
    dataset = BrainTumorMRI(split="test")
    label_names = dataset.info.features["label"].names
    assert label_names == LABELS
