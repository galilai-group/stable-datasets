import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.cct20 import CCT20


# Same folder layout as the real archives.
IMAGES_DIR = "eccv_18_all_images_sm"
ANNOTATIONS_DIR = "eccv_18_annotation_files"

# Fake images per official annotation file: (file name, category ids, location, grayscale).
# Category ids are the real ones: 1 = opossum (label 0), 3 = raccoon (label 1), 99 = rodent (label 14), 30 = empty.
IMAGES = {
    "train": [
        ("a.jpg", [1], 38, False),
        ("b.jpg", [3], 38, True),  # night-time infrared image -> must come out as RGB
        ("c.jpg", [99], 43, False),
        ("d.jpg", [1, 3], 43, False),  # two categories -> excluded
    ],
    "cis_val": [("e.jpg", [3], 38, False), ("f.jpg", [30], 38, False)],  # "empty" -> excluded
    "cis_test": [("g.jpg", [1], 43, False)],
    "trans_val": [("h.jpg", [99], 125, False)],
    "trans_test": [("i.jpg", [1], 46, False), ("j.jpg", [3], 46, False)],
}
LABELS = {1: 0, 3: 1, 99: 14}


def _jpeg_bytes(seed, grayscale):
    """A small random JPEG, grayscale ('L') or RGB like the real camera-trap images."""
    rng = np.random.default_rng(seed)
    shape = (48, 64) if grayscale else (48, 64, 3)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _add_file(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


@pytest.fixture
def fake_cct20(tmp_path, monkeypatch):
    """Write a miniature CCT20 (images + annotations archives) and make the builder use it offline."""
    images_tar = tmp_path / "cct20_images_sm.tar.gz"
    annotations_tar = tmp_path / "cct20_annotations.tar.gz"
    with tarfile.open(images_tar, "w:gz") as images, tarfile.open(annotations_tar, "w:gz") as annotations:
        image_id = 0
        for split, rows in IMAGES.items():
            data = {"images": [], "annotations": [], "categories": []}
            for file_name, category_ids, location, grayscale in rows:
                image_id += 1
                _add_file(images, f"{IMAGES_DIR}/{file_name}", _jpeg_bytes(image_id, grayscale))
                data["images"].append({"id": image_id, "file_name": file_name, "location": location})
                for category_id in category_ids:
                    annotation_id = len(data["annotations"])
                    data["annotations"].append({"id": annotation_id, "image_id": image_id, "category_id": category_id})
            _add_file(annotations, f"{ANNOTATIONS_DIR}/{split}_annotations.json", json.dumps(data).encode())

    files = {"cct20_images_sm.tar.gz": images_tar, "cct20_annotations.tar.gz": annotations_tar}
    monkeypatch.setattr(
        "stable_datasets.images.cct20.bulk_download",
        lambda infos, dest_folder: [files[info.filename] for info in infos],
    )
    return tmp_path


def test_split_sizes_and_exclusions(fake_cct20):
    ds = CCT20(processed_cache_dir=fake_cct20 / "processed")
    # train loses the two-category image d.jpg, validation loses the "empty" image f.jpg
    assert {name: len(split) for name, split in ds.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
        "trans_validation": 1,
        "trans_test": 2,
    }


def test_labels_locations_and_rgb(fake_cct20):
    train = CCT20(split="train", processed_cache_dir=fake_cct20 / "processed")
    samples = sorted((train[i] for i in range(len(train))), key=lambda s: s["label"])
    assert [s["label"] for s in samples] == [LABELS[1], LABELS[3], LABELS[99]]
    assert [s["location"] for s in samples] == [38, 38, 43]
    # every image is RGB, including the grayscale b.jpg (label 1)
    assert all(s["image"].mode == "RGB" for s in samples)
    assert train.features["label"].names[14] == "rodent"


def test_cct20_real_data():
    """Real download: official split sizes after the documented exclusions, class names and image format."""
    ds = CCT20()
    assert {name: len(split) for name, split in ds.items()} == {
        "train": 13546,
        "validation": 2489,
        "test": 14646,
        "trans_validation": 1603,
        "trans_test": 21488,
    }

    assert ds["train"].features["label"].names == [
        "opossum",
        "raccoon",
        "squirrel",
        "bobcat",
        "skunk",
        "dog",
        "coyote",
        "rabbit",
        "bird",
        "cat",
        "badger",
        "car",
        "deer",
        "fox",
        "rodent",
    ]

    sample = ds["train"][0]
    assert set(sample) == {"image", "label", "location"}
    image = np.array(sample["image"])
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[1] == 1024
    assert 0 <= sample["label"] < 15
