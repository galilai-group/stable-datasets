import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.deep_weeds import DeepWeeds


# Same folder layout as the real GitHub archive: DeepWeeds-<commit>/labels/*.csv
REPO_DIR = "DeepWeeds-da084f62bcd1a2b0afeb8b81f1a27be3186391a1/labels"

# 9 fake images, one per class (img0 -> class 0, ..., img8 -> class 8)
LABELS = {f"img{i}.jpg": i % 9 for i in range(9)}

# Two different official splits, so a test can tell them apart
SPLITS = {
    "subset0": {
        "train": ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
        "val": ["img5.jpg", "img6.jpg"],
        "test": ["img7.jpg", "img8.jpg"],
    },
    "subset1": {
        "train": ["img4.jpg", "img5.jpg", "img6.jpg", "img7.jpg", "img8.jpg"],
        "val": ["img0.jpg", "img1.jpg"],
        "test": ["img2.jpg", "img3.jpg"],
    },
}


def _jpeg_bytes(seed):
    """A random 256x256 RGB JPEG, like a real DeepWeeds image."""
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _csv(rows):
    """CSV text in the authors' format: header 'Filename,Label', one row per image."""
    return "Filename,Label\n" + "".join(f"{f},{label}\n" for f, label in rows)


@pytest.fixture
def fake_deepweeds(tmp_path, monkeypatch):
    """Write a miniature DeepWeeds (images zip + repository zip) and make the builder use it offline."""
    images_zip = tmp_path / "deepweeds_images.zip"
    with zipfile.ZipFile(images_zip, "w") as z:
        for i, name in enumerate(LABELS):
            z.writestr(name, _jpeg_bytes(i))

    repo_zip = tmp_path / "deepweeds_labels.zip"
    with zipfile.ZipFile(repo_zip, "w") as z:
        z.writestr(f"{REPO_DIR}/labels.csv", _csv(LABELS.items()))
        for k in range(5):
            splits = SPLITS.get(f"subset{k}", SPLITS["subset0"])
            for split, names in splits.items():
                # Deliberate conflict, like the real image 20170714-110407-3.jpg:
                # the split CSV gives img0 a different label than labels.csv.
                rows = [(n, (LABELS[n] + 1) % 9 if n == "img0.jpg" else LABELS[n]) for n in names]
                z.writestr(f"{REPO_DIR}/{split}_subset{k}.csv", _csv(rows))

    files = {"deepweeds_images.zip": images_zip, "deepweeds_labels.zip": repo_zip}
    monkeypatch.setattr(
        "stable_datasets.images.deep_weeds.bulk_download",
        lambda infos, dest_folder: [files[info.filename] for info in infos],
    )
    return tmp_path


def test_deep_weeds_real_data():
    """Real download: official subset0 sizes, class names and image format."""
    ds = DeepWeeds()  # default variant subset0, all splits
    assert {name: len(split) for name, split in ds.items()} == {"train": 10501, "validation": 3501, "test": 3507}

    assert ds["train"].features["label"].names == [
        "Chinee_apple",
        "Lantana",
        "Parkinsonia",
        "Parthenium",
        "Prickly_acacia",
        "Rubber_vine",
        "Siam_weed",
        "Snake_weed",
        "Negative",
    ]

    sample = ds["train"][0]
    image = np.array(sample["image"])
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8
    assert 0 <= sample["label"] < 9


def test_split_sizes_and_samples(fake_deepweeds):
    ds = DeepWeeds(processed_cache_dir=fake_deepweeds / "processed")
    assert {name: len(split) for name, split in ds.items()} == {"train": 5, "validation": 2, "test": 2}
    sample = ds["train"][0]
    assert set(sample) == {"image", "label"}
    assert sample["image"].size == (256, 256)


def test_labels_come_from_labels_csv(fake_deepweeds):
    train = DeepWeeds(split="train", processed_cache_dir=fake_deepweeds / "processed")
    got = sorted(train[i]["label"] for i in range(len(train)))
    assert got == sorted(LABELS[n] for n in SPLITS["subset0"]["train"])


def test_config_selects_official_split(fake_deepweeds):
    val = DeepWeeds(config_name="subset1", split="validation", processed_cache_dir=fake_deepweeds / "processed")
    assert sorted(val[i]["label"] for i in range(len(val))) == sorted(LABELS[n] for n in SPLITS["subset1"]["val"])
