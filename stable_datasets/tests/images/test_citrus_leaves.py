import io
import zipfile
from collections import Counter

from PIL import Image

from stable_datasets.images import CitrusLeaves
from stable_datasets.schema import ClassLabel
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import download


LABEL_NAMES = ["Black spot", "canker", "greening", "healthy"]
SPLIT_COUNTS = {"train": [169, 163, 204, 58]}


def _builder():
    builder = object.__new__(CitrusLeaves)
    builder.__init__()
    return builder


def test_citrus_leaves_metadata():
    builder = _builder()
    assert str(CitrusLeaves.VERSION) == "1.0.0"
    assert set(CitrusLeaves.SOURCE["assets"]) == set(SPLIT_COUNTS)
    assert builder.info.supervised_keys == ("image", "label")
    assert set(builder.info.features) == {"image", "label"}
    assert isinstance(builder.info.features["image"], ImageFeature)
    assert isinstance(builder.info.features["label"], ClassLabel)
    assert builder.info.features["label"].names == LABEL_NAMES
    assert builder.info.homepage == CitrusLeaves.SOURCE["homepage"]
    assert builder.info.citation.startswith("@misc{")


def test_citrus_leaves_real_archive_and_dataset(tmp_path):
    # A fresh directory forces a real source download, even with a populated user cache.
    download_dir = tmp_path / "downloads"
    archive_path = download(CitrusLeaves.SOURCE["assets"]["train"], dest_folder=download_dir)
    builder = _builder()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        image_paths = {
            name for name in archive.namelist() if name.startswith("Citrus/Leaves/") and name.endswith(".png")
        }
        assert Counter(name.split("/")[-2] for name in image_paths) == dict(zip(LABEL_NAMES, [169, 163, 204, 58]))
        assert any(name.startswith("Citrus/Fruits/") and name.endswith(".jpg") for name in archive.namelist())
    keys_by_split = {}
    for split, expected_counts in SPLIT_COUNTS.items():
        keys = []
        labels = Counter()
        for key, sample in builder._generate_examples(archive_path, split):
            assert isinstance(key, str)
            assert key in image_paths
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], bytes)
            assert isinstance(sample["label"], int)
            assert 0 <= sample["label"] < len(LABEL_NAMES)
            assert LABEL_NAMES[sample["label"]] == key.split("/")[-2]
            with Image.open(io.BytesIO(sample["image"])) as image:
                image.verify()
            keys.append(key)
            labels[sample["label"]] += 1
        assert len(keys) == len(set(keys)) == sum(expected_counts)
        assert labels == dict(enumerate(expected_counts))
        keys_by_split[split] = set(keys)
    assert set.union(*keys_by_split.values()) == image_paths

    # Exercise the public builder, Arrow preparation and decoding, not just the generator.
    datasets = CitrusLeaves(download_dir=download_dir, processed_cache_dir=tmp_path / "processed")
    assert set(datasets) == set(SPLIT_COUNTS)
    for split, expected_counts in SPLIT_COUNTS.items():
        dataset = datasets[split]
        assert len(dataset) == sum(expected_counts)
        labels = Counter()
        for sample in dataset:
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], Image.Image)
            assert sample["image"].mode == "RGB"
            assert sample["image"].size == (256, 256)
            sample["image"].load()
            assert isinstance(sample["label"], int)
            assert 0 <= sample["label"] < len(LABEL_NAMES)
            labels[sample["label"]] += 1
        assert labels == dict(enumerate(expected_counts))
