import io
import zipfile

import numpy as np
import pytest
from PIL import Image

import stable_datasets.utils as utils
from stable_datasets.images.jena_flowers30 import JenaFlowers30
from stable_datasets.schema import ClassLabel
from stable_datasets.utils import BaseDatasetBuilder


@pytest.fixture
def local_archive(tmp_path):
    """A small ZIP with the layout and CSV format of the official JF30-1.zip."""
    path = tmp_path / "JF30-1.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("jpg/", b"")
        for name, mode, size in [
            ("JFT_01479.jpg", "CMYK", (19, 11)),
            ("JFT_00018.jpg", "RGB", (17, 9)),
            ("JFT_00001.jpg", "L", (13, 7)),
        ]:
            buffer = io.BytesIO()
            Image.new(mode, size).save(buffer, format="JPEG")
            archive.writestr(f"jpg/{name}", buffer.getvalue())
        archive.writestr(
            "classlabels.txt",
            "JFT_00018.jpg, 2\nJFT_01479.jpg, 30\nJFT_00001.jpg, 1\n",
        )
        archive.writestr("readme.txt", "Jena Flower 30 Dataset\n")
        for fold, image_ids in enumerate([(1, 18, 1479), (18, 1479, 1), (1479, 1, 18)], start=1):
            for prefix, image_id in zip(("trn", "val", "tst"), image_ids):
                archive.writestr(f"{prefix}{fold}.txt", f"{image_id}\n")
        archive.writestr("jpg/unlabelled.jpg", b"Not an image")
        archive.writestr("__MACOSX/jpg/._JFT_00001.jpg", b"Not an image")
    return path


def test_jena_flowers30_metadata():
    builder = object.__new__(JenaFlowers30)
    builder.__init__()

    assert isinstance(builder, BaseDatasetBuilder)
    assert builder.config.name == "all"
    assert [config.name for config in builder.BUILDER_CONFIGS] == ["all", "fold1", "fold2", "fold3"]
    assert builder.info.config_name == "all"
    assert set(builder.info.features) == {"image", "label"}
    assert builder.info.supervised_keys == ("image", "label")
    assert isinstance(builder.info.features["label"], ClassLabel)
    assert builder.info.features["label"].names == [str(label) for label in range(1, 31)]
    assert builder.info.homepage == "https://doi.org/10.7910/DVN/QDHYST"
    assert builder.info.license == "CC BY-NC 4.0"
    assert "10.1371/journal.pone.0170629" in builder.info.citation
    assert set(builder.SOURCE["assets"]) == {"train"}

    with pytest.raises(TypeError):
        JenaFlowers30.SOURCE["homepage"] = "https://example.com"
    with pytest.raises(TypeError):
        JenaFlowers30.SOURCE["assets"]["train"] = "https://example.com/archive.zip"


def test_jena_flowers30_rejects_unknown_config():
    builder = object.__new__(JenaFlowers30)
    with pytest.raises(ValueError, match="Unknown config"):
        builder.__init__(config_name="fold4")


def test_jena_flowers30_examples_join_labels_by_filename(local_archive):
    builder = object.__new__(JenaFlowers30)
    builder.__init__()
    examples = list(builder._generate_examples(data_path=local_archive, split="train"))

    assert [key for key, _ in examples] == [
        "jpg/JFT_00001.jpg",
        "jpg/JFT_00018.jpg",
        "jpg/JFT_01479.jpg",
    ]
    assert [sample["label"] for _, sample in examples] == [0, 1, 29]
    assert [sample["image"].size for _, sample in examples] == [(13, 7), (17, 9), (19, 11)]
    with zipfile.ZipFile(local_archive) as archive:
        for key, sample in examples:
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["label"], int)
            assert isinstance(sample["image"], Image.Image)
            assert sample["image"].mode == "RGB"
            assert np.asarray(sample["image"]).dtype == np.uint8
            with Image.open(io.BytesIO(archive.read(key))) as source_image:
                np.testing.assert_array_equal(np.asarray(sample["image"]), np.asarray(source_image.convert("RGB")))


def test_jena_flowers30_public_builder_and_cache(local_archive, tmp_path, monkeypatch):
    calls = []

    def local_download(urls, dest_folder):
        calls.append((list(urls), dest_folder))
        return [local_archive]

    monkeypatch.setattr(utils, "bulk_download", local_download)
    kwargs = {
        "download_dir": tmp_path / "downloads",
        "processed_cache_dir": tmp_path / "processed",
    }
    datasets = JenaFlowers30(split=None, **kwargs)
    assert set(datasets) == {"train"}
    assert len(datasets["train"]) == 3
    assert datasets["train"].features["label"].names == [str(label) for label in range(1, 31)]
    assert datasets["train"][0]["image"].size == (13, 7)
    assert datasets["train"][0]["image"].mode == "RGB"
    assert isinstance(datasets["train"][0]["label"], int)
    assert datasets["train"].table.column("label").to_pylist() == [0, 1, 29]
    assert calls == [([JenaFlowers30.SOURCE["assets"]["train"]], kwargs["download_dir"])]

    def no_regeneration(*args, **kwargs):
        pytest.fail("Prepared Arrow cache should be reused")

    monkeypatch.setattr(JenaFlowers30, "_generate_examples", no_regeneration)
    dataset = JenaFlowers30(config_name="all", split="train", **kwargs)
    assert len(dataset) == 3
    assert dataset[2]["label"] == 29
    with pytest.raises(ValueError, match="Split 'test' not found"):
        JenaFlowers30(split="test", **kwargs)


@pytest.mark.parametrize(
    ("config_name", "image_ids", "labels"),
    [
        ("fold1", (1, 18, 1479), (0, 1, 29)),
        ("fold2", (18, 1479, 1), (1, 29, 0)),
        ("fold3", (1479, 1, 18), (29, 0, 1)),
    ],
)
def test_jena_flowers30_official_folds_use_numeric_filename_ids(local_archive, config_name, image_ids, labels):
    builder = object.__new__(JenaFlowers30)
    builder.__init__(config_name=config_name)
    source = builder._source()
    assert set(source["assets"]) == {"train", "validation", "test"}
    assert set(source["assets"].values()) == {JenaFlowers30.SOURCE["assets"]["train"]}
    with pytest.raises(TypeError):
        source["assets"]["test"] = "https://example.com/archive.zip"

    memberships = []
    for split, image_id, expected_label in zip(("train", "validation", "test"), image_ids, labels):
        examples = list(builder._generate_examples(data_path=local_archive, split=split))
        assert len(examples) == 1
        assert examples[0][0] == f"jpg/JFT_{image_id:05d}.jpg"
        assert examples[0][1]["label"] == expected_label
        assert examples[0][1]["image"].mode == "RGB"
        memberships.append({key for key, _ in examples})
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    assert set.union(*memberships) == {"jpg/JFT_00001.jpg", "jpg/JFT_00018.jpg", "jpg/JFT_01479.jpg"}


def test_jena_flowers30_fold_configs_have_separate_reusable_caches(local_archive, tmp_path, monkeypatch):
    downloads = []

    def local_download(urls, dest_folder):
        downloads.append(list(urls))
        return [local_archive]

    monkeypatch.setattr(utils, "bulk_download", local_download)
    kwargs = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    assert len(JenaFlowers30(split="train", **kwargs)) == 3
    expected = {"fold1": (0, 1, 29), "fold2": (1, 29, 0), "fold3": (29, 0, 1)}
    for config_name, labels in expected.items():
        datasets = JenaFlowers30(config_name=config_name, split=None, **kwargs)
        assert set(datasets) == {"train", "validation", "test"}
        for split, label in zip(("train", "validation", "test"), labels):
            assert len(datasets[split]) == 1
            assert datasets[split][0]["label"] == label
            assert datasets[split].info.config_name == config_name
    assert downloads == [[JenaFlowers30.SOURCE["assets"]["train"]]] * 4

    def no_regeneration(*args, **kwargs):
        pytest.fail("Each configuration should reuse its own prepared Arrow cache")

    monkeypatch.setattr(JenaFlowers30, "_generate_examples", no_regeneration)
    for config_name, labels in expected.items():
        dataset = JenaFlowers30(config_name=config_name, split="train", **kwargs)
        assert len(dataset) == 1
        assert dataset[0]["label"] == labels[0]
    assert len(JenaFlowers30(config_name="all", split="train", **kwargs)) == 3


@pytest.mark.parametrize(
    ("member", "contents", "error", "message"),
    [
        ("trn1.txt", None, KeyError, "trn1.txt"),
        ("trn1.txt", "", ValueError, "No image IDs"),
        ("trn1.txt", "1\n1\n", ValueError, "Duplicate"),
        ("trn1.txt", "0\n", ValueError, "Unknown image ID"),
        ("trn1.txt", "1480\n", ValueError, "Unknown image ID"),
        ("trn1.txt", "2\n", ValueError, "Unknown image ID"),
        ("trn1.txt", "1\n18\n", ValueError, "Overlapping"),
        (
            "classlabels.txt",
            "JFT_00018.jpg, 2\nJFT_01479.jpg, 30\nJFT_00001.jpg, 1\nJFT_00002.jpg, 1\n",
            ValueError,
            "Missing split membership",
        ),
    ],
)
def test_jena_flowers30_rejects_invalid_fold_membership(tmp_path, member, contents, error, message):
    metadata = {
        "classlabels.txt": "JFT_00018.jpg, 2\nJFT_01479.jpg, 30\nJFT_00001.jpg, 1\n",
        "trn1.txt": "1\n",
        "val1.txt": "18\n",
        "tst1.txt": "1479\n",
    }
    if contents is None:
        metadata.pop(member)
    else:
        metadata[member] = contents
    path = tmp_path / "invalid_fold.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in metadata.items():
            archive.writestr(name, text)
    builder = object.__new__(JenaFlowers30)
    builder.__init__(config_name="fold1")
    with pytest.raises(error, match=message):
        list(builder._generate_examples(data_path=path, split="train"))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("", "No labels"),
        ("JFT_00001.jpg, 0\n", "class ID"),
        ("JFT_00001.jpg, 31\n", "class ID"),
        ("JFT_00001.jpg, 1\nJFT_00001.jpg, 2\n", "Duplicate"),
    ],
)
def test_jena_flowers30_rejects_invalid_labels(tmp_path, metadata, message):
    path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classlabels.txt", metadata)
    builder = object.__new__(JenaFlowers30)
    builder.__init__()
    with pytest.raises(ValueError, match=message):
        list(builder._generate_examples(data_path=path, split="train"))


def test_jena_flowers30_missing_labelled_image_is_an_error(tmp_path):
    path = tmp_path / "missing.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classlabels.txt", "JFT_00001.jpg, 1\n")
    builder = object.__new__(JenaFlowers30)
    builder.__init__()
    with pytest.raises(KeyError, match="jpg/JFT_00001.jpg"):
        list(builder._generate_examples(data_path=path, split="train"))


@pytest.mark.large
@pytest.mark.parametrize("config_name", ["all", "fold1", "fold2", "fold3"])
def test_jena_flowers30_dataset(config_name):
    datasets = JenaFlowers30(config_name=config_name, split=None)
    counts = {"train": 1479} if config_name == "all" else {"train": 279, "validation": 279, "test": 921}
    assert set(datasets) == set(counts)
    for split, count in counts.items():
        dataset = datasets[split]
        assert len(dataset) == count
        assert dataset.features["label"].num_classes == 30
        assert set(dataset.table.column("label").to_pylist()) == set(range(30))
        sample = dataset[0]
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        image = np.asarray(sample["image"])
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 30
    assert len(JenaFlowers30(config_name=config_name, split="train")) == counts["train"]
