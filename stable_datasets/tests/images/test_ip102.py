import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import stable_datasets.images.ip102 as ip102_module
from stable_datasets.images.ip102 import IP102
from stable_datasets.schema import ClassLabel


def make_archive(path, lists=None, missing_image=False):
    lists = lists or {
        "train": "00001.jpg 0\n00002.jpg 101\n",
        "val": "00003.jpg 1\n",
        "test": "00004.jpg 2\n",
    }
    with tarfile.open(path, "w") as archive:
        for number, mode in [(4, "CMYK"), (2, "RGB"), (3, "RGB"), (1, "L")]:
            if missing_image and number == 1:
                continue
            buffer = io.BytesIO()
            Image.new(mode, (11 + number, 9)).save(buffer, format="JPEG")
            content = buffer.getvalue()
            entry = tarfile.TarInfo(f"Images/{number:05d}.jpg")
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    paths = {"archive": path}
    for split, lines in lists.items():
        paths[split] = path.with_name(f"{path.stem}_{split}.txt")
        paths[split].write_text(lines)
    return paths


@pytest.fixture
def source_paths(tmp_path):
    return make_archive(tmp_path / "source.tar")


@pytest.fixture
def builder(tmp_path):
    instance = object.__new__(IP102)
    instance.__init__()
    instance._raw_download_dir = tmp_path / "downloads"
    instance._processed_cache_dir = tmp_path / "processed"
    return instance


def test_ip102_metadata(builder):
    assert set(builder.info.features) == {"image", "label"}
    assert builder.info.supervised_keys == ("image", "label")
    labels = builder.info.features["label"]
    assert isinstance(labels, ClassLabel)
    assert labels.num_classes == 102
    assert labels.names[0] == "rice leaf roller"
    assert labels.names[-1] == "Cicadellidae"
    assert builder.info.homepage == "https://github.com/xpwu95/IP102"
    with pytest.raises(TypeError):
        IP102.SOURCE["assets"]["archive"] = "https://example.com"


@pytest.mark.parametrize(("split", "labels"), [("train", [0, 101]), ("val", [1]), ("test", [2])])
def test_ip102_uses_official_lists_and_zero_based_labels(builder, source_paths, split, labels):
    rows = list(builder._generate_examples(source_paths, split))
    assert [sample["label"] for _, sample in rows] == labels
    for key, sample in rows:
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["label"], int)
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        number = int(Path(key).stem)
        assert sample["image"].size == (11 + number, 9)
        assert np.asarray(sample["image"]).dtype == np.uint8


def test_ip102_public_builder_download_and_cache(source_paths, tmp_path, monkeypatch):
    calls = []

    def download(url, output, **kwargs):
        calls.append(url)
        key = next(key for key, value in IP102.SOURCE["assets"].items() if value == url)
        shutil.copyfile(source_paths[key], output)
        return output

    monkeypatch.setattr(ip102_module.gdown, "download", download)
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "processed"}
    datasets = IP102(split=None, **kwargs)
    assert set(datasets) == {"train", "validation", "test"}
    assert {split: len(data) for split, data in datasets.items()} == {"train": 2, "validation": 1, "test": 1}
    assert datasets["train"][0]["label"] == 0
    assert datasets["train"][1]["label"] == 101
    assert set(calls) == set(IP102.SOURCE["assets"].values())
    assert len(calls) == 4

    def no_download(*args, **kwargs):
        pytest.fail("A prepared dataset should not download again")

    monkeypatch.setattr(ip102_module.gdown, "download", no_download)
    assert len(IP102(split="test", **kwargs)) == 1
    assert len(IP102(split="validation", **kwargs)) == 1
    assert IP102(split="train", **kwargs)[0]["image"].mode == "RGB"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ("", "Empty"),
        ("00001.jpg -1\n", "label"),
        ("00001.jpg 102\n", "label"),
        ("00001.jpg 0\n00001.jpg 1\n", "Duplicate"),
        ("00001.jpg\n", "Invalid"),
        ("../00001.jpg 0\n", "filename"),
    ],
)
def test_ip102_rejects_invalid_split_lists(builder, tmp_path, lines, message):
    path = make_archive(tmp_path / "invalid.tar", {"train": lines, "val": "00003.jpg 1\n", "test": "00004.jpg 2\n"})
    with pytest.raises(ValueError, match=message):
        list(builder._generate_examples(path, "train"))


def test_ip102_rejects_overlapping_splits(builder, tmp_path):
    path = make_archive(
        tmp_path / "overlap.tar", {"train": "00001.jpg 0\n", "val": "00001.jpg 0\n", "test": "00004.jpg 2\n"}
    )
    with pytest.raises(ValueError, match="multiple splits"):
        list(builder._generate_examples(path, "train"))


def test_ip102_missing_image_is_an_error(builder, tmp_path):
    path = make_archive(tmp_path / "missing.tar", missing_image=True)
    with pytest.raises(KeyError, match="00001.jpg"):
        list(builder._generate_examples(path, "train"))


def test_ip102_failed_download_is_not_cached(builder, monkeypatch):
    monkeypatch.setattr(ip102_module.gdown, "download", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="IP102"):
        builder._split_generators()
    assert not (Path(builder._raw_download_dir) / "ip102" / "Images.tar").exists()


def test_ip102_label_comes_from_manifest_not_filename(builder, tmp_path):
    path = tmp_path / "filename.tar"
    paths = make_archive(path, {"train": "IP022001503.jpg 22\n", "val": "00003.jpg 1\n", "test": "00004.jpg 2\n"})
    buffer = io.BytesIO()
    Image.new("RGB", (17, 9)).save(buffer, format="JPEG")
    with tarfile.open(path, "a") as archive:
        member = tarfile.TarInfo("Images/IP022001503.jpg")
        member.size = len(buffer.getvalue())
        archive.addfile(member, io.BytesIO(buffer.getvalue()))
    rows = list(builder._generate_examples(paths, "train"))
    assert rows[0][1]["label"] == 22
    assert builder.info.features["label"].int2str(22) == "corn borer"


@pytest.mark.large
def test_ip102_dataset():
    datasets = IP102(split=None)
    assert set(datasets) == {"train", "validation", "test"}
    for split, count in [("train", 45095), ("validation", 7508), ("test", 22619)]:
        dataset = datasets[split]
        assert len(dataset) == count
        assert dataset.features["label"].num_classes == 102
        assert set(dataset.table.column("label").to_pylist()) == set(range(102))
        for index in [0, len(dataset) - 1]:
            sample = dataset[index]
            assert set(sample) == {"image", "label"}
            assert sample["image"].mode == "RGB"
            assert np.asarray(sample["image"]).dtype == np.uint8
            assert 0 <= sample["label"] < 102
