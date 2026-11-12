"""Offline Stanford40 tests plus a real-data check."""

import io
import zipfile
from collections import Counter

import pytest
from PIL import Image

from stable_datasets.images.stanford40 import Stanford40, _to_rgb_bytes


def _builder():
    builder = object.__new__(Stanford40)
    Stanford40.__init__(builder)
    return builder


def _write_splits(path, train, test):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ImageSplits/train.txt", "\n".join(train) + "\n")
        archive.writestr("ImageSplits/test.txt", "\n".join(test) + "\n")
    return path


@pytest.fixture
def archives(tmp_path):
    images_path = tmp_path / "images.zip"
    # Deliberately differ from both alphabetical and archive order.
    train = ["writing_on_a_book_001.jpg", "applauding_001.jpg"]
    test = ["applauding_002.jpg"]
    with zipfile.ZipFile(images_path, "w") as archive:
        for filename, size in [
            ("applauding_001.jpg", (11, 7)),
            ("applauding_002.jpg", (9, 5)),
            ("writing_on_a_book_001.jpg", (13, 8)),
        ]:
            buffer = io.BytesIO()
            Image.new("RGB", size, color=(50, 100, 150)).save(buffer, format="JPEG")
            archive.writestr(f"JPEGImages/{filename}", buffer.getvalue())
    splits_path = _write_splits(tmp_path / "splits.zip", train, test)
    return images_path, splits_path


def test_examples_follow_split_order_and_labels(archives):
    builder = _builder()
    train = list(builder._generate_examples(*archives, split="train"))
    test = list(builder._generate_examples(*archives, split="test"))
    assert [key for key, _ in train] == ["writing_on_a_book_001.jpg", "applauding_001.jpg"]
    assert [sample["label"] for _, sample in train] == [39, 0]
    assert [key for key, _ in test] == ["applauding_002.jpg"]
    assert test[0][1]["label"] == 0
    for (_, sample), size in zip(train + test, [(13, 8), (11, 7), (9, 5)]):
        assert set(sample) == {"image", "label"}
        with Image.open(io.BytesIO(sample["image"])) as image:
            image.load()
            assert image.size == size
            assert image.mode == "RGB"


def test_metadata():
    builder = _builder()
    info = builder.info
    assert str(Stanford40.VERSION) == "1.0.0"
    assert info.description
    assert info.homepage == Stanford40.SOURCE["homepage"]
    assert info.citation == Stanford40.SOURCE["citation"]
    assert info.license == Stanford40.SOURCE["license"] == "Unknown"
    assert info.supervised_keys == ("image", "label")
    assert set(info.features) == {"image", "label"}
    names = info.features["label"].names
    assert len(names) == len(set(names)) == 40
    assert names == sorted(names)
    assert names[0] == "applauding"
    assert names[39] == "writing_on_a_book"
    assert "watching_TV" in names
    assert "throwing_frisby" in names
    assert builder._candidate_splits() == ["train", "test"]


def test_assets_have_checksummed_archive_fallbacks():
    for asset in Stanford40.SOURCE["assets"].values():
        assert asset.url.startswith("http://vision.stanford.edu/Datasets/")
        assert len(asset.fallbacks) == 1
        assert asset.fallbacks[0].startswith("https://web.archive.org/web/")
        assert asset.fallbacks[0].endswith(asset.url.rsplit("/", 1)[1])
        assert asset.checksum.startswith("sha256:")


@pytest.mark.parametrize(
    "train,test,error",
    [
        ([], ["applauding_002.jpg"], "Empty"),
        (["applauding_001.jpg"], [], "Empty"),
        (["applauding_001.jpg"] * 2, ["applauding_002.jpg"], "Duplicate"),
        (["applauding_001.jpg"], ["applauding_002.jpg"] * 2, "Duplicate"),
        (["applauding_001.jpg"], ["applauding_001.jpg"], "overlap"),
        (["../applauding_001.jpg"], ["applauding_002.jpg"], "filename"),
        (["folder\\applauding_001.jpg"], ["applauding_002.jpg"], "filename"),
        (["applauding_001.png"], ["applauding_002.jpg"], "filename"),
        (["unknown_action_001.jpg"], ["applauding_002.jpg"], "action"),
        (["applauding_bad.jpg"], ["applauding_002.jpg"], "index"),
    ],
)
def test_invalid_split_contents(tmp_path, train, test, error):
    path = _write_splits(tmp_path / "bad.zip", train, test)
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError, match=error):
        _builder()._read_splits(archive)


def _jpeg(mode, size, color):
    buffer = io.BytesIO()
    Image.new(mode, size, color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_rgb_jpeg_bytes_are_kept_unchanged():
    data = _jpeg("RGB", (6, 4), (10, 20, 30))
    assert _to_rgb_bytes(data) is data


@pytest.mark.parametrize("mode,color", [("L", 128), ("CMYK", (0, 50, 100, 0))])
def test_non_rgb_images_are_converted_to_rgb(tmp_path, mode, color):
    images_path = tmp_path / "images.zip"
    with zipfile.ZipFile(images_path, "w") as archive:
        archive.writestr("JPEGImages/feeding_a_horse_191.jpg", _jpeg(mode, (7, 5), color))
        archive.writestr("JPEGImages/applauding_001.jpg", _jpeg("RGB", (7, 5), (1, 2, 3)))
    splits_path = _write_splits(tmp_path / "splits.zip", ["feeding_a_horse_191.jpg"], ["applauding_001.jpg"])
    ((_, sample),) = _builder()._generate_examples(images_path, splits_path, "train")
    with Image.open(io.BytesIO(sample["image"])) as image:
        assert image.mode == "RGB"
        assert image.size == (7, 5)
    assert sample["label"] == 9


def test_unknown_split(archives):
    with pytest.raises(ValueError, match="Unknown Stanford40 split"):
        list(_builder()._generate_examples(*archives, split="validation"))


def test_missing_image(archives, tmp_path):
    splits = _write_splits(tmp_path / "missing.zip", ["applauding_999.jpg"], ["applauding_002.jpg"])
    with pytest.raises(KeyError, match="applauding_999.jpg"):
        list(_builder()._generate_examples(archives[0], splits, "train"))


def test_loading_and_cache_reuse(archives, tmp_path, monkeypatch):
    calls = []

    def fake_download(specs, dest_folder):
        assert len(specs) == 2
        calls.append(dest_folder)
        return list(archives)

    monkeypatch.setattr("stable_datasets.images.stanford40.bulk_download", fake_download)
    kwargs = {
        "download_dir": tmp_path / "raw",
        "processed_cache_dir": tmp_path / "processed",
    }
    datasets = Stanford40(**kwargs)
    assert set(datasets.keys()) == {"train", "test"}
    assert calls == [kwargs["download_dir"]]
    for split, labels, sizes in [
        ("train", [39, 0], [(13, 8), (11, 7)]),
        ("test", [0], [(9, 5)]),
    ]:
        assert len(datasets[split]) == len(labels)
        for index, (label, size) in enumerate(zip(labels, sizes)):
            sample = datasets[split][index]
            assert set(sample) == {"image", "label"}
            assert sample["label"] == label
            assert isinstance(sample["image"], Image.Image)
            sample["image"].load()
            assert sample["image"].size == size

    def no_download(*args, **kwargs):
        pytest.fail("A cached dataset must not download again")

    monkeypatch.setattr("stable_datasets.images.stanford40.bulk_download", no_download)
    for split, count, first_label in [("train", 2, 39), ("test", 1, 0)]:
        reopened = Stanford40(split=split, **kwargs)
        assert len(reopened) == count
        assert reopened[0]["label"] == first_label
        reopened[0]["image"].load()
    with pytest.raises(ValueError, match="not found"):
        Stanford40(split="validation", **kwargs)


def test_stanford40_real_data():
    """Download if needed; decode every image and check official split sizes."""
    datasets = Stanford40()
    assert set(datasets.keys()) == {"train", "test"}
    for split, count in [("train", 4000), ("test", 5532)]:
        dataset = datasets[split]
        assert len(dataset) == count
        labels = Counter()
        for index, sample in enumerate(dataset):
            context = f"{split}[{index}]"
            assert set(sample) == {"image", "label"}, context
            assert isinstance(sample["image"], Image.Image), context
            sample["image"].load()
            assert sample["image"].mode == "RGB", context
            width, height = sample["image"].size
            assert width > 0 and height > 0, context
            label = sample["label"]
            assert isinstance(label, int) and 0 <= label < 40, context
            labels[label] += 1
        assert set(labels) == set(range(40))
        if split == "train":
            assert set(labels.values()) == {100}
