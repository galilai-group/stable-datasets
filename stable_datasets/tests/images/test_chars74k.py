import io
import tarfile

import pytest
from PIL import Image

from stable_datasets.images.chars74k import Chars74K


def make_test_tar(path, folder_ids=(1, 11, 62), sample_ids=(1, 2, 5, 6)):
    """Create a small Chars74K-like archive."""
    with tarfile.open(path, "w:gz") as archive:
        for folder_id in folder_ids:
            for sample_id in sample_ids:
                image = Image.new("RGB", (32, 40), color="white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data = buffer.getvalue()

                name = f"English/Hnd/Img/Sample{folder_id:03d}/img{folder_id:03d}-{sample_id:03d}.png"

                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def test_label_mapping():
    assert len(Chars74K.LABELS) == 62
    assert Chars74K.LABELS[0] == "0"
    assert Chars74K.LABELS[9] == "9"
    assert Chars74K.LABELS[10] == "A"
    assert Chars74K.LABELS[35] == "Z"
    assert Chars74K.LABELS[36] == "a"
    assert Chars74K.LABELS[61] == "z"


def test_train_test_splits(tmp_path):
    path = tmp_path / "test.tgz"
    make_test_tar(path)

    builder = object.__new__(Chars74K)

    train = list(builder._generate_examples(path, "train"))
    test = list(builder._generate_examples(path, "test"))

    assert len(train) == 6
    assert len(test) == 6

    train_ids = {key for key, _ in train}
    test_ids = {key for key, _ in test}
    assert train_ids.isdisjoint(test_ids)

    assert all(0 <= sample["label"] < 62 for _, sample in train + test)
    assert all(sample["image"].mode == "RGB" for _, sample in train + test)


def test_invalid_split(tmp_path):
    path = tmp_path / "test.tgz"
    make_test_tar(path)

    builder = object.__new__(Chars74K)

    with pytest.raises(ValueError, match="Unsupported split"):
        list(builder._generate_examples(path, "validation"))


def test_invalid_class_path(tmp_path):
    path = tmp_path / "invalid.tgz"

    image = Image.new("RGB", (32, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()

    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("English/Hnd/Img/Sample063/img063-001.png")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    builder = object.__new__(Chars74K)

    with pytest.raises(ValueError, match="Invalid class"):
        list(builder._generate_examples(path, "test"))
