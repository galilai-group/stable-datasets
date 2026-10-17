import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import Omniglot


def _archive(tmp_path, split, names=None):
    names = names or (
        ["Alpha/character01/0001_02.png", "Alpha/character01/0001_01.png", "Beta/character01/0002_01.png"]
        if split == "background"
        else ["Gamma/character01/0965_01.png"]
    )
    array = np.full((105, 105), 255, dtype=np.uint8)
    array[10:30, 20:40] = 0
    buffer = io.BytesIO()
    Image.fromarray(array).convert("1").save(buffer, format="PNG")
    path = tmp_path / f"images_{split}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(f"images_{split}/{name}", buffer.getvalue())
        archive.writestr("__MACOSX/._ignored.png", b"not an image")
        archive.writestr(f"images_{split}/README.txt", "fixture")
    return path, array


def test_pixels_global_labels_and_metadata(tmp_path):
    path, pixels = _archive(tmp_path, "background")
    rows = list(object.__new__(Omniglot)._generate_examples(path, "background"))
    assert len(rows) == 3
    assert [row["label"] for _, row in rows] == [0, 0, 1]
    assert [row["example_id"] for _, row in rows] == [1, 2, 1]
    for _, row in rows:
        assert row["image"].mode == "L"
        np.testing.assert_array_equal(row["image"], pixels)
    assert rows[0][1]["character"] == "Alpha/character01"
    assert rows[2][1]["character"] == "Beta/character01"
    assert rows[0][1]["filename"] == "Alpha/character01/0001_01.png"
    path, _ = _archive(tmp_path, "evaluation")
    row = next(object.__new__(Omniglot)._generate_examples(path, "evaluation"))[1]
    assert row["label"] == 964  # evaluation must not restart at label zero


@pytest.mark.parametrize("filename", ["0001_01.png", "0965_00.png", "0965_21.png", "bad.png"])
def test_invalid_evaluation_ids_rejected(tmp_path, filename):
    path, _ = _archive(tmp_path, "evaluation", [f"Alpha/character01/{filename}"])
    with pytest.raises(ValueError, match="Omniglot"):
        list(object.__new__(Omniglot)._generate_examples(path, "evaluation"))


def test_public_api_tensor_range_and_cache(tmp_path, monkeypatch):
    bg, pixels = _archive(tmp_path, "background")
    ev, _ = _archive(tmp_path, "evaluation")
    paths = {Omniglot.SOURCE.assets["background"].url: bg, Omniglot.SOURCE.assets["evaluation"].url: ev}
    monkeypatch.setattr("stable_datasets.images.omniglot.download", lambda asset, **kwargs: paths[asset.url])
    kwargs = {"download_dir": tmp_path / "raw", "processed_cache_dir": tmp_path / "processed"}
    datasets = Omniglot(**kwargs)
    assert {key: len(value) for key, value in datasets.items()} == {"background": 3, "evaluation": 1}
    np.testing.assert_array_equal(datasets["evaluation"][0]["image"], pixels)
    tensor = datasets["evaluation"].with_format("torch")[0]["image"]
    assert tuple(tensor.shape) == (1, 105, 105)
    assert float(tensor.min()) == 0.0
    assert float(tensor.max()) == 1.0

    def no_download(*args, **kwargs):
        pytest.fail("Reopening a complete processed cache must not download")

    monkeypatch.setattr("stable_datasets.images.omniglot.download", no_download)
    assert Omniglot(split="evaluation", **kwargs)[0]["label"] == 964


@pytest.mark.large
def test_omniglot_real_download(tmp_path):
    datasets = Omniglot(download_dir=tmp_path / "raw", processed_cache_dir=tmp_path / "processed")
    assert {key: len(value) for key, value in datasets.items()} == {"background": 19280, "evaluation": 13180}
    class_sets = []
    for split, expected_alphabets, expected_classes in (("background", 30, 964), ("evaluation", 20, 659)):
        labels, alphabets = {}, set()
        for row in datasets[split]:
            assert row["image"].size == (105, 105)
            assert row["image"].mode == "L"
            labels[row["label"]] = labels.get(row["label"], 0) + 1
            alphabets.add(row["alphabet"])
        assert len(labels) == expected_classes
        assert set(labels.values()) == {20}
        assert len(alphabets) == expected_alphabets
        class_sets.append(set(labels))
    assert class_sets[0].isdisjoint(class_sets[1])
