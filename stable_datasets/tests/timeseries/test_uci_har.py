import io
import zipfile

import numpy as np
import pytest

from stable_datasets.timeseries import UCIHAR


def _archive(tmp_path, replacement=None):
    members = {}
    for split, labels, subjects in [("train", [1, 6], [1, 3]), ("test", [2], [2])]:
        prefix = f"UCI HAR Dataset/{split}/"
        members[f"{prefix}y_{split}.txt"] = "\n".join(map(str, labels))
        members[f"{prefix}subject_{split}.txt"] = "\n".join(map(str, subjects))
        for channel, name in enumerate(UCIHAR.CHANNEL_NAMES):
            rows = [np.arange(128) + channel * 10000 + row * 1000 for row in range(len(labels))]
            members[f"{prefix}Inertial Signals/{name}_{split}.txt"] = "\n".join(
                " ".join(map(str, row)) for row in rows
            )
    if replacement:
        members.update(replacement)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    path = tmp_path / "uci_har.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("UCI HAR Dataset.zip", inner.getvalue())
    return path


def _builder():
    builder = object.__new__(UCIHAR)
    builder.__init__()
    return builder


def test_uci_har_splits_values_and_cache(tmp_path, monkeypatch):
    archive = _archive(tmp_path)

    def download(specs, dest_folder):
        assert len(specs) == 1  # The two splits share one download.
        assert specs[0].checksum.startswith("sha256:")
        return [archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    cache = tmp_path / "cache"
    data = UCIHAR(processed_cache_dir=cache, download_dir=tmp_path)
    assert set(data) == {"train", "test"}
    assert [len(data[split]) for split in ("train", "test")] == [2, 1]
    first = data["train"][0]
    assert set(first) == {"series", "label", "subject_id"}
    values = np.asarray(first["series"])
    assert values.shape == (128, 9)
    np.testing.assert_array_equal(values[:, 0], np.arange(128))
    np.testing.assert_array_equal(values[:, 8], np.arange(128) + 80000)
    assert [row["label"] for row in data["train"]] == [0, 5]
    assert data["test"][0]["label"] == 1
    assert {row["subject_id"] for row in data["train"]} == {1, 3}
    assert {row["subject_id"] for row in data["test"]} == {2}
    assert _builder().info.features["series"].feature.feature.dtype == "float32"

    def no_download(*args, **kwargs):
        pytest.fail("A complete processed cache must not download again")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    cached = UCIHAR(split="train", processed_cache_dir=cache, download_dir=tmp_path)
    np.testing.assert_array_equal(cached[1]["series"], data["train"][1]["series"])
    with pytest.raises(ValueError, match="Split 'validation' not found"):
        UCIHAR(split="validation", processed_cache_dir=cache, download_dir=tmp_path)


@pytest.mark.parametrize(
    "member, content, message",
    [
        ("y_train.txt", "0\n6", "labels"),
        ("y_train.txt", "1.5\n6", "invalid literal"),
        ("subject_train.txt", "1", "subject"),
        ("subject_train.txt", "1\n31", "subject"),
        ("Inertial Signals/body_acc_x_train.txt", "1 2\n3 4", "shape"),
        ("Inertial Signals/body_acc_x_train.txt", " ".join(["1"] * 128), "shape"),
        ("Inertial Signals/body_acc_x_train.txt", "\n".join([" ".join(["nan"] * 128)] * 2), "finite"),
    ],
)
def test_uci_har_rejects_invalid_data(tmp_path, member, content, message):
    archive = _archive(tmp_path, {f"UCI HAR Dataset/train/{member}": content})
    with pytest.raises(ValueError, match=message):
        list(_builder()._generate_examples(archive, "train"))
