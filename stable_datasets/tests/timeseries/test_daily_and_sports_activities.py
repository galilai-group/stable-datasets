"""Offline parser and cache tests for the subject-disjoint activity builder."""

import io
import zipfile
from collections import Counter

import numpy as np
import pytest

from stable_datasets.schema import ClassLabel, Sequence
from stable_datasets.timeseries.daily_and_sports_activities import DailyAndSportsActivities


def _builder():
    builder = object.__new__(DailyAndSportsActivities)
    builder.__init__()
    return builder


def _csv(shape=(125, 45), value=None):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if value is not None:
        data[0, 0] = value
    stream = io.StringIO()
    np.savetxt(stream, data, delimiter=",")
    return stream.getvalue()


def _archive(tmp_path, members):
    path = tmp_path / "activities.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return path


def test_metadata():
    info = _builder().info
    assert isinstance(info.features["series"], Sequence)
    assert isinstance(info.features["label"], ClassLabel)
    assert info.features["label"].num_classes == 19
    assert info.license == "CC BY 4.0"
    assert len(DailyAndSportsActivities.CHANNEL_NAMES) == 45
    assert DailyAndSportsActivities.CHANNEL_NAMES[:3] == ("torso_acc_x", "torso_acc_y", "torso_acc_z")
    assert not DailyAndSportsActivities.SPLIT_SUBJECTS["train"] & DailyAndSportsActivities.SPLIT_SUBJECTS["test"]


def test_parser_preserves_labels_axes_and_subjects(tmp_path):
    path = _archive(
        tmp_path,
        {"data/a01/p1/s01.txt": _csv(), "data/a19/p7/s60.txt": _csv(), "README.txt": "ignored"},
    )
    train = list(_builder()._generate_examples(path, "train"))
    test = list(_builder()._generate_examples(path, "test"))
    assert len(train) == len(test) == 1
    key, row = train[0]
    assert key == "a01/p1/s01"
    assert row["label"] == 0 and row["subject_id"] == 1 and row["segment_id"] == 1
    assert row["series"].shape == (125, 45) and row["series"].dtype == np.float32
    assert row["series"][1, 0] == 45 and row["series"][0, 1] == 1
    assert test[0][1]["label"] == 18 and test[0][1]["subject_id"] == 7
    assert test[0][1]["segment_id"] == 60


@pytest.mark.parametrize("shape", [(124, 45), (125, 44), (126, 45), (125, 46)])
def test_reject_wrong_shape(tmp_path, shape):
    path = _archive(tmp_path, {"data/a01/p1/s01.txt": _csv(shape)})
    with pytest.raises(ValueError, match="Expected"):
        list(_builder()._generate_examples(path, "train"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_reject_nonfinite(tmp_path, value):
    path = _archive(tmp_path, {"data/a01/p1/s01.txt": _csv(value=value)})
    with pytest.raises(ValueError, match="Non-finite"):
        list(_builder()._generate_examples(path, "train"))


@pytest.mark.parametrize("name", ["a00/p1/s01.txt", "a20/p1/s01.txt", "a01/p9/s01.txt", "a01/p1/s61.txt"])
def test_reject_bad_identifiers(tmp_path, name):
    path = _archive(tmp_path, {name: _csv()})
    with pytest.raises(ValueError, match="Invalid"):
        list(_builder()._generate_examples(path, "train"))


def test_reject_duplicate_window(tmp_path):
    path = _archive(tmp_path, {"data/a01/p1/s01.txt": _csv(), "other/a01/p1/s01.txt": _csv()})
    with pytest.raises(ValueError, match="Duplicate"):
        list(_builder()._generate_examples(path, "train"))


def test_reject_empty_and_unknown_split(tmp_path):
    path = _archive(tmp_path, {"README.txt": "not sensor data"})
    with pytest.raises(ValueError, match="No Daily"):
        list(_builder()._generate_examples(path, "train"))
    with pytest.raises(ValueError, match="Unknown split"):
        list(_builder()._generate_examples(path, "validation"))


def test_cache_reload_and_formats(tmp_path, monkeypatch):
    path = _archive(tmp_path, {"data/a01/p1/s01.txt": _csv(), "data/a19/p8/s60.txt": _csv()})
    calls = []

    def download(specs, dest_folder):
        calls.append(specs)
        return [path]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    data = DailyAndSportsActivities(processed_cache_dir=tmp_path / "cache", download_dir=tmp_path / "raw")
    assert set(data) == {"train", "test"}
    assert len(calls) == 1 and len(calls[0]) == 1
    assert np.asarray(data["train"][0]["series"]).shape == (125, 45)
    assert data["test"][0]["subject_id"] == 8
    assert data["train"].with_format("torch")[0]["label"].item() == 0
    assert np.asarray(data["test"].with_format("numpy")[0]["series"]).shape == (125, 45)
    reloaded = DailyAndSportsActivities(split="test", processed_cache_dir=tmp_path / "cache")
    assert len(calls) == 1
    assert reloaded[0]["label"] == 18


def test_official_download_and_full_subject_split(tmp_path):
    """Exercise the real 163 MB UCI download and all original windows."""
    raw = tmp_path / "raw"
    kwargs = {"processed_cache_dir": tmp_path / "cache", "download_dir": raw}
    dataset = DailyAndSportsActivities(**kwargs)
    assert set(dataset) == {"train", "test"}
    assert len(dataset["train"]) == 6840
    assert len(dataset["test"]) == 2280
    archives = list(raw.glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        for split, subjects, per_class in [("train", set(range(1, 7)), 360), ("test", {7, 8}, 120)]:
            counts, people, identifiers = Counter(), set(), set()
            for row in dataset[split]:
                signal = np.asarray(row["series"], dtype=np.float32)
                assert signal.shape == (125, 45) and np.isfinite(signal).all()
                assert 0 <= row["label"] < 19
                assert 1 <= row["segment_id"] <= 60
                counts[row["label"]] += 1
                people.add(row["subject_id"])
                identifiers.add((row["label"], row["subject_id"], row["segment_id"]))
            assert counts == dict.fromkeys(range(19), per_class)
            assert people == subjects
            assert len(identifiers) == len(dataset[split])
            for index in [0, len(dataset[split]) - 1]:
                row = dataset[split][index]
                suffix = f"a{row['label'] + 1:02d}/p{row['subject_id']}/s{row['segment_id']:02d}.txt"
                members = [name for name in archive.namelist() if name.endswith(suffix)]
                assert len(members) == 1
                with archive.open(members[0]) as stream:
                    expected = np.loadtxt(stream, delimiter=",", dtype=np.float32)
                np.testing.assert_array_equal(np.asarray(row["series"], dtype=np.float32), expected)
    reloaded = DailyAndSportsActivities(split="test", **kwargs)
    assert len(reloaded) == 2280
    assert reloaded[0] == dataset["test"][0]
