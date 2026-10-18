"""Offline parsing, schema and cache checks for all robot tasks."""

import zipfile

import numpy as np
import pytest

from stable_datasets.timeseries import RobotExecutionFailures


def builder(config="lp1"):
    obj = object.__new__(RobotExecutionFailures)
    obj.__init__(config_name=config)
    return obj


def record(label="normal", rows=15, columns=6, value=None):
    x = np.arange(rows * columns).reshape(rows, columns).astype(str)
    if value is not None:
        x[0, 0] = value
    return label + "\n" + "\n".join("\t".join(row) for row in x) + "\n\n"


def archive(tmp_path, members):
    path = tmp_path / "robot.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return path


@pytest.mark.parametrize("config", RobotExecutionFailures.CLASS_NAMES)
def test_every_configuration_preserves_axes_and_labels(tmp_path, config):
    obj = builder(config)
    names = RobotExecutionFailures.CLASS_NAMES[config]
    path = archive(tmp_path, {f"nested/{config}.data": record(names[0]) + record(names[-1]), "a.out": "unused"})
    rows = list(obj._generate_examples(path, "all"))
    assert obj.info.features["label"].names == list(names)
    assert [key for key, row in rows] == [0, 1]
    assert [row["label"] for key, row in rows] == [0, len(names) - 1]
    assert rows[1][1]["sample_id"] == 1
    x = rows[0][1]["series"]
    assert x.shape == (15, 6) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0], [0, 1, 2, 3, 4, 5])
    assert x[1, 0] == 6


@pytest.mark.parametrize(
    "content",
    [
        "",
        record(rows=14),
        record(rows=16),
        record(columns=5),
        record(columns=7),
        record(label="unknown"),
        record(value="nan"),
        record(value="inf"),
        record(value="0.5"),
        record(value="16777217"),
    ],
)
def test_rejects_malformed_records(tmp_path, content):
    path = archive(tmp_path, {"lp1.data": content})
    with pytest.raises(ValueError):
        list(builder()._generate_examples(path, "all"))


def test_rejects_missing_duplicate_members_and_invalid_split(tmp_path):
    obj = builder()
    for members in [{"other.data": record()}, {"a/lp1.data": record(), "b/lp1.data": record()}]:
        with pytest.raises(ValueError, match="exactly one"):
            list(obj._generate_examples(archive(tmp_path, members), "all"))
    with pytest.raises(ValueError, match="unsplit"):
        list(obj._generate_examples(None, "train"))
    with pytest.raises(ValueError):
        builder("not-a-task")


def test_cache_keeps_configurations_separate(tmp_path, monkeypatch):
    path = archive(tmp_path, {"lp1.data": record(), "lp3.data": record("lost")})
    calls = []

    def download(specs, dest_folder):
        calls.append(specs)
        return [path]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    kwargs = {"processed_cache_dir": tmp_path / "cache", "download_dir": tmp_path / "raw", "split": "all"}
    first = RobotExecutionFailures(config_name="lp1", **kwargs)
    third = RobotExecutionFailures(config_name="lp3", **kwargs)
    assert first[0]["label"] == 0 and third[0]["label"] == 3
    assert third.with_format("torch")[0]["sample_id"].item() == 0
    assert np.asarray(first.with_format("numpy")[0]["series"]).shape == (15, 6)
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda *a, **k: pytest.fail("Downloaded cached data"))
    reloaded = RobotExecutionFailures(config_name="lp3", **kwargs)
    assert reloaded[0]["label"] == 3 and len(calls) == 2
