"""Offline parsing, schema and cache checks for all robot tasks."""

import zipfile
from collections import Counter

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
    rows = list(obj._generate_examples(path, "train"))
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
        list(builder()._generate_examples(path, "train"))


def test_rejects_missing_duplicate_members_and_invalid_split(tmp_path):
    obj = builder()
    for members in [{"other.data": record()}, {"a/lp1.data": record(), "b/lp1.data": record()}]:
        with pytest.raises(ValueError, match="exactly one"):
            list(obj._generate_examples(archive(tmp_path, members), "train"))
    with pytest.raises(ValueError, match="unsplit"):
        list(obj._generate_examples(None, "all"))
    with pytest.raises(ValueError):
        builder("not-a-task")


def test_cache_keeps_configurations_separate(tmp_path, monkeypatch):
    path = archive(tmp_path, {"lp1.data": record(), "lp3.data": record("lost")})
    calls = []

    def download(specs, dest_folder):
        calls.append(specs)
        return [path]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", download)
    kwargs = {"processed_cache_dir": tmp_path / "cache", "download_dir": tmp_path / "raw", "split": "train"}
    first = RobotExecutionFailures(config_name="lp1", **kwargs)
    third = RobotExecutionFailures(config_name="lp3", **kwargs)
    assert first[0]["label"] == 0 and third[0]["label"] == 3
    assert third.with_format("torch")[0]["sample_id"].item() == 0
    assert np.asarray(first.with_format("numpy")[0]["series"]).shape == (15, 6)
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda *a, **k: pytest.fail("Downloaded cached data"))
    reloaded = RobotExecutionFailures(config_name="lp3", **kwargs)
    assert reloaded[0]["label"] == 3 and len(calls) == 2


@pytest.fixture(scope="module")
def official_cache(tmp_path_factory):
    """Share one genuine UCI download across all five configurations."""
    root = tmp_path_factory.mktemp("official_robot")
    return {"processed_cache_dir": root / "cache", "download_dir": root / "raw"}


@pytest.mark.parametrize(
    ("config", "counts"),
    [
        ("lp1", [21, 17, 34, 16]),
        ("lp2", [20, 7, 6, 5, 9]),
        ("lp3", [20, 15, 9, 3]),
        ("lp4", [24, 72, 21]),
        ("lp5", [44, 26, 47, 21, 26]),
    ],
)
def test_official_download_counts_and_exact_measurements(official_cache, config, counts):
    """Validate real archive members, every label and all 90 measurements."""
    dataset = RobotExecutionFailures(config_name=config, **official_cache)
    assert set(dataset) == {"train"}
    data = dataset["train"]
    assert len(data) == sum(counts)
    archives = list(official_cache["download_dir"].glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        lines = [line.strip() for line in archive.read(f"{config}.data").decode("ascii").splitlines() if line.strip()]
    assert len(lines) == 16 * sum(counts)
    observed = Counter()
    for index, row in enumerate(data):
        block = lines[index * 16 : (index + 1) * 16]
        expected = np.array([[int(value) for value in line.split()] for line in block[1:]], dtype=np.float32)
        actual = np.asarray(row["series"], dtype=np.float32)
        assert actual.shape == (15, 6) and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
        assert RobotExecutionFailures.CLASS_NAMES[config][row["label"]] == block[0]
        assert row["sample_id"] == index
        observed[row["label"]] += 1
    assert observed == dict(enumerate(counts))
    reloaded = RobotExecutionFailures(config_name=config, split="train", **official_cache)
    assert len(reloaded) == len(data)
    assert reloaded[-1] == data[-1]
