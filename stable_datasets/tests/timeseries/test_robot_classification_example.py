"""Check evaluation grouping and out-of-fold coverage independently of downloads."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[3] / "examples" / "robot_failure_classification.py"
spec = importlib.util.spec_from_file_location("robot_example", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_features_and_duplicates_ignore_labels():
    a = np.tile(np.arange(6), (15, 1)).astype(np.float32)
    rows = [{"series": a, "label": label} for label in (0, 1)]
    x, y, groups = example.arrays_and_groups(rows)
    assert groups[0] == groups[1]
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_array_equal(x[0], np.concatenate([a[0], np.zeros(6), a[0], a[0]]))


def test_group_folds_keep_duplicates_together_and_cover_all_rows():
    y = np.repeat([0, 1], 12)
    groups = np.repeat(np.arange(12), 2)
    partition = example.folds(y, groups, 42)
    assert len(partition) == 3
    tested = []
    for train, test in partition:
        assert not set(groups[train]) & set(groups[test])
        tested.extend(test.tolist())
    assert sorted(tested) == list(range(24))


def test_full_pipeline_reports_out_of_fold_predictions(monkeypatch):
    rows = [
        {"series": np.full((15, 6), label * 100 + i, dtype=np.float32), "label": label}
        for label in range(4)
        for i in range(9)
    ]

    class FakeBuilder:
        CLASS_NAMES = {"lp1": ("a", "b", "c", "d")}

        def __new__(cls, **kwargs):
            return rows

    monkeypatch.setattr(example, "RobotExecutionFailures", FakeBuilder)
    report = example.evaluate("lp1", None, None, 42)
    assert report["count"] == 36
    assert report["results"]["random_forest"]["accuracy"] == 1
    assert len(report["results"]["random_forest"]["out_of_fold_predictions"]) == 36


def test_rejects_invalid_windows():
    with pytest.raises(ValueError, match="15-by-6"):
        example.arrays_and_groups([{"series": np.zeros((6, 15)), "label": 0}])
