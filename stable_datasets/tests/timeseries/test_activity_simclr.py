"""Offline tests for the example's split and preprocessing contract."""

import numpy as np
import torch

from examples import activity_simclr as example


def test_mhealth_windows_preserve_timeline_and_channel_order():
    labels = np.zeros(640, dtype=np.int64)
    labels[:192] = 1
    labels[256:384] = 12
    labels[384:448] = 2
    labels[448:512] = 3
    labels[512:] = 2
    signals = np.arange(640 * 23).reshape(640, 23)
    windows = list(example.mhealth_windows({"series": signals, "label": labels}))
    assert [(label, start) for _, label, start in windows] == [(0, 0), (0, 64), (11, 256), (1, 512)]
    for series, _, start in windows:
        assert series.shape == (128, 21)
        np.testing.assert_array_equal(series, signals[start : start + 128, example.MOTION_CHANNELS])


def test_uci_har_split_keeps_official_test(monkeypatch):
    def row(subject):
        return {"series": np.full((128, 9), subject), "label": 0, "subject_id": subject}

    monkeypatch.setattr(example, "UCIHAR", lambda: {"train": [row(1), row(27), row(30)], "test": [row(2)]})
    arrays = example.prepare_data("uci_har")
    assert arrays["train"]["subject_id"].tolist() == [1]
    assert arrays["validation"]["subject_id"].tolist() == [27, 30]
    assert arrays["test"]["subject_id"].tolist() == [2]
    assert arrays["validation"]["source_index"].tolist() == [1, 2]


def test_mhealth_subjects_are_split_before_windowing(monkeypatch):
    records = [{"series": np.full((128, 23), s), "label": np.ones(128), "subject_id": s} for s in range(1, 11)]
    monkeypatch.setattr(example, "MHEALTH", lambda split: records)
    arrays = example.prepare_data("mhealth")
    assert [arrays[s]["subject_id"].tolist() for s in example.SPLITS] == [list(range(1, 7)), [7, 8], [9, 10]]
    assert all(np.all(rows["label"] == 0) for rows in arrays.values())


def test_normalization_fits_training_only():
    arrays = {s: {"series": np.zeros((2, 128, 2), dtype=np.float32)} for s in example.SPLITS}
    arrays["train"]["series"][1, :, 0] = 2
    arrays["validation"]["series"][:] = 100
    arrays["test"]["series"][:] = -100
    tensors, mean, std = example.standardize(arrays)
    np.testing.assert_array_equal(mean, [1, 0])
    np.testing.assert_array_equal(std, [1, 1])
    assert tensors["train"].shape == (2, 2, 128)
    assert tensors["validation"][0, 0, 0] == 99
    assert tensors["test"][0, 0, 0] == -101


def test_views_are_independent_without_labels_or_inplace_changes():
    torch.manual_seed(0)
    signals = torch.ones(2, 9, 128)
    batch = example.TwoViews(signals)[0]
    assert set(batch) == {"views"}
    a, b = [view["image"] for view in batch["views"]]
    assert a.shape == b.shape == (9, 128)
    assert not torch.equal(a, b)
    assert torch.all(signals == 1)


def test_linear_probe_ignores_test_labels_for_selection_and_fitting():
    features = {
        "train": np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        "validation": np.array([[0.2, 0], [0.8, 1]], dtype=float),
        "test": np.array([[-100, 0], [100, 0]], dtype=float),
    }
    arrays = {
        "train": {"label": np.array([0, 0, 1, 1])},
        "validation": {"label": np.array([0, 1])},
        "test": {"label": np.array([0, 1])},
    }
    first, prediction, model = example.linear_probe(features, arrays)
    arrays["test"]["label"] = np.array([1, 0])
    second, changed_prediction, _ = example.linear_probe(features, arrays)
    assert first["C"] == second["C"] == 0.01
    assert first["candidates"] == second["candidates"]
    np.testing.assert_array_equal(prediction, changed_prediction)
    np.testing.assert_array_equal(model.steps[0][1].mean_, [0.5, 0.5])
    assert first["test"]["accuracy"] == 1.0
    assert second["test"]["accuracy"] == 0.0
