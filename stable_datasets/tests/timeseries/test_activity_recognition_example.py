"""Check axis semantics and subject isolation in the evaluation example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[3] / "examples" / "activity_recognition.py"
spec = importlib.util.spec_from_file_location("activity_recognition_example", path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_features_reduce_time_not_channels():
    x = np.array([[1, 10], [3, 30]], dtype=np.float32)
    np.testing.assert_allclose(example.window_features(x), [2, 20, 1, 10, 1, 10, 3, 30])


def test_evaluation_rejects_subject_leakage():
    class LeakyBuilder:
        SEQUENCE_LENGTH = 2
        CHANNEL_NAMES = ("x",)

        def __new__(cls, **kwargs):
            row = {"series": [[1], [2]], "label": 0, "subject_id": 1}
            return {"train": [row], "test": [row]}

    with pytest.raises(ValueError, match="Subject leakage"):
        example.evaluate(LeakyBuilder, None, None, 42)


def test_shape_validation_precedes_training():
    with pytest.raises(ValueError, match="Unexpected window shape"):
        example.split_arrays([{"series": [[1, 2]]}], (2, 1))


@pytest.mark.parametrize("builder", [example.DailyAndSportsActivities])
def test_both_builders_support_evaluation_metadata(builder):
    assert len(builder.ACTIVITY_NAMES) >= 2
    assert builder.SEQUENCE_LENGTH > 0
    assert len(builder.CHANNEL_NAMES) > 0


def test_full_evaluation_pipeline():
    class TinyBuilder:
        SEQUENCE_LENGTH = 2
        CHANNEL_NAMES = ("x",)
        ACTIVITY_NAMES = ("low", "high")

        def __new__(cls, **kwargs):
            return {
                split: [
                    {"series": [[value], [value]], "label": label, "subject_id": subject}
                    for label, value in [(0, 0), (1, 100)]
                    for _ in range(5)
                ]
                for split, subject in [("train", 1), ("test", 2)]
            }

    report = example.evaluate(TinyBuilder, None, None, 42)
    assert report["results"]["random_forest"]["accuracy"] == 1.0
    assert report["results"]["majority"]["accuracy"] == 0.5
