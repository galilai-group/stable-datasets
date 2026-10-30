"""Tests for the Forex dataset."""

import numpy as np
import pandas as pd

from stable_datasets.splits import Split
from stable_datasets.timeseries.Forex import Forex


def create_sample_data(tmp_path, n_days=100):
    """Create synthetic daily EURUSD observations."""

    dates = pd.date_range("2020-01-01", periods=n_days)

    df = pd.DataFrame(
        {
            "Time": dates,
            "Open": np.linspace(1.10, 1.20, n_days),
            "High": np.linspace(1.11, 1.21, n_days),
            "Low": np.linspace(1.09, 1.19, n_days),
            "Close": np.linspace(1.10, 1.20, n_days),
            "Volume": np.arange(1000, 1000 + n_days),
        }
    )

    path = tmp_path / "forex.parquet"
    df.to_parquet(path, index=False)

    return path


def test_forex_generate_examples(tmp_path):
    """Check that raw Forex observations are preserved."""

    path = create_sample_data(tmp_path)
    builder = object.__new__(Forex)

    examples = list(builder._generate_examples(path, Split.TRAIN))

    # 70% of 100 daily observations.
    assert len(examples) == 70

    for _, example in examples:
        assert set(example.keys()) == {
            "date",
            "pair",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        assert isinstance(example["date"], str)
        assert example["pair"] == "EURUSD"

        for column in ("open", "high", "low", "close"):
            assert np.isfinite(example[column])

        assert example["volume"] >= 0

    # Original values must not be normalized.
    first = examples[0][1]

    assert first["date"].startswith("2020-01-01")
    assert first["pair"] == "EURUSD"
    assert np.isclose(first["open"], 1.10)
    assert np.isclose(first["high"], 1.11)
    assert np.isclose(first["low"], 1.09)
    assert np.isclose(first["close"], 1.10)
    assert first["volume"] == 1000


def test_forex_chronological_splits(tmp_path):
    """Check split sizes and chronological ordering."""

    path = create_sample_data(tmp_path)
    builder = object.__new__(Forex)

    train = list(builder._generate_examples(path, Split.TRAIN))
    validation = list(builder._generate_examples(path, Split.VALIDATION))
    test = list(builder._generate_examples(path, Split.TEST))

    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15

    train_dates = {example["date"] for _, example in train}
    val_dates = {example["date"] for _, example in validation}
    test_dates = {example["date"] for _, example in test}

    # No date overlap.
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    # Chronological ordering.
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)


def test_forex_real_data():
    """Check that the real Forex dataset loads correctly."""

    dataset = Forex(split="train")

    assert len(dataset) > 0

    example = dataset[0]

    assert set(example.keys()) == {
        "date",
        "pair",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    assert example["pair"] == "EURUSD"
    assert example["open"] > 0
    assert example["high"] >= example["low"]
    assert example["close"] > 0
    assert example["volume"] >= 0
