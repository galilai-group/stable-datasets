"""Tests for the SP500 dataset."""

import numpy as np
import pandas as pd

from stable_datasets.splits import Split
from stable_datasets.timeseries.SP500 import SP500


def create_sample_data(tmp_path, n_days=100, n_stocks=2):
    """Create synthetic daily stock observations."""

    dates = pd.date_range("2020-01-01", periods=n_days)
    tickers = ["AAPL", "MSFT"][:n_stocks]

    rows = []

    for ticker in tickers:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "Name": ticker,
                    "open": 100.0 + i,
                    "high": 102.0 + i,
                    "low": 99.0 + i,
                    "close": 101.0 + i,
                    "volume": 1000 + i,
                }
            )

    df = pd.DataFrame(rows)

    csv_path = tmp_path / "stocks.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_sp500_generate_examples(tmp_path):
    """Check that examples preserve raw financial data."""

    csv_path = create_sample_data(tmp_path)

    builder = object.__new__(SP500)

    examples = list(builder._generate_examples(csv_path, Split.TRAIN))

    # 70 training dates, with 2 stocks per date.
    assert len(examples) == 140

    for _, example in examples:
        assert set(example.keys()) == {
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        assert isinstance(example["date"], str)
        assert isinstance(example["ticker"], str)

        assert example["ticker"] in {"AAPL", "MSFT"}

        for column in ("open", "high", "low", "close"):
            assert np.isfinite(example[column])

        assert example["volume"] >= 0

    # Verify the original values were not normalized.
    first = examples[0][1]

    assert first["date"] == "2020-01-01"
    assert first["ticker"] == "AAPL"
    assert first["open"] == 100.0
    assert first["high"] == 102.0
    assert first["low"] == 99.0
    assert first["close"] == 101.0
    assert first["volume"] == 1000


def test_sp500_chronological_splits(tmp_path):
    """Check chronological splitting without date overlap."""

    csv_path = create_sample_data(tmp_path)

    builder = object.__new__(SP500)

    train = list(builder._generate_examples(csv_path, Split.TRAIN))
    validation = list(builder._generate_examples(csv_path, Split.VALIDATION))
    test = list(builder._generate_examples(csv_path, Split.TEST))

    # 100 dates, 2 stocks per date.
    assert len(train) == 140
    assert len(validation) == 30
    assert len(test) == 30

    train_dates = {example["date"] for _, example in train}
    val_dates = {example["date"] for _, example in validation}
    test_dates = {example["date"] for _, example in test}

    # No overlap between splits.
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    # Strict chronological ordering.
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)


def test_sp500_real_data():
    """Check that the real SP500 dataset loads correctly."""

    dataset = SP500(split="train")

    assert len(dataset) > 0

    example = dataset[0]

    assert set(example.keys()) == {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    assert example["open"] > 0
    assert example["high"] >= example["low"]
    assert example["close"] > 0
    assert example["volume"] >= 0
