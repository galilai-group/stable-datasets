
import numpy as np
import pandas as pd

from stable_datasets.timeseries.Forex import (
    Forex,
    make_forex_windows,
)
from stable_datasets.splits import Split


def test_forex_windows():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=40),
        "open": np.linspace(1.10, 1.20, 40),
        "high": np.linspace(1.11, 1.21, 40),
        "low": np.linspace(1.09, 1.19, 40),
        "close": np.linspace(1.10, 1.20, 40),
    })

    windows = make_forex_windows(df)

    assert len(windows) == 11
    assert windows[0].shape == (30, 4)
    assert np.isfinite(windows[0]).all()
    assert np.isclose(windows[0][0, 3], 0.0)


def test_forex_splits(tmp_path):
    dates = pd.date_range("2020-01-01", periods=300)

    df = pd.DataFrame({
        "Time": dates,
        "Open": np.linspace(1.10, 1.30, 300),
        "High": np.linspace(1.11, 1.31, 300),
        "Low": np.linspace(1.09, 1.29, 300),
        "Close": np.linspace(1.10, 1.30, 300),
        "Volume": np.ones(300),
    })

    path = tmp_path / "forex.parquet"
    df.to_parquet(path)

    builder = object.__new__(Forex)

    expected = {
        Split.TRAIN: 181,
        Split.VALIDATION: 16,
        Split.TEST: 16,
    }

    for split, count in expected.items():
        examples = list(builder._generate_examples(path, split))

        assert len(examples) == count
        assert all(
            example["series"].shape == (30, 4)
            for _, example in examples
        )
