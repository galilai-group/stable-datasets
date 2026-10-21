import numpy as np

from stable_datasets.timeseries.SP500 import make_windows
from stable_datasets.timeseries.SP500 import SP500
from stable_datasets.splits import Split


def test_make_windows():
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=35)

    df = pd.DataFrame({
        "date": dates,
        "open": np.arange(100, 135),
        "high": np.arange(101, 136),
        "low": np.arange(99, 134),
        "close": np.arange(100, 135),
        "volume": np.full(35, 1000),
    })

    windows = make_windows(df, sequence_length=30)

    # 35 days produce 6 overlapping 30-day windows.
    assert len(windows) == 6

    # Each window has 30 days and 5 features.
    assert windows[0].shape == (30, 5)

    # The first day's closing price is the normalization reference.
    assert windows[0][0, 3] == 0.0

    # No missing or infinite values.
    assert np.isfinite(windows[0]).all()





def test_sp500_generate_examples(tmp_path):
    import pandas as pd

    # Create 120 days of synthetic stock data.
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=120),
        "open": range(100, 220),
        "high": range(101, 221),
        "low": range(99, 219),
        "close": range(100, 220),
        "volume": [1000] * 120,
        "Name": ["AAPL"] * 120,
    })

    csv_path = tmp_path / "stocks.csv"
    df.to_csv(csv_path, index=False)

    # Bypass the downloader to test example generation directly.
    builder = object.__new__(SP500)

    train_examples = list(
        builder._generate_examples(csv_path, Split.TRAIN)
    )

    assert len(train_examples) > 0

    for _, example in train_examples:
        assert example["series"].shape == (30, 5)
        assert np.isfinite(example["series"]).all()






def test_sp500_chronological_splits(tmp_path):
    import numpy as np
    import pandas as pd

    # Use enough dates for 30-day windows in every split.
    dates = pd.date_range("2020-01-01", periods=300)

    df = pd.DataFrame({
        "date": list(dates) * 2,
        "open": list(range(100, 400)) * 2,
        "high": list(range(101, 401)) * 2,
        "low": list(range(99, 399)) * 2,
        "close": list(range(100, 400)) * 2,
        "volume": [1000] * 600,
        "Name": ["AAPL"] * 300 + ["MSFT"] * 300,
    })

    csv_path = tmp_path / "stocks.csv"
    df.to_csv(csv_path, index=False)

    builder = object.__new__(SP500)

    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        examples = list(builder._generate_examples(csv_path, split))

        assert len(examples) > 0

        for _, example in examples:
            series = np.asarray(example["series"])
            assert series.shape == (30, 5)
            assert np.isfinite(series).all()
