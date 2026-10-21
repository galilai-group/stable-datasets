
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from stable_datasets.schema import (
    DatasetInfo,
    DatasetSource,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder
from stable_datasets.splits import Split, SplitGenerator


SEQUENCE_LENGTH = 30
PRICE_COLUMNS = ["open", "high", "low", "close"]


def make_windows(stock_data, sequence_length=SEQUENCE_LENGTH):
    """Convert one stock's daily observations into time-series windows."""

    stock_data = stock_data.sort_values("date")

    prices = stock_data[PRICE_COLUMNS].to_numpy(dtype=np.float32)
    volumes = stock_data["volume"].to_numpy(dtype=np.float32)

    windows = []

    for start in range(len(stock_data) - sequence_length + 1):
        end = start + sequence_length

        price_window = prices[start:end]
        volume_window = volumes[start:end]

        # Express prices relative to the first day's closing price.
        initial_price = price_window[0, 3]

        if initial_price <= 0 or not np.isfinite(initial_price):
            continue

        normalized_prices = price_window / initial_price - 1.0

        # Compress the scale of trading volume.
        log_volume = np.log1p(volume_window)

        window = np.column_stack(
            [normalized_prices, log_volume]
        ).astype(np.float32)

        if np.isfinite(window).all():
            windows.append(window)

    return windows


class SP500(BaseDatasetBuilder):
    """S&P 500 daily stock prices as multivariate time series."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.kaggle.com/datasets/camnugent/sandp500",
        assets={},
        citation="S&P 500 Stock Data, Kaggle (camnugent).",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "S&P 500 historical stock prices represented "
                "as 30-day multivariate time-series windows."
            ),
            features=Features({
                "series": Sequence(Sequence(Value("float32"))),
            }),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """Download the S&P 500 dataset and define chronological splits."""

        dataset_dir = Path(
            kagglehub.dataset_download("camnugent/sandp500")
        )

        data_path = dataset_dir / "all_stocks_5yr.csv"

        if not data_path.is_file():
            raise FileNotFoundError(
                f"Could not find all_stocks_5yr.csv in {dataset_dir}"
            )

        return [
            SplitGenerator(
                name=split,
                gen_kwargs={
                    "data_path": data_path,
                    "split": split,
                },
            )
            for split in (
                Split.TRAIN,
                Split.VALIDATION,
                Split.TEST,
            )
        ]

    def _candidate_splits(self):
        """Return the available dataset splits."""

        return [
            Split.TRAIN,
            Split.VALIDATION,
            Split.TEST,
        ]

    
    def _generate_examples(self, data_path, split):
        """Generate 30-day sequences using common chronological splits."""

        df = pd.read_csv(data_path, parse_dates=["date"])
        df = df.sort_values(["Name", "date"])

        # Establish shared date boundaries across all stocks.
        unique_dates = np.sort(df["date"].dropna().unique())

        train_end = int(len(unique_dates) * 0.70)
        val_end = int(len(unique_dates) * 0.85)

        if split == Split.TRAIN:
            allowed_dates = unique_dates[:train_end]
        elif split == Split.VALIDATION:
            allowed_dates = unique_dates[train_end:val_end]
        elif split == Split.TEST:
            allowed_dates = unique_dates[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        if len(allowed_dates) == 0:
            return

        start_date = allowed_dates[0]
        end_date = allowed_dates[-1]

        subset = df[
            (df["date"] >= start_date)
            & (df["date"] <= end_date)
        ]

        example_id = 0

        for _, stock_data in subset.groupby("Name"):
            windows = make_windows(stock_data)

            for window in windows:
                yield example_id, {"series": window}
                example_id += 1
