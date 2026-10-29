
"""S&P 500 daily stock prices with chronological splits."""

import numpy as np
import pandas as pd

from stable_datasets.schema import (
    DatasetInfo,
    DatasetSource,
    Features,
    Value,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


class SP500(BaseDatasetBuilder):
    """Raw daily OHLCV observations for S&P 500 stocks."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.kaggle.com/datasets/camnugent/sandp500",
        assets={
            "data": (
                "https://raw.githubusercontent.com/"
                "plotly/datasets/master/all_stocks_5yr.csv"
            ),
        },
        citation="""@misc{nugent_sp500,
  author = {Nugent, Cam},
  title = {S&P 500 Stock Data},
  howpublished = {Kaggle},
  url = {https://www.kaggle.com/datasets/camnugent/sandp500}
}""",
        license="CC0-1.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Daily S&P 500 stock observations, including "
                "date, ticker, OHLC prices, and trading volume. "
                "Split chronologically into 70% training, "
                "15% validation, and 15% testing."
            ),
            features=Features({
                "date": Value("string"),
                "ticker": Value("string"),
                "open": Value("float32"),
                "high": Value("float32"),
                "low": Value("float32"),
                "close": Value("float32"),
                "volume": Value("int64"),
            }),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """Download the CSV using stable-datasets' cache."""

        data_path = download(
            self._source()["assets"]["data"],
            dest_folder=self._raw_download_dir,
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
        return [
            Split.TRAIN,
            Split.VALIDATION,
            Split.TEST,
        ]

    def _generate_examples(self, data_path, split):
        """Yield raw daily observations in chronological splits."""

        df = pd.read_csv(data_path, parse_dates=["date"])
        df = df.sort_values(["date", "Name"])

        required = [
            "date", "Name", "open", "high",
            "low", "close", "volume",
        ]
        df = df.dropna(subset=required)

        unique_dates = np.sort(df["date"].unique())

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

        subset = df[df["date"].isin(allowed_dates)]

        for example_id, row in enumerate(subset.itertuples(index=False)):
            yield example_id, {
                "date": row.date.strftime("%Y-%m-%d"),
                "ticker": str(row.Name),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }

            for window in windows:
                yield example_id, {"series": window}
                example_id += 1
