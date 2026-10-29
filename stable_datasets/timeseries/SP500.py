"""S&P 500 daily stock prices with chronological splits."""

import zipfile

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
            "data": ("https://www.kaggle.com/api/v1/datasets/download/camnugent/sandp500"),
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
                "Trading dates are split chronologically into "
                "70% training, 15% validation, and 15% testing."
            ),
            features=Features(
                {
                    "date": Value("string"),
                    "ticker": Value("string"),
                    "open": Value("float32"),
                    "high": Value("float32"),
                    "low": Value("float32"),
                    "close": Value("float32"),
                    "volume": Value("int64"),
                }
            ),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """Download the Kaggle archive and prepare chronological splits."""

        archive_path = download(
            self._source()["assets"]["data"],
            dest_folder=self._raw_download_dir,
            filename="sandp500.zip",
            checksum="sha256:4f775195e67179c9c19b63f084e5f2e049fae734023f9d10e233388d6423752a",
        )

        data_path = self._raw_download_dir / "all_stocks_5yr.csv"

        if not data_path.exists():
            with zipfile.ZipFile(archive_path) as archive:
                with archive.open("all_stocks_5yr.csv") as source:
                    temporary_path = data_path.with_suffix(".csv.tmp")
                    with open(temporary_path, "wb") as destination:
                        import shutil

                        shutil.copyfileobj(source, destination)

                    temporary_path.replace(data_path)

        return [
            SplitGenerator(
                name=split,
                gen_kwargs={"data_path": data_path, "split": split},
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

        required = [
            "date",
            "Name",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        df = df.dropna(subset=required)
        df = df.sort_values(["date", "Name"])

        # Use unique trading dates to avoid overlap between splits.
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
            yield (
                example_id,
                {
                    "date": row.date.strftime("%Y-%m-%d"),
                    "ticker": str(row.Name),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume),
                },
            )
