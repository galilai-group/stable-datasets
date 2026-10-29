
"""Daily EUR/USD forex dataset from Hugging Face."""

import pandas as pd

from stable_datasets.schema import (
    DatasetInfo,
    DatasetSource,
    Features,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder, download
from stable_datasets.splits import Split, SplitGenerator


class Forex(BaseDatasetBuilder):
    """Raw daily EUR/USD exchange rates."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://huggingface.co/datasets/Yllvar/fx-historical-data",
        assets={
            "data": (
                "https://huggingface.co/datasets/"
                "Yllvar/fx-historical-data/resolve/main/"
                "EURUSD/EURUSD_D1.parquet"
            ),
        },
        citation="""@misc{yllvar_forex,
  author = {Yllvar},
  title = {Forex Historical Data},
  howpublished = {Hugging Face},
  url = {https://huggingface.co/datasets/Yllvar/fx-historical-data}
}""",
        license="CC-BY-4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Daily EUR/USD forex observations, including "
                "date, currency pair, OHLC exchange rates, and "
                "tick volume. Observations are split chronologically "
                "into 70% training, 15% validation, and 15% testing."
            ),
            features=Features({
                "date": Value("string"),
                "pair": Value("string"),
                "open": Value("float64"),
                "high": Value("float64"),
                "low": Value("float64"),
                "close": Value("float64"),
                "volume": Value("int64"),
            }),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        data_path = download(
            self._source()["assets"]["data"],
            dest_folder=self._raw_download_dir,
        )

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
        df = pd.read_parquet(data_path)

        df.columns = df.columns.str.lower()

        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})

        df["date"] = pd.to_datetime(df["date"], utc=True)

        required = [
            "date", "open", "high",
            "low", "close", "volume",
        ]
        df = df.dropna(subset=required)
        df = df.sort_values("date").drop_duplicates("date")

        train_end = int(len(df) * 0.70)
        val_end = int(len(df) * 0.85)

        if split == Split.TRAIN:
            subset = df.iloc[:train_end]
        elif split == Split.VALIDATION:
            subset = df.iloc[train_end:val_end]
        elif split == Split.TEST:
            subset = df.iloc[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        for index, row in enumerate(subset.itertuples(index=False)):
            yield index, {
                "date": row.date.isoformat(),
                "pair": "EURUSD",
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }

