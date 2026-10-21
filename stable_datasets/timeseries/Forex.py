
"""Daily EUR/USD forex time-series dataset from Hugging Face."""

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

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

REPO_ID = "Yllvar/fx-historical-data"
FILE_NAME = "EURUSD/EURUSD_D1.parquet"


def make_forex_windows(data, sequence_length=SEQUENCE_LENGTH):
    """Create normalized OHLC windows from chronological forex data."""
    data = data.sort_values("date")
    prices = data[PRICE_COLUMNS].to_numpy(dtype=np.float32)

    windows = []

    for start in range(len(prices) - sequence_length + 1):
        window = prices[start:start + sequence_length]

        initial_close = window[0, 3]

        if not np.isfinite(initial_close) or initial_close <= 0:
            continue

        normalized = window / initial_close - 1.0

        if np.isfinite(normalized).all():
            windows.append(normalized.astype(np.float32))

    return windows


class Forex(BaseDatasetBuilder):
    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://huggingface.co/datasets/Yllvar/fx-historical-data",
        assets={},
        citation="Yllvar, fx-historical-data, Hugging Face. CC BY 4.0.",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Daily EUR/USD forex OHLC rates represented as "
                "30-observation normalized time-series windows."
            ),
            features=Features({
                "series": Sequence(Sequence(Value("float32")))
            }),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        data_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILE_NAME,
            repo_type="dataset",
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
        df = pd.read_parquet(data_path)

        # The source uses capitalized column names.
        df.columns = df.columns.str.lower()

        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.dropna(subset=["date"] + PRICE_COLUMNS)
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

        for index, window in enumerate(make_forex_windows(subset)):
            yield index, {"series": window}
