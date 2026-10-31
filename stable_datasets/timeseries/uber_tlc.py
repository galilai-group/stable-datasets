import zipfile

import numpy as np
import pandas as pd

from stable_datasets.schema import (
    BuilderConfig,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder


_URL = (
    "https://github.com/fivethirtyeight/uber-tlc-foil-response/raw/master/uber-trip-data/"
    "uber-raw-data-janjune-15.csv.zip"
)
_CHECKSUM = "sha256:b85c7373ce0b0186eb204e14600033033abf0b2ce47cc97b4ca59e1ec6ec8bec"
_START, _END = "2015-01-01", "2015-07-01"  # January to June 2015, end exclusive


class UberTLC(BaseDatasetBuilder):
    """Uber pickups in New York City per taxi zone, January-June 2015, for forecasting.

    Built from the raw trip records FiveThirtyEight obtained from the NYC Taxi & Limousine Commission: every
    pickup is counted in its taxi zone (``locationID``) and hour or day, with zero for periods without pickups.
    One example per zone (262 zones), 4,344 hourly or 181 daily counts. Splits follow the usual forecasting
    protocol (Chronos, GluonTS): ``test`` holds each full series and the forecast target is its last
    ``prediction_length`` points; ``train`` holds the same series with those points removed.

    Configs: ``hourly`` and ``daily``; ``UberTLC.SETTINGS[config]`` gives (frequency, prediction length, seasonality).
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(name="hourly", version=Version("1.0.0"), description="Pickups per zone per hour."),
        BuilderConfig(name="daily", version=Version("1.0.0"), description="Pickups per zone per day."),
    ]
    DEFAULT_CONFIG_NAME = "hourly"
    SOURCE = DatasetSource(
        homepage="https://github.com/fivethirtyeight/uber-tlc-foil-response",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="uber-raw-data-janjune-15.csv.zip"),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="uber-raw-data-janjune-15.csv.zip"),
        },
        citation="""@misc{fivethirtyeight2015uber,
    title={Uber TLC FOIL Response},
    author={FiveThirtyEight},
    howpublished={\\url{https://github.com/fivethirtyeight/uber-tlc-foil-response}},
    year={2015}
}""",
        license="See dataset homepage (NYC Taxi & Limousine Commission records, released under FOIL).",
    )

    #: config name -> (frequency, prediction length, seasonality)
    SETTINGS = {"hourly": ("h", 24, 24), "daily": ("D", 7, 1)}

    def _info(self):
        return DatasetInfo(
            description=f"Uber pickups per NYC taxi zone, January-June 2015 ({self.config.name}).",
            features=Features(
                {
                    "series": Sequence(Value("float32")),
                    "item_id": Value("string"),
                    "start": Value("string"),
                }
            ),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        freq, prediction_length, _ = self.SETTINGS[self.config.name]
        with zipfile.ZipFile(data_path) as archive, archive.open("uber-raw-data-janjune-15.csv") as handle:
            trips = pd.read_csv(handle, usecols=["Pickup_date", "locationID"], parse_dates=["Pickup_date"])
        periods = pd.date_range(_START, _END, freq=freq, inclusive="left")
        counts = (
            trips.groupby(["locationID", trips["Pickup_date"].dt.floor(freq)])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=periods, fill_value=0)
        )
        end = -prediction_length if split == "train" else None
        for idx, (zone, row) in enumerate(counts.iterrows()):
            yield (
                idx,
                {
                    "series": row.to_numpy(np.float32)[:end],
                    "item_id": f"location_{zone:03d}",
                    "start": periods[0].isoformat(),
                },
            )
