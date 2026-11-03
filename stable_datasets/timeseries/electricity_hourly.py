import zipfile

import numpy as np

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Sequence, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1"
_CHECKSUM = "sha256:eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c"


class ElectricityHourly(BaseDatasetBuilder):
    """Hourly electricity consumption of 321 clients (Monash Forecasting Archive), for forecasting.

    One example per client: the hourly load in kW from 2012 to 2014, 26,304 points. Splits follow the
    usual forecasting protocol (Chronos, GluonTS): ``test`` holds each full series and the forecast target is
    its last ``PREDICTION_LENGTH`` points; ``train`` holds the same series with those points removed.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/4656140",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="electricity_hourly_dataset.zip"),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM, filename="electricity_hourly_dataset.zip"),
        },
        citation="""@inproceedings{godahewa2021monash,
    title={Monash Time Series Forecasting Archive},
    author={Godahewa, Rakshitha and Bergmeir, Christoph and Webb, Geoffrey I. and Hyndman, Rob J. and Montero-Manso, Pablo},
    booktitle={Neural Information Processing Systems Track on Datasets and Benchmarks},
    year={2021}
}""",
        license="CC BY 4.0",
    )

    FREQ = "h"
    PREDICTION_LENGTH = 24
    SEASONALITY = 24

    def _info(self):
        return DatasetInfo(
            description="Hourly electricity consumption (kW) of 321 clients, 2012-2014 (Monash Forecasting Archive).",
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
        with zipfile.ZipFile(data_path) as archive:
            text = archive.read("electricity_hourly_dataset.tsf").decode("utf-8")
        for idx, (name, start, values) in enumerate(_read_tsf(text)):
            if split == "train":
                values = values[: -self.PREDICTION_LENGTH]
            yield idx, {"series": values, "item_id": name, "start": start}


def _read_tsf(text: str):
    """Yield ``(series_name, start, values)`` from a Monash ``.tsf`` file with one start timestamp per series.

    Monash writes timestamps as ``YYYY-MM-DD HH-MM-SS``; they are returned in ISO form.
    """
    _, data = text.split("@data", 1)
    for line in data.strip().splitlines():
        name, start, values = line.split(":", 2)
        day, time = start.split(" ")
        yield name, f"{day}T{time.replace('-', ':')}", np.array(values.split(","), dtype=np.float32)
