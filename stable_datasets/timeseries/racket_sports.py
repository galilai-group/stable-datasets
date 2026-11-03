import tempfile
import zipfile

import numpy as np

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder, load_from_tsfile_to_dataframe


_URL = "https://timeseriesclassification.com/aeon-toolkit/RacketSports.zip"
# Wayback Machine capture of the same file, used if the archive's server is down.
_FALLBACK = f"https://web.archive.org/web/20250224100042id_/{_URL}"
_CHECKSUM = "sha256:230c10f30f6800297313eb9aa77a9c2c8ba0714ce1364d3356cb3c342f69a90a"


class RacketSports(BaseDatasetBuilder):
    """RacketSports multivariate time series classification dataset from the UEA archive.

    303 recordings from a smart watch worn by university students playing badminton or squash: the x, y and z axes
    of its gyroscope and accelerometer (6 channels), sampled at 10 Hz for 3 seconds (30 steps). The task is to tell
    the sport and the stroke: badminton clear or smash, squash forehand or backhand boast. The archive's split has
    151 recordings in ``train`` and 152 in ``test``. Each ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=RacketSports",
        assets={
            "train": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
        },
        citation="""@article{bagnall2018uea,
    title={The UEA multivariate time series classification archive, 2018},
    author={Bagnall, Anthony and Dau, Hoang Anh and Lines, Jason and Flynn, Michael and Large, James and Bostrom, Aaron and Southam, Paul and Keogh, Eamonn},
    journal={arXiv preprint arXiv:1811.00075},
    year={2018}
}""",
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 30
    NUM_CHANNELS = 6
    _CLASSES = ["badminton_clear", "badminton_smash", "squash_backhandboast", "squash_forehandboast"]

    def _info(self):
        return DatasetInfo(
            description="Smart watch gyroscope and accelerometer recordings of badminton and squash strokes (UEA archive).",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=self._CLASSES),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        series, labels = _read_ts(data_path, f"RacketSports_{split.upper()}.ts")
        if series.shape[1:] != (self.SEQUENCE_LENGTH, self.NUM_CHANNELS):
            raise ValueError(f"Unexpected RacketSports series shape: {series.shape}")
        label_to_id = {name: idx for idx, name in enumerate(self._CLASSES)}
        for idx, (x, y) in enumerate(zip(series, labels)):
            yield idx, {"series": x, "label": label_to_id[y]}


def _read_ts(zip_path, member):
    """Return ``(series, labels)`` from a ``.ts`` file in the archive's zip, with series of shape (cases, steps, channels)."""
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp:
        x_df, labels = load_from_tsfile_to_dataframe(archive.extract(member, tmp))
    series = np.stack([np.stack(x_df[col].map(np.asarray)) for col in x_df.columns], axis=-1)
    return series.astype(np.float32), [str(label) for label in labels]
