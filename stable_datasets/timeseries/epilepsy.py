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


_URL = "https://timeseriesclassification.com/aeon-toolkit/Epilepsy.zip"
# Wayback Machine capture of the same file, used if the archive's server is down.
_FALLBACK = f"https://web.archive.org/web/20250225182932id_/{_URL}"
_CHECKSUM = "sha256:8e5ce818a4989e520aa620f3009621692f06c8e8e73d72ee98cc0dbdee7c10c9"


class Epilepsy(BaseDatasetBuilder):
    """Epilepsy multivariate time series classification dataset from the UEA archive.

    275 recordings of a tri-axial accelerometer on the dominant wrist of six healthy participants, sampled at 16 Hz
    and truncated by the archive to 206 steps. Each recording shows one of four activities: a mimicked epileptic
    seizure, walking, running or sawing. The archive's split puts three participants in ``train`` (137 recordings)
    and the other three in ``test`` (138). Each ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=Epilepsy",
        assets={
            "train": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
        },
        citation="""@article{villar2016generalized,
    title={Generalized Models for the Classification of Abnormal Movements in Daily Life and its Applicability to Epilepsy Convulsion Recognition},
    author={Villar, José R. and Vergara, Paula and Menéndez, Manuel and de la Cal, Enrique and González, Víctor M. and Sedano, Javier},
    journal={International Journal of Neural Systems},
    volume={26},
    number={6},
    pages={1650037},
    year={2016}
}
@article{bagnall2018uea,
    title={The UEA multivariate time series classification archive, 2018},
    author={Bagnall, Anthony and Dau, Hoang Anh and Lines, Jason and Flynn, Michael and Large, James and Bostrom, Aaron and Southam, Paul and Keogh, Eamonn},
    journal={arXiv preprint arXiv:1811.00075},
    year={2018}
}""",
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 206
    NUM_CHANNELS = 3
    _CLASSES = ["epilepsy", "running", "sawing", "walking"]

    def _info(self):
        return DatasetInfo(
            description="Tri-axial wrist accelerometer recordings of four activities, one a mimicked epileptic seizure (UEA archive).",
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
        series, labels = _read_ts(data_path, f"Epilepsy_{split.upper()}.ts")
        if series.shape[1:] != (self.SEQUENCE_LENGTH, self.NUM_CHANNELS):
            raise ValueError(f"Unexpected Epilepsy series shape: {series.shape}")
        label_to_id = {name: idx for idx, name in enumerate(self._CLASSES)}
        for idx, (x, y) in enumerate(zip(series, labels)):
            yield idx, {"series": x, "label": label_to_id[y]}


def _read_ts(zip_path, member):
    """Return ``(series, labels)`` from a ``.ts`` file in the archive's zip, with series of shape (cases, steps, channels)."""
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp:
        x_df, labels = load_from_tsfile_to_dataframe(archive.extract(member, tmp))
    series = np.stack([np.stack(x_df[col].map(np.asarray)) for col in x_df.columns], axis=-1)
    return series.astype(np.float32), [str(label) for label in labels]
