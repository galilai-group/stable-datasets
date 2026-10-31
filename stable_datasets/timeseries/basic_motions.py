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
from stable_datasets.utils import BaseDatasetBuilder

from ._uea import UEA_ARCHIVE_CITATION, iter_examples


_URL = "https://timeseriesclassification.com/aeon-toolkit/BasicMotions.zip"
_CHECKSUM = "sha256:f427609dfe343b1da0dc54ee361f61579e7f18c2a007121740a9f0d90286fac3"


class BasicMotions(BaseDatasetBuilder):
    """BasicMotions multivariate time series classification dataset from the UEA archive.

    Smart watch recordings from a student project in which four students performed four activities: the x, y and z
    axes of the watch's accelerometer and gyroscope (6 channels), sampled at 10 Hz for 10 seconds (100 steps). The
    classes are ``badminton``, ``running``, ``standing`` and ``walking``. ``train`` and ``test`` hold 40 recordings
    each. Each ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=BasicMotions",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation=UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 100
    NUM_CHANNELS = 6
    _CLASSES = ["badminton", "running", "standing", "walking"]

    def _info(self):
        return DatasetInfo(
            description="Smart watch accelerometer and gyroscope recordings of four everyday activities (UEA archive).",
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
        shape = (self.SEQUENCE_LENGTH, self.NUM_CHANNELS)
        yield from iter_examples(data_path, f"BasicMotions_{split.upper()}.ts", shape, self._CLASSES)
