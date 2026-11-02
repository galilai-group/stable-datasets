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


_URL = "https://timeseriesclassification.com/aeon-toolkit/Cricket.zip"
_CHECKSUM = "sha256:8748da8725cc9d5a1ea4499165426baedfab50fa651c6ecf9f8be2d2818df56b"


class Cricket(BaseDatasetBuilder):
    """Cricket multivariate time series classification dataset from the UEA archive.

    Cricket umpires signal events to the scorer with hand motions. Four umpires performed twelve signals, ten times
    each, with an accelerometer on each wrist (Ko et al., 2005): x, y and z acceleration of both wrists (6 channels)
    at 184 Hz, 1197 steps. The archive labels the signals ``1.0`` to ``12.0`` without naming them, so the classes are
    the numbers ``1`` to ``12``. ``train`` holds 108 recordings, ``test`` 72. Each ``series`` has shape
    ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=Cricket",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@inproceedings{ko2005online,
    title={Online Context Recognition in Multisensor Systems using Dynamic Time Warping},
    author={Ko, Ming Hsiao and West, G. and Venkatesh, S. and Kumar, M.},
    booktitle={2005 International Conference on Intelligent Sensors, Sensor Networks and Information Processing},
    pages={283--288},
    year={2005}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 1197
    NUM_CHANNELS = 6
    _CLASSES = [str(signal) for signal in range(1, 13)]
    # Labels as written in the archive's files, in the order of _CLASSES.
    _LABELS = [f"{n}.0" for n in range(1, 13)]

    def _info(self):
        return DatasetInfo(
            description="Wrist accelerometer recordings of twelve cricket umpire signals (UEA archive).",
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
        yield from iter_examples(data_path, f"Cricket_{split.upper()}.ts", shape, self._LABELS)
