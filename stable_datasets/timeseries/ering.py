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


_URL = "https://timeseriesclassification.com/aeon-toolkit/ERing.zip"
_CHECKSUM = "sha256:1d491aa64ff2b80ab61612b867013e2203d00c8a4c65c043687428ee8d6ce213"


class ERing(BaseDatasetBuilder):
    """ERing multivariate time series classification dataset from the UEA archive.

    Recordings of eRing, a prototype finger ring that detects hand and finger gestures with electric field sensing
    (Wilhelm et al., 2015). Each recording holds the measurements of 4 electrodes, which vary with the distance to the
    hand, over 65 steps. The six classes are postures of the thumb, index and middle finger; the archive labels them
    ``1`` to ``6`` without naming them. ``train`` holds 30 recordings, ``test`` 270. Each ``series`` has shape
    ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=ERing",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@inproceedings{wilhelm2015ering,
    title={eRing: Multiple Finger Gesture Recognition with one Ring Using an Electric Field},
    author={Wilhelm, Mathias and Krakowczyk, Daniel and Trollmann, Frank and Albayrak, Sahin},
    booktitle={Proceedings of the 2nd International Workshop on Sensor-based Activity Recognition and Interaction},
    pages={1--6},
    year={2015}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 65
    NUM_CHANNELS = 4
    _CLASSES = [str(posture) for posture in range(1, 7)]

    def _info(self):
        return DatasetInfo(
            description="Electric field sensing recordings of six finger postures from the eRing prototype (UEA archive).",
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
        yield from iter_examples(data_path, f"ERing_{split.upper()}.ts", shape, self._CLASSES)
