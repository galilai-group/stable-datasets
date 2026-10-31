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


_URL = "https://timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip"
_CHECKSUM = "sha256:c6a771794bb14a815e204307b7d7e85d7a8291ddb16626cb9460aa5a75b6e042"


class UWaveGestureLibrary(BaseDatasetBuilder):
    """UWaveGestureLibrary multivariate time series classification dataset from the UEA archive.

    Accelerometer recordings of eight simple gestures from the uWave gesture library (Liu et al., 2009): the x, y and
    z acceleration of each motion (3 channels), 315 steps. The classes are the gesture numbers ``1`` to ``8`` of the
    uWave paper. ``train`` holds 120 recordings, ``test`` 320. Each ``series`` has shape
    ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@inproceedings{liu2009uwave,
    title={uWave: Accelerometer-based Personalized Gesture Recognition and its Applications},
    author={Liu, Jiayang and Wang, Zhen and Zhong, Lin and Wickramasuriya, Jehan and Vasudevan, Venu},
    booktitle={2009 IEEE International Conference on Pervasive Computing and Communications},
    pages={1--9},
    year={2009}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 315
    NUM_CHANNELS = 3
    _CLASSES = [str(gesture) for gesture in range(1, 9)]
    # Labels as written in the archive's files, in the order of _CLASSES.
    _LABELS = [f"{gesture}.0" for gesture in range(1, 9)]

    def _info(self):
        return DatasetInfo(
            description="Accelerometer recordings of eight gestures from the uWave gesture library (UEA archive).",
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
        yield from iter_examples(data_path, f"UWaveGestureLibrary_{split.upper()}.ts", shape, self._LABELS)
