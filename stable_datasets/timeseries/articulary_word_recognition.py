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


_URL = "https://timeseriesclassification.com/aeon-toolkit/ArticularyWordRecognition.zip"
_CHECKSUM = "sha256:b5ac536689b7d5a6d4425f01a55897d5aba11fd0c2613bdacffb74eceb6518a2"


class ArticularyWordRecognition(BaseDatasetBuilder):
    """ArticularyWordRecognition multivariate time series classification dataset from the UEA archive.

    Tongue and lip movements of native English speakers saying 25 words, recorded with an electromagnetic
    articulograph (Wang et al., 2013) at 200 Hz: 9 of the 36 position dimensions of the original sensors (9 channels),
    144 steps. The archive labels the words ``1.0`` to ``25.0`` without naming them, so the classes are the numbers
    ``1`` to ``25``. ``train`` holds 275 recordings, ``test`` 300. Each ``series`` has shape
    ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@inproceedings{wang2013word,
    title={Word Recognition from Continuous Articulatory Movement Time-series Data using Symbolic Representations},
    author={Wang, Jun and Balasubramanian, Arvind and Mojica de la Vega, Luis and Green, Jordan R. and Samal, Ashok and Prabhakaran, Balakrishnan},
    booktitle={Proceedings of the Fourth Workshop on Speech and Language Processing for Assistive Technologies},
    pages={119--127},
    year={2013}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 144
    NUM_CHANNELS = 9
    _CLASSES = [str(word) for word in range(1, 26)]
    # Labels as written in the archive's files, in the order of _CLASSES.
    _LABELS = [f"{n}.0" for n in range(1, 26)]

    def _info(self):
        return DatasetInfo(
            description="Electromagnetic articulograph recordings of tongue and lip movements while speaking 25 words (UEA archive).",
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
        yield from iter_examples(data_path, f"ArticularyWordRecognition_{split.upper()}.ts", shape, self._LABELS)
