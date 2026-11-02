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


_URL = "https://timeseriesclassification.com/aeon-toolkit/Handwriting.zip"
_CHECKSUM = "sha256:e8a19ce94c82ee76205d441c77928a08ec34f7dcbf3edc11f569055c6ed51a8b"


class Handwriting(BaseDatasetBuilder):
    """Handwriting multivariate time series classification dataset from the UEA archive.

    Motion of a smart watch while its wearer writes the 26 letters of the alphabet (Shokoohi-Yekta et al., 2017):
    x, y and z acceleration (3 channels), 152 steps. The archive labels the letters ``1.0`` to ``26.0`` without
    documenting which number is which letter, so the classes are the numbers ``1`` to ``26``. ``train`` holds 150
    recordings (2 to 11 per class), ``test`` 850. Each ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=Handwriting",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@article{shokoohiyekta2017generalizing,
    title={Generalizing DTW to the multi-dimensional case requires an adaptive approach},
    author={Shokoohi-Yekta, Mohammad and Hu, Bing and Jin, Hongxia and Wang, Jun and Keogh, Eamonn},
    journal={Data Mining and Knowledge Discovery},
    volume={31},
    number={1},
    pages={1--31},
    year={2017}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 152
    NUM_CHANNELS = 3
    _CLASSES = [str(letter) for letter in range(1, 27)]
    # Labels as written in the archive's files, in the order of _CLASSES.
    _LABELS = [f"{n}.0" for n in range(1, 27)]

    def _info(self):
        return DatasetInfo(
            description="Smart watch accelerometer recordings of writing the 26 letters of the alphabet (UEA archive).",
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
        yield from iter_examples(data_path, f"Handwriting_{split.upper()}.ts", shape, self._LABELS)
