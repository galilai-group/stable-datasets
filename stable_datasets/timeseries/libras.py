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


_URL = "https://timeseriesclassification.com/aeon-toolkit/Libras.zip"
# Wayback Machine capture of the same file, used if the archive's server is down.
_FALLBACK = f"https://web.archive.org/web/20251212101231id_/{_URL}"
_CHECKSUM = "sha256:8c7f9c67e9c6c66ca659f65e077e3d1f212ab1f237dd212d488df13f25fa3688"


class Libras(BaseDatasetBuilder):
    """Libras multivariate time series classification dataset from the UEA archive.

    Hand movements of the Brazilian Sign Language (LIBRAS), from the LIBRAS Movement Database (Dias et al., 2009):
    curves traced by the hand in videos of four people, time-normalised to 45 frames and given as two normalised
    coordinates per frame (2 channels). The fifteen classes are the movement types the database documents, such as
    ``curved_swing`` or ``circle``. ``train`` and ``test`` hold 180 recordings each. The original database already
    contains repeated curves, so 19 test series also occur in ``train`` with the same label. Each ``series`` has shape
    ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=Libras",
        assets={
            "train": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
        },
        citation="""@inproceedings{dias2009hand,
    title={Hand Movement Recognition for Brazilian Sign Language: A Study Using Distance-Based Neural Networks},
    author={Dias, Daniel B. and Madeo, Renata C. B. and Rocha, Thiago and Bíscaro, Helton H. and Peres, Sarajane M.},
    booktitle={2009 International Joint Conference on Neural Networks},
    pages={697--704},
    year={2009}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 45
    NUM_CHANNELS = 2
    _CLASSES = [
        "curved_swing",
        "horizontal_swing",
        "vertical_swing",
        "anti_clockwise_arc",
        "clockwise_arc",
        "circle",
        "horizontal_straight_line",
        "vertical_straight_line",
        "horizontal_zigzag",
        "vertical_zigzag",
        "horizontal_wavy",
        "vertical_wavy",
        "face_up_curve",
        "face_down_curve",
        "tremble",
    ]
    # Labels as written in the archive's files (1-15, documented in its LIBRAS description), in the order of _CLASSES.
    _LABELS = [str(movement) for movement in range(1, 16)]

    def _info(self):
        return DatasetInfo(
            description="Hand movement curves of fifteen Brazilian Sign Language (LIBRAS) movement types (UEA archive).",
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
        yield from iter_examples(data_path, f"Libras_{split.upper()}.ts", shape, self._LABELS)
