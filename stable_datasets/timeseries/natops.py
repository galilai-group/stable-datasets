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


_URL = "https://timeseriesclassification.com/aeon-toolkit/NATOPS.zip"
# Wayback Machine capture of the same file, used if the archive's server is down.
_FALLBACK = f"https://web.archive.org/web/20250225053625id_/{_URL}"
_CHECKSUM = "sha256:38e91561d70426b508abf6db253070b1bed00a1c303d70844997ef4d4ab3b7ff"


class NATOPS(BaseDatasetBuilder):
    """NATOPS multivariate time series classification dataset from the UEA archive.

    Six aircraft handling signals from the Naval Air Training and Operating Procedures Standardization (NATOPS),
    recorded by sensors on both hand tips, elbows, wrists and thumbs: x, y and z coordinates of the eight locations
    (24 channels), 51 steps. The data comes from the AALTD 2016 challenge (Ghouaiel et al., 2017). ``train`` and
    ``test`` hold 180 recordings each. Each ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=NATOPS",
        assets={
            "train": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, fallbacks=[_FALLBACK], checksum=_CHECKSUM),
        },
        citation="""@article{ghouaiel2017continuous,
    title={Continuous pattern detection and recognition in stream - a benchmark for online gesture recognition},
    author={Ghouaiel, Nehla and Marteau, Pierre-François and Dupont, Marc},
    journal={International Journal of Applied Pattern Recognition},
    volume={4},
    number={2},
    year={2017}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 51
    NUM_CHANNELS = 24
    _CLASSES = ["i_have_command", "all_clear", "not_clear", "spread_wings", "fold_wings", "lock_wings"]
    # Labels as written in the archive's files, in the order of _CLASSES (the archive numbers the signals 1-6).
    _LABELS = [f"{signal}.0" for signal in range(1, 7)]

    def _info(self):
        return DatasetInfo(
            description="Body sensor recordings of six NATOPS aircraft handling signals (UEA archive).",
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
        yield from iter_examples(data_path, f"NATOPS_{split.upper()}.ts", shape, self._LABELS)
