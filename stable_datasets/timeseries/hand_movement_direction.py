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


_URL = "https://timeseriesclassification.com/aeon-toolkit/HandMovementDirection.zip"
_CHECKSUM = "sha256:953697729f390de68cc36f3912c65bb7f856ee80c5853818ed7aee7d400d27f2"


class HandMovementDirection(BaseDatasetBuilder):
    """HandMovementDirection multivariate time series classification dataset from the UEA archive.

    MEG of two healthy subjects moving a joystick with the right hand and wrist toward one of four self-chosen targets
    (BCI Competition IV, data set 3). Each trial covers 0.4 s before to 0.6 s after movement onset, 10 MEG channels
    above the motor areas at 400 Hz (400 steps), labelled ``backward``, ``forward``, ``left`` or ``right``. The archive
    keeps the competition's split but merges the two subjects: ``train`` holds 160 trials, ``test`` 74. Each
    ``series`` has shape ``(SEQUENCE_LENGTH, NUM_CHANNELS)``.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://timeseriesclassification.com/description.php?Dataset=HandMovementDirection",
        assets={
            "train": DownloadInfo(url=_URL, checksum=_CHECKSUM),
            "test": DownloadInfo(url=_URL, checksum=_CHECKSUM),
        },
        citation="""@article{tangermann2012review,
    title={Review of the BCI Competition IV},
    author={Tangermann, Michael and Müller, Klaus-Robert and Aertsen, Ad and Birbaumer, Niels and Braun, Christoph and Brunner, Clemens and Leeb, Robert and Mehring, Carsten and Miller, Kai J. and Müller-Putz, Gernot R. and Nolte, Guido and Pfurtscheller, Gert and Preissl, Hubert and Schalk, Gerwin and Schlögl, Alois and Vidaurre, Carmen and Waldert, Stephan and Blankertz, Benjamin},
    journal={Frontiers in Neuroscience},
    volume={6},
    pages={55},
    year={2012}
}
"""
        + UEA_ARCHIVE_CITATION,
        license="CC BY 4.0",
    )

    SEQUENCE_LENGTH = 400
    NUM_CHANNELS = 10
    _CLASSES = ["backward", "forward", "left", "right"]

    def _info(self):
        return DatasetInfo(
            description="MEG trials of right-hand joystick movements in four directions, BCI Competition IV data set 3 (UEA archive).",
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
        yield from iter_examples(data_path, f"HandMovementDirection_{split.upper()}.ts", shape, self._CLASSES)
