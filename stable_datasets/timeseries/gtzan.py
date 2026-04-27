import tarfile

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

from ._audio_utils import wav_bytes_to_series


GTZAN_LABELS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZAN(BaseDatasetBuilder):
    """GTZAN music genre classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="http://marsyas.info/downloads/datasets.html",
        assets={
            "train": DownloadInfo(
                url="http://opihi.cs.uvic.ca/sound/genres.tar.gz",
                filename="genres.tar.gz",
            ),
        },
        citation="""@article{tzanetakis2002musical,
                     title={Musical genre classification of audio signals},
                     author={Tzanetakis, George and Cook, Perry},
                     journal={IEEE Transactions on Speech and Audio Processing},
                     year={2002}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="GTZAN music genre classification dataset.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=GTZAN_LABELS),
                    "genre": Value("string"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with tarfile.open(data_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.name.lower().endswith(".wav"):
                    continue
                parts = member.name.split("/")
                if len(parts) < 2:
                    continue
                genre = parts[-2]
                if genre not in GTZAN_LABELS:
                    continue
                filename = parts[-1]
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield member.name, {
                    "series": wav_bytes_to_series(extracted.read()),
                    "label": GTZAN_LABELS.index(genre),
                    "genre": genre,
                    "filename": filename,
                }
