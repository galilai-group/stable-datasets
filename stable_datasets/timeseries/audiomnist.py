import zipfile

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


class AudioMNIST(BaseDatasetBuilder):
    """AudioMNIST spoken-digit classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/soerenab/AudioMNIST",
        assets={
            "train": DownloadInfo(
                url="https://github.com/soerenab/AudioMNIST/archive/master.zip",
                filename="AudioMNIST-master.zip",
            ),
        },
        citation="""@article{audiomnist2023,
                     title = {AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark},
                     journal = {Journal of the Franklin Institute},
                     year = {2023},
                     issn = {0016-0032},
                     doi = {https://doi.org/10.1016/j.jfranklin.2023.11.038},
                     url = {https://www.sciencedirect.com/science/article/pii/S0016003223007536},
                     author = {Sören Becker and Johanna Vielhaben and Marcel Ackermann and Klaus-Robert Müller and Sebastian Lapuschkin and Wojciech Samek},
                     keywords = {Deep learning, Neural networks, Interpretability, Explainable artificial intelligence, Audio classification, Speech recognition}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="AudioMNIST spoken-digit recordings with speaker metadata.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(num_classes=10),
                    "speaker_id": Value("int32"),
                    "utterance_id": Value("int32"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with zipfile.ZipFile(data_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".wav"):
                    continue
                filename = name.rsplit("/", 1)[-1]
                stem = filename[:-4]
                try:
                    digit_str, speaker_str, utterance_str = stem.split("_")
                except ValueError:
                    continue
                yield (
                    filename,
                    {
                        "series": wav_bytes_to_series(archive.read(name)),
                        "label": int(digit_str),
                        "speaker_id": int(speaker_str) - 1,
                        "utterance_id": int(utterance_str),
                        "filename": filename,
                    },
                )
