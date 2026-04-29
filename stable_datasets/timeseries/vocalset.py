import zipfile

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Sequence, Value, Version
from stable_datasets.utils import BaseDatasetBuilder

from ._audio_utils import wav_bytes_to_series


class VocalSet(BaseDatasetBuilder):
    """VocalSet singing voice dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/1442513",
        assets={
            "train": DownloadInfo(
                url="https://zenodo.org/record/1442513/files/VocalSet11.zip?download=1",
                fallbacks=["https://zenodo.org/records/1442513/files/VocalSet11.zip"],
                filename="VocalSet11.zip",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="VocalSet singing voice recordings with singer, gender, and vowel metadata.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "singer": Value("string"),
                    "gender": Value("string"),
                    "vowel": Value("string"),
                    "relative_path": Value("string"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with zipfile.ZipFile(data_path) as archive:
            for filename in sorted(archive.namelist()):
                if not filename.lower().endswith(".wav") or "excerpts" in filename or filename.startswith("_"):
                    continue
                vowel = filename[-5]
                if vowel not in {"a", "e", "i", "o", "u"}:
                    continue
                parts = filename.split("/")
                if len(parts) < 3:
                    continue
                singer = parts[1]
                gender = "".join(ch for ch in singer if ch.isalpha())
                yield (
                    filename,
                    {
                        "series": wav_bytes_to_series(archive.read(filename)),
                        "singer": singer,
                        "gender": gender,
                        "vowel": vowel,
                        "relative_path": filename,
                        "filename": parts[-1],
                    },
                )
