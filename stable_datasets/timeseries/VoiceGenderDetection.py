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

from ._audio_utils import audiosegment_bytes_to_series


class VoiceGenderDetection(BaseDatasetBuilder):
    """Voice gender detection dataset derived from VoxCeleb clips."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://dagshub.com/DagsHub/audio-datasets/src/main/voice_gender_detection",
        assets={
            "train": DownloadInfo(
                url="https://drive.google.com/u/0/uc?id=1HRbWocxwClGy9Fj1MQeugpR4vOaL9ebO&export=download",
                filename="VoxCeleb_gender.zip",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Voice gender detection dataset with male/female labels.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=["male", "female"]),
                    "gender": Value("string"),
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
                if name.endswith("/") or not name.lower().endswith(".m4a"):
                    continue
                parts = name.split("/")
                if len(parts) < 2:
                    continue
                gender = parts[-2]
                if gender not in {"males", "females"}:
                    continue
                filename = parts[-1]
                yield name, {
                    "series": audiosegment_bytes_to_series(archive.read(name), format="m4a"),
                    "label": 0 if gender == "males" else 1,
                    "gender": gender[:-1],
                    "filename": filename,
                }
