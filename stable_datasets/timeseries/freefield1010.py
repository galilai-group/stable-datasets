import csv
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
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download

from ._audio_utils import wav_bytes_to_series


class Freefield1010(BaseDatasetBuilder):
    """Freefield1010 bird-presence classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads",
        assets={
            "audio": DownloadInfo(
                url="https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
                filename="ff1010bird_wav.zip",
            ),
            "metadata": DownloadInfo(
                url="https://ndownloader.figshare.com/files/6035814",
                filename="ff1010bird_metadata.csv",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Freefield1010 binary bird-presence audio classification dataset.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=["no_bird", "bird"]),
                    "recording_id": Value("string"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        audio_path, metadata_path = bulk_download(
            [
                self._normalize_download_info(source["assets"]["audio"], asset_name="audio"),
                self._normalize_download_info(source["assets"]["metadata"], asset_name="metadata"),
            ],
            dest_folder=self._raw_download_dir,
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"audio_path": audio_path, "metadata_path": metadata_path, "split": "train"},
            )
        ]

    def _candidate_splits(self) -> list:
        return [Split.TRAIN]

    def _generate_examples(self, audio_path, metadata_path, split):
        del split
        with open(metadata_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        with zipfile.ZipFile(audio_path) as archive:
            for row in rows:
                recording_id = str(row.get("itemid", row.get("id", "")))
                filename = f"{recording_id}.wav"
                member = _zip_member_by_suffix(archive, f"wav/{filename}")
                yield recording_id, {
                    "series": wav_bytes_to_series(archive.read(member)),
                    "label": int(row.get("hasbird", row.get("label", 0))),
                    "recording_id": recording_id,
                    "filename": filename,
                }


def _zip_member_by_suffix(archive: zipfile.ZipFile, suffix: str) -> str:
    suffix = suffix.lower()
    for name in archive.namelist():
        if name.lower().endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find {suffix!r} in {archive.filename}")
