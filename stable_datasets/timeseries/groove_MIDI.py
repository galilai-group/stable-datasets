import csv
import io
import zipfile

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Sequence, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download

from ._audio_utils import soundfile_bytes_to_series


class GrooveMIDI(BaseDatasetBuilder):
    """Groove MIDI dataset with aligned synthesized audio and MIDI files."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://magenta.tensorflow.org/datasets/groove",
        assets={
            "archive": DownloadInfo(
                url="https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip",
                filename="groove-v1.0.0.zip",
            ),
        },
        citation="""@inproceedings{groove2019,
                     author={Jon Gillick and Adam Roberts and Jesse Engel and Douglas Eck and David Bamman},
                     title={Learning to Groove with Inverse Sequence Transformations},
                     booktitle={International Conference on Machine Learning (ICML)},
                     year={2019}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="Groove MIDI dataset with aligned audio, MIDI, and performance metadata.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "midi": Value("binary"),
                    "drummer": Value("string"),
                    "session": Value("string"),
                    "example_id": Value("string"),
                    "style": Value("string"),
                    "bpm": Value("string"),
                    "beat_type": Value("string"),
                    "time_signature": Value("string"),
                    "duration": Value("string"),
                    "split": Value("string"),
                    "audio_filename": Value("string"),
                    "midi_filename": Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        archive_path = bulk_download(
            [self._normalize_download_info(self._source()["assets"]["archive"], asset_name="archive")],
            dest_folder=self._raw_download_dir,
        )[0]
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": archive_path, "split": "train"}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"data_path": archive_path, "split": "validation"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data_path": archive_path, "split": "test"}),
        ]

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            with archive.open("groove/info.csv") as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))

            for row in rows:
                if row["split"] != split:
                    continue
                audio_member = f"groove/{row['audio_filename']}"
                midi_member = f"groove/{row['midi_filename']}"
                try:
                    audio_bytes = archive.read(audio_member)
                    midi_bytes = archive.read(midi_member)
                except KeyError:
                    continue
                try:
                    series = soundfile_bytes_to_series(audio_bytes)
                except RuntimeError:
                    continue
                yield (
                    row["id"],
                    {
                        "series": series,
                        "midi": midi_bytes,
                        "drummer": row["drummer"],
                        "session": row["session"],
                        "example_id": row["id"],
                        "style": row["style"],
                        "bpm": row["bpm"],
                        "beat_type": row["beat_type"],
                        "time_signature": row["time_signature"],
                        "duration": row["duration"],
                        "split": row["split"],
                        "audio_filename": row["audio_filename"],
                        "midi_filename": row["midi_filename"],
                    },
                )
