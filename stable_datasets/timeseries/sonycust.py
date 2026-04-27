import csv
import io
import tarfile

import numpy as np

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Sequence, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download

from ._audio_utils import wav_bytes_to_series


COARSE_LABELS = [
    "engine",
    "machinery-impact",
    "non-machinery-impact",
    "powered-saw",
    "alert-signal",
    "music",
    "human-voice",
    "dog",
]

FINE_LABEL_BLOCKS = [0, 4, 9, 10, 14, 19, 23, 28, 29]


class SONYCUST(BaseDatasetBuilder):
    """SONYC Urban Sound Tagging development dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/3233082",
        assets={
            "audio": DownloadInfo(
                url="https://zenodo.org/record/3233082/files/audio-dev.tar.gz?download=1",
                fallbacks=["https://zenodo.org/records/3233082/files/audio-dev.tar.gz"],
                filename="audio-dev.tar.gz",
            ),
            "annotations": DownloadInfo(
                url="https://zenodo.org/record/3233082/files/annotations-dev.csv?download=1",
                fallbacks=["https://zenodo.org/records/3233082/files/annotations-dev.csv"],
                filename="annotations-dev.csv",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="SONYC Urban Sound Tagging development dataset with fine and coarse multilabel annotations.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "fine_labels": Sequence(Value("int32")),
                    "coarse_labels": Sequence(Value("int32")),
                    "fine_label_names": Sequence(Value("string")),
                    "coarse_label_names": Sequence(Value("string")),
                    "relative_path": Value("string"),
                    "filename": Value("string"),
                    "metadata": Sequence(Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        audio_path, annotations_path = bulk_download(
            [
                self._normalize_download_info(source["assets"]["audio"], asset_name="audio"),
                self._normalize_download_info(source["assets"]["annotations"], asset_name="annotations"),
            ],
            dest_folder=self._raw_download_dir,
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"audio_path": audio_path, "annotations_path": annotations_path, "split": "train"},
            )
        ]

    def _candidate_splits(self) -> list:
        return [Split.TRAIN]

    def _generate_examples(self, audio_path, annotations_path, split):
        del split
        with open(annotations_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = list(reader)

        fine_label_names = header[4:33]
        with tarfile.open(audio_path, "r:gz") as archive:
            for idx, row in enumerate(rows):
                if len(row) < 33:
                    continue
                relative_path = f"{row[0]}/{row[2]}"
                member = _find_member(archive, relative_path)
                fine_labels = np.asarray(row[4:33], dtype="float32").astype("int32")
                coarse_labels = []
                for start, stop in zip(FINE_LABEL_BLOCKS[:-1], FINE_LABEL_BLOCKS[1:]):
                    coarse_labels.append(int(fine_labels[start:stop].max()))
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield idx, {
                    "series": wav_bytes_to_series(extracted.read()),
                    "fine_labels": fine_labels.tolist(),
                    "coarse_labels": coarse_labels,
                    "fine_label_names": fine_label_names,
                    "coarse_label_names": COARSE_LABELS,
                    "relative_path": relative_path,
                    "filename": row[2],
                    "metadata": row[:4],
                }


def _find_member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    suffix = suffix.lstrip("/")
    for member in archive.getmembers():
        if member.name.endswith(suffix):
            return member
    raise FileNotFoundError(f"Could not find {suffix!r} in {archive.name}")
