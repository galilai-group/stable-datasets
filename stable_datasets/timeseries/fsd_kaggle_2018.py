import csv
import io
import zipfile

import numpy as np
from scipy.io.wavfile import read as wav_read

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


class FSDKaggle2018(BaseDatasetBuilder):
    """FSDKaggle2018 sound classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/2552860",
        assets={
            "train_audio": DownloadInfo(
                url="https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1",
                fallbacks=["https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip"],
                filename="FSDKaggle2018.audio_train.zip",
            ),
            "test_audio": DownloadInfo(
                url="https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1",
                fallbacks=["https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_test.zip"],
                filename="FSDKaggle2018.audio_test.zip",
            ),
            "meta": DownloadInfo(
                url="https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip?download=1",
                fallbacks=["https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip"],
                filename="FSDKaggle2018.meta.zip",
            ),
        },
        citation="""@dataset{fonseca2019fsdkaggle2018,
                         title={FSDKaggle2018},
                         author={Fonseca, Eduardo and Plakal, Manoj and Font, Frederic and Ellis, Daniel P.W. and Serra, Xavier},
                         year={2019},
                         publisher={Zenodo},
                         doi={10.5281/zenodo.2552860}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="FSDKaggle2018 audio classification dataset with official train/test splits.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(num_classes=41),
                    "filename": Value("string"),
                    "fsid": Value("string"),
                    "verified": Value("string"),
                    "usage": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            raise RuntimeError("Expected _raw_download_dir to be set before split generation.")

        train_audio, test_audio, meta = bulk_download(
            [
                self._normalize_download_info(source["assets"]["train_audio"], asset_name="train_audio"),
                self._normalize_download_info(source["assets"]["test_audio"], asset_name="test_audio"),
                self._normalize_download_info(source["assets"]["meta"], asset_name="meta"),
            ],
            dest_folder=download_dir,
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"audio_path": train_audio, "meta_path": meta, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"audio_path": test_audio, "meta_path": meta, "split": "test"},
            ),
        ]

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.TEST]

    def _generate_examples(self, audio_path, meta_path, split):
        metadata = _load_metadata(meta_path)
        label_to_id = {
            name: idx for idx, name in enumerate(sorted({row["label"] for rows in metadata.values() for row in rows}))
        }
        split_rows = {row["filename"]: row for row in metadata[split]}

        with zipfile.ZipFile(audio_path) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".wav"):
                    continue
                filename = member.rsplit("/", 1)[-1]
                if filename not in split_rows:
                    continue
                row = split_rows[filename]
                with archive.open(member) as fh:
                    sample_rate, wav = wav_read(io.BytesIO(fh.read()))
                del sample_rate
                series = np.asarray(wav, dtype="float32")
                if series.ndim == 1:
                    series = series[:, None]

                yield (
                    filename,
                    {
                        "series": series,
                        "label": label_to_id[row["label"]],
                        "filename": filename,
                        "fsid": row.get("fsid", ""),
                        "verified": row.get("verified", ""),
                        "usage": row.get("usage", ""),
                    },
                )


def _load_metadata(meta_path) -> dict[str, list[dict[str, str]]]:
    members = {
        "train": "FSDKaggle2018.meta/train_post_competition.csv",
        "test": "FSDKaggle2018.meta/test_post_competition_scoring_clips.csv",
    }
    out = {}
    with zipfile.ZipFile(meta_path) as archive:
        for split, member in members.items():
            with archive.open(member) as fh:
                rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
            normalized = []
            for row in rows:
                normalized.append(
                    {
                        "filename": row["fname"],
                        "label": row["label"],
                        "verified": row.get("manually_verified", ""),
                        "usage": row.get("usage", ""),
                        "fsid": row.get("fsID", ""),
                    }
                )
            out[split] = normalized
    return out
