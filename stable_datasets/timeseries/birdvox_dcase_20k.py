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


class BirdVoxDCASE20k(BaseDatasetBuilder):
    """BirdVox-DCASE-20k bird-detection dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://wp.nyu.edu/birdvox/birdvox-dcase-20k/",
        assets={
            "audio": DownloadInfo(
                url="https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip?download=1",
                fallbacks=["https://zenodo.org/records/1208080/files/BirdVox-DCASE-20k.zip"],
                filename="BirdVox-DCASE-20k.zip",
            ),
            "metadata": DownloadInfo(
                url="https://ndownloader.figshare.com/files/10853300",
                filename="data_labels.csv",
            ),
        },
        citation="""@inproceedings{lostanlen2018icassp,
                     title={BirdVox-full-night: a dataset and benchmark for avian flight call detection},
                     author={Lostanlen, Vincent and Salamon, Justin and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
                     booktitle={Proc. IEEE ICASSP},
                     year={2018}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="BirdVox-DCASE-20k binary bird-presence audio classification dataset.",
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
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        with zipfile.ZipFile(audio_path) as archive:
            for row in rows:
                recording_id = str(row.get(fieldnames[0], "")) if fieldnames else ""
                label_value = row.get("label")
                if label_value is None and len(fieldnames) >= 3:
                    label_value = row.get(fieldnames[2])
                filename = f"{recording_id}.wav"
                member = _zip_member_by_stem(archive, recording_id)
                yield (
                    recording_id,
                    {
                        "series": wav_bytes_to_series(archive.read(member)),
                        "label": int(label_value),
                        "recording_id": recording_id,
                        "filename": filename,
                    },
                )


def _zip_member_by_stem(archive: zipfile.ZipFile, stem: str) -> str:
    suffix = f"{stem}.wav"
    for name in archive.namelist():
        if name.endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find {suffix!r} in {archive.filename}")
