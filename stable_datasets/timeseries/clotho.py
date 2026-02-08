import csv
from pathlib import Path

import aiohttp
import datasets
from loguru import logger as logging

from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class Clotho(BaseDatasetBuilder):
    """Clotho: An Audio Captioning Dataset"""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/audio-captioning/clotho-dataset",
        "assets": {
            "train_audio": "https://zenodo.org/records/3490684/files/clotho_audio_development.7z",
            "validation_audio": "https://zenodo.org/records/3490684/files/clotho_audio_evaluation.7z",
            "train_captions": "https://zenodo.org/records/3490684/files/clotho_captions_development.csv",
            "validation_captions": "https://zenodo.org/records/3490684/files/clotho_captions_evaluation.csv",
            "train_metadata": "https://zenodo.org/records/3490684/files/clotho_metadata_development.csv",
            "validation_metadata": "https://zenodo.org/records/3490684/files/clotho_metadata_evaluation.csv",
        },
        "citation": """@inproceedings{9052990,
                        author={Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
                        booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
                        title={Clotho: an Audio Captioning Dataset},
                        year={2020},
                        volume={},
                        number={},
                        pages={736-740},
                        keywords={Training;Conferences;Employment;Signal processing;Task analysis;Speech processing;Tuning;audio captioning;dataset;Clotho},
                        doi={10.1109/ICASSP40776.2020.9052990}
                    }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The Clotho dataset is an audio captioning dataset with 4981 total samples containing general audio content and 5 captions per clip describing the content of the clip. The development (train) split has 2893 samples, the evaluation (validation) split has 1045 samples, and the test split is not included here because it is \"withheld [by the dataset's creators] for potential usage in scientific challenges\".""",
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=44100),  # All audio files in Clotho are at 44.1 kHz
                    "captions": datasets.Sequence(datasets.Value("string")),
                    "keywords": datasets.Sequence(datasets.Value("string")),
                    "freesound_id": datasets.Value("int32"),
                    "freesound_link": datasets.Value("string"),
                    "start_sample": datasets.Value("int32"),
                    "end_sample": datasets.Value("int32"),
                    "manufacturer": datasets.Value("string"),
                    "license": datasets.Value("string"),
                }
            ),
            supervised_keys=None,  # Any of the 5 captions can be used as the target
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager):
        # Configures longer timeout for large file downloads (since the
        # compressed .7z train audio file is 3.4GB in size and takes around
        # 10 minutes to download)
        dl_manager.download_config.storage_options = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=None)}}

        splits = []
        split_names = {
            "train": datasets.Split.TRAIN,
            "validation": datasets.Split.VALIDATION,
        }
        original_split_names = {
            "train": "development",
            "validation": "evaluation",
        }
        for split in split_names.keys():
            audio_path = dl_manager.download_and_extract(self._source()["assets"][f"{split}_audio"])
            captions_path, metadata_path = bulk_download(
                [self._source()["assets"][f"{split}_captions"], self._source()["assets"][f"{split}_metadata"]],
                dest_folder=self._raw_download_dir,
            )
            logging.info(f"Downloaded audio path: {audio_path}")
            logging.info(f"Downloaded captions path: {captions_path}")
            logging.info(f"Downloaded metadata path: {metadata_path}")

            splits.append(
                datasets.SplitGenerator(
                    name=split_names[split],
                    gen_kwargs={
                        "audio_path": audio_path,
                        "captions_path": captions_path,
                        "metadata_path": metadata_path,
                        "split": split,
                        "original_split": original_split_names[split],
                    },
                )
            )

        return splits

    def _generate_examples(self, audio_path, captions_path, metadata_path, split, original_split):
        # Loads the CSV files
        audio_dir = Path(audio_path) / original_split
        num_split_files = sum(1 for _ in open(captions_path, encoding="utf-8")) - 1
        captions_reader = csv.DictReader(open(captions_path, encoding="utf-8"))
        metadata_reader = csv.DictReader(open(metadata_path, encoding="utf-8"))

        # Iterates through data items
        for idx in range(num_split_files):
            # Makes sure that rows correspond to the same file name
            captions_row = next(captions_reader)
            metadata_row = next(metadata_reader)
            assert captions_row["file_name"] == metadata_row["file_name"]

            # Processes the metadata
            audio_path = str(audio_dir / captions_row["file_name"])
            keywords = metadata_row["keywords"].split(";")
            freesound_id = int(metadata_row["sound_id"]) if metadata_row["sound_id"].isdigit() else None
            freesound_link = metadata_row["sound_link"] if metadata_row["sound_link"].startswith("https://") else None
            manufacturer = metadata_row["manufacturer"]
            license = metadata_row["license"]

            captions = []
            for i in range(1, 6):
                captions.append(captions_row[f"caption_{i}"])

            start_end_samples = metadata_row["start_end_samples"]
            if start_end_samples:
                start_sample, end_sample = start_end_samples.split(", ")
                start_sample = int(start_sample[1:])
                end_sample = int(end_sample[:-1])
            else:
                start_sample = None
                end_sample = None

            # Returns the example
            yield (
                idx,
                {
                    "audio": audio_path,
                    "captions": captions,
                    "keywords": keywords,
                    "freesound_id": freesound_id,
                    "freesound_link": freesound_link,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "manufacturer": manufacturer,
                    "license": license,
                },
            )
