import csv
import datasets
from pathlib import Path
import os

from loguru import logger as logging

from stable_datasets.utils import BaseDatasetBuilder


class ESC50(BaseDatasetBuilder):
    """ESC-50: Environmental Sound Classification Dataset with 50 Classes"""

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://github.com/karolpiczak/ESC-50",
        "assets": {
            "test": "https://github.com/karoldvl/ESC-50/archive/master.zip",
        },
        "citation": """@inproceedings{piczak2015dataset,
                        title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
                        author = {Piczak, Karol J.},
                        booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
                        date = {2015-10-13},
                        url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
                        doi = {10.1145/2733373.2806390},
                        location = {{Brisbane, Australia}},
                        isbn = {978-1-4503-3459-4},
                        publisher = {{ACM Press}},
                        pages = {1015--1018}
                    }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""The ESC-50 dataset contains audio recordings from various environments, and has been used for benchmarking many methods of environmental sound classification. It holds 2000 labeled environmental sounds, each of length 5 seconds, which have been grouped into 50 evenly-sized classes that have further been grouped into 5 major categories.""",
            features=datasets.Features(
                {
                    # Useful info for training
                    "audio": datasets.Audio(sampling_rate=44100),  # All audio files in ESC-50 are at 44.1 kHz
                    "fold": datasets.Value("int32"),
                    "category": datasets.ClassLabel(names=self._categories()),
                    "major_category": datasets.ClassLabel(names=self._major_categories()),

                    # Data source info
                    "esc10": datasets.Value("bool"),
                    "clip_id": datasets.Value("int32"),
                    "take": datasets.Value("string"),
                }
            ),
            supervised_keys=("audio", "category"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    # Only defined this function to use the existing download_and_extract() utility
    def _split_generators(self, dl_manager):
        url = self._source()["assets"]["test"]
        downloaded_file = dl_manager.download_and_extract(url)
        logging.info(f"downloaded_file: {downloaded_file}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_path": downloaded_file,
                    "split": "test",
                },
            ),
        ]

    # Note: The esc50-human.xlsx file contains per-category information, not per-example information,
    # so we do not use it here
    def _generate_examples(self, data_path, split):
        # Finds paths
        download_root_dir = Path(data_path) / "ESC-50-master"
        csv_path = download_root_dir / "meta" / "esc50.csv"
        audio_dir = download_root_dir / "audio"        
        logging.info(f"Reading CSV from {csv_path}")
        logging.info(f"Audio files in {audio_dir}")
        
        # Reads and parses the CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Finds audio path
                filename = row["filename"]
                audio_path = audio_dir / filename
                if not audio_path.exists():
                    logging.warning(f"Audio file not found: {audio_path}")
                    continue

                # Makes sure the filename is formatted correctly
                fold, clip_id, take, target = os.path.splitext(filename)[0].split("-")
                fold = int(fold)
                clip_id = int(clip_id)
                target = int(target)
                assert fold == int(row["fold"])
                assert clip_id == int(row["src_file"])
                assert take == row["take"]
                assert target == int(row["target"])
                
                yield idx, {
                    "audio": str(audio_path),
                    "fold": fold,
                    "category": target,
                    "major_category": target // 10,
                    "esc10": row["esc10"].lower() == "true",
                    "clip_id": clip_id,
                    "take": take,
                }

    @staticmethod
    def _categories():
        return [
            "dog",
            "rooster",
            "pig",
            "cow",
            "frog",
            "cat",
            "hen",
            "insects",
            "sheep",
            "crow",
            "rain",
            "sea_waves",
            "crackling_fire",
            "crickets",
            "chirping_birds",
            "water_drops",
            "wind",
            "pouring_water",
            "toilet_flush",
            "thunderstorm",
            "crying_baby",
            "sneezing",
            "clapping",
            "breathing",
            "coughing",
            "footsteps",
            "laughing",
            "brushing_teeth",
            "snoring",
            "drinking_sipping",
            "door_wood_knock",
            "mouse_click",
            "keyboard_typing",
            "door_wood_creaks",
            "can_opening",
            "washing_machine",
            "vacuum_cleaner",
            "clock_alarm",
            "clock_tick",
            "glass_breaking",
            "helicopter",
            "chainsaw",
            "siren",
            "car_horn",
            "engine",
            "train",
            "church_bells",
            "airplane",
            "fireworks",
            "hand_saw",
        ]

    @staticmethod
    def _major_categories():
        return [
            "animals",
            "natural_soundscapes_and_water_sounds",
            "human_sounds",
            "interior_or_domestic_sounds",
            "exterior_or_urban_sounds",
        ]
