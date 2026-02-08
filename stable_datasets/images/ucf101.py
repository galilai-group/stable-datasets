import io
from pathlib import Path
import os
import zipfile
import rarfile
import subprocess
from urllib.parse import urlparse
import re

import aiohttp
import datasets
import numpy as np
import scipy.io as sio
from loguru import logger as logging

from stable_datasets.utils import BaseDatasetBuilder


CURRENT_VERSION = datasets.Version("1.0.0")


def _wget_download(url: str, dest_folder: Path) -> Path:
    """Download a file using wget with resume support.

    Adapted from CLEVRER's dataset builder.
    """
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    local_path = dest_folder / filename
    if os.path.isfile(local_path):
        return local_path

    cmd = ["wget", "-c", "--no-check-certificate", "--progress=bar:force:noscroll", "-P", str(dest_folder), url]
    logging.info(f"Downloading (or resuming): {url}")
    subprocess.run(cmd, check=True, cwd=str(dest_folder))

    return local_path


class UCF101Config(datasets.BuilderConfig):
    def __init__(self, variant, **kwargs):
        super().__init__(version=CURRENT_VERSION, **kwargs)
        self.variant = variant


class UCF101(BaseDatasetBuilder):
    """UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild"""

    VERSION = CURRENT_VERSION

    SOURCE = {
        "homepage": "https://www.crcv.ucf.edu/data/UCF101.php",
        "citation": """@inproceedings{UCF101,
                        author = {Soomro, K. and Roshan Zamir, A. and Shah, M.},
                        booktitle = {CRCV-TR-12-01},
                        title = {{UCF101}: A Dataset of 101 Human Actions Classes From Videos in The Wild},
                        year = {2012}
                    }""",
        "assets": {
            "all_videos": "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar",
            "action_recognition_train_test_splits": "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip",
            "action_detection_train_test_splits": "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-DetectionTask.zip",
        },
    }

    BUILDER_CONFIGS = [
        UCF101Config(name="action_recognition_01", variant="action_recognition_01"),
        UCF101Config(name="action_recognition_02", variant="action_recognition_02"),
        UCF101Config(name="action_recognition_03", variant="action_recognition_03"),
        UCF101Config(name="action_detection_01", variant="action_detection_01"),
        UCF101Config(name="action_detection_02", variant="action_detection_02"),
        UCF101Config(name="action_detection_03", variant="action_detection_03"),
    ]

    def _info(self):
        # Checks variant
        variant = self.config.variant  # The variant is already checked to be valid in DatasetBuilder._create_builder_config()
        if "action_recognition" in variant:
            action_classes = self._action_recognition_classes()
        elif "action_detection" in variant:
            action_classes = self._action_detection_classes()

        return datasets.DatasetInfo(
            description="""The UCF-101 dataset is an action recognition dataset of videos recorded in the wild depicting 101 human action classes. The dataset contains 13,320 videos, with a minimum of 100 videos per class. See https://www.crcv.ucf.edu/data/UCF101.php for more information.""",
            features=datasets.Features(
                {
                    "video": datasets.Video(),
                    "fine_label": datasets.ClassLabel(names=action_classes),
                    "coarse_label": datasets.ClassLabel(names=self._action_types()),
                }
            ),
            supervised_keys=("video", "fine_label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager):
        # Downloads everything
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        download_dir = Path(download_dir)
        
        videos_path = _wget_download(self.SOURCE["assets"]["all_videos"], download_dir)
        action_recognition_splits_path = _wget_download(self.SOURCE["assets"]["action_recognition_train_test_splits"], download_dir)
        action_detection_splits_path = _wget_download(self.SOURCE["assets"]["action_detection_train_test_splits"], download_dir)
        logging.info(f"Archive path for videos: {videos_path}")
        logging.info(f"Archive path for action recognition train-test splits: {action_recognition_splits_path}")
        logging.info(f"Archive path for action detection train-test splits: {action_detection_splits_path}")

        # Unarchives everything
        to_unarchive = [
            (videos_path, "UCF-101"),
            (action_recognition_splits_path, "ucfTrainTestlist"),
            (action_detection_splits_path, "UCF101_Action_detection_splits"),
        ]
        for path, dir_name in to_unarchive:
            abs_dir_name = os.path.abspath(os.path.join(path.parent, dir_name))
            if not os.path.isdir(abs_dir_name):
                if str(path).endswith(".rar"):
                    with rarfile.RarFile(path) as rf:
                        rf.extractall(path=path.parent)
                elif str(path).endswith(".zip"):
                    with zipfile.ZipFile(path) as zf:
                        zf.extractall(path=path.parent)
            assert os.path.isdir(abs_dir_name)
            logging.info(f"Unarchived files are at {abs_dir_name}")

        videos_dir = os.path.join(videos_path.parent, "UCF-101")
        if "action_recognition" in self.config.variant:
            train_test_splits_dir = os.path.join(action_recognition_splits_path.parent, "ucfTrainTestlist")
        else:
            train_test_splits_dir = os.path.join(action_detection_splits_path.parent, "UCF101_Action_detection_splits")
        logging.info(f"Config variant: {self.config.variant}")
        logging.info(f"Videos directory: {videos_dir}")
        logging.info(f"Train-test splits directory for config variant: {train_test_splits_dir}")

        # Returns both splits
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "videos_dir": videos_dir,
                    "train_test_splits_dir": train_test_splits_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "videos_dir": videos_dir,
                    "train_test_splits_dir": train_test_splits_dir,
                },
            ),
        ]

    def _generate_examples(self, split, videos_dir, train_test_splits_dir):
        # Gets label indices
        fine_label_names = self._action_recognition_classes() \
            if "action_recognition" in self.config.variant \
            else self._action_detection_classes()
        fine_label_name_to_idx = {name: idx for idx, name in enumerate(fine_label_names)}

        fine_name_to_coarse_name = {fine: coarse for coarse in self._action_types() for fine in self._action_type_to_classes()[coarse]}
        coarse_label_name_to_idx = {name: idx for idx, name in enumerate(self._action_types())}

        # Iterates over the split's item list
        list_num = re.search(r"(\d+)$", self.config.variant).group(1)
        video_list_file = os.path.join(train_test_splits_dir, f"{split}list{list_num}.txt")
        with open(video_list_file) as f:
            for idx, line in enumerate(f):
                stripped_line = line.strip()
                video_rel_path = stripped_line.split(" ")[0]
                video_path = os.path.abspath(os.path.join(videos_dir, video_rel_path))

                fine_label_name = video_rel_path.split("/")[0]
                fine_label_idx = fine_label_name_to_idx[fine_label_name]

                coarse_label_name = fine_name_to_coarse_name[fine_label_name]
                coarse_label_idx = coarse_label_name_to_idx[coarse_label_name]

                yield idx, {"video": video_path, "fine_label": fine_label_idx, "coarse_label": coarse_label_idx}

    @staticmethod
    def _action_recognition_classes():
        return [
            "ApplyEyeMakeup",
            "ApplyLipstick",
            "Archery",
            "BabyCrawling",
            "BalanceBeam",
            "BandMarching",
            "BaseballPitch",
            "Basketball",
            "BasketballDunk",
            "BenchPress",
            "Biking",
            "Billiards",
            "BlowDryHair",
            "BlowingCandles",
            "BodyWeightSquats",
            "Bowling",
            "BoxingPunchingBag",
            "BoxingSpeedBag",
            "BreastStroke",
            "BrushingTeeth",
            "CleanAndJerk",
            "CliffDiving",
            "CricketBowling",
            "CricketShot",
            "CuttingInKitchen",
            "Diving",
            "Drumming",
            "Fencing",
            "FieldHockeyPenalty",
            "FloorGymnastics",
            "FrisbeeCatch",
            "FrontCrawl",
            "GolfSwing",
            "Haircut",
            "Hammering",
            "HammerThrow",
            "HandstandPushups",
            "HandstandWalking",
            "HeadMassage",
            "HighJump",
            "HorseRace",
            "HorseRiding",
            "HulaHoop",
            "IceDancing",
            "JavelinThrow",
            "JugglingBalls",
            "JumpingJack",
            "JumpRope",
            "Kayaking",
            "Knitting",
            "LongJump",
            "Lunges",
            "MilitaryParade",
            "Mixing",
            "MoppingFloor",
            "Nunchucks",
            "ParallelBars",
            "PizzaTossing",
            "PlayingCello",
            "PlayingDaf",
            "PlayingDhol",
            "PlayingFlute",
            "PlayingGuitar",
            "PlayingPiano",
            "PlayingSitar",
            "PlayingTabla",
            "PlayingViolin",
            "PoleVault",
            "PommelHorse",
            "PullUps",
            "Punch",
            "PushUps",
            "Rafting",
            "RockClimbingIndoor",
            "RopeClimbing",
            "Rowing",
            "SalsaSpin",
            "ShavingBeard",
            "Shotput",
            "SkateBoarding",
            "Skiing",
            "Skijet",
            "SkyDiving",
            "SoccerJuggling",
            "SoccerPenalty",
            "StillRings",
            "SumoWrestling",
            "Surfing",
            "Swing",
            "TableTennisShot",
            "TaiChi",
            "TennisSwing",
            "ThrowDiscus",
            "TrampolineJumping",
            "Typing",
            "UnevenBars",
            "VolleyballSpiking",
            "WalkingWithDog",
            "WallPushups",
            "WritingOnBoard",
            "YoYo",
        ]

    @staticmethod
    def _action_detection_classes():
        return [
            "Basketball",
            "BasketballDunk",
            "Biking",
            "CliffDiving",
            "CricketBowling",
            "Diving",
            "Fencing",
            "FloorGymnastics",
            "GolfSwing",
            "HorseRiding",
            "IceDancing",
            "LongJump",
            "PoleVault",
            "RopeClimbing",
            "SalsaSpin",
            "SkateBoarding",
            "Skiing",
            "Skijet",
            "SoccerJuggling",
            "Surfing",
            "TennisSwing",
            "TrampolineJumping",
            "VolleyballSpiking",
            "WalkingWithDog",
        ]

    @staticmethod
    def _action_types():
        return [
            "HumanObjectInteraction",
            "BodyMotionOnly",
            "HumanHumanInteraction",
            "PlayingMusicalInstruments",
            "Sports",
        ]

    @staticmethod
    def _action_type_to_classes():
        return {
            "HumanObjectInteraction": {
                "HulaHoop",
                "JugglingBalls",
                "JumpRope",
                "Mixing",
                "Nunchucks",
                "PizzaTossing",
                "SkateBoarding",
                "SoccerJuggling",
                "YoYo",
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BlowDryHair",
                "BrushingTeeth",
                "CuttingInKitchen",
                "Hammering",
                "Knitting",
                "MoppingFloor",
                "ShavingBeard",
                "Typing",
                "WritingOnBoard",
            },
            "BodyMotionOnly": {
                "JumpingJack",
                "Lunges",
                "PullUps",
                "PushUps",
                "RockClimbingIndoor",
                "RopeClimbing",
                "Swing",
                "TaiChi",
                "TrampolineJumping",
                "WalkingWithDog",
                "BabyCrawling",
                "BlowingCandles",
                "BodyWeightSquats",
                "HandstandPushups",
                "HandstandWalking",
                "WallPushups",
            },
            "HumanHumanInteraction": {
                "MilitaryParade",
                "SalsaSpin",
                "BandMarching",
                "Haircut",
                "HeadMassage",
            },
            "PlayingMusicalInstruments": {
                "Drumming",
                "PlayingGuitar",
                "PlayingPiano",
                "PlayingTabla",
                "PlayingViolin",
                "PlayingCello",
                "PlayingDaf",
                "PlayingDhol",
                "PlayingFlute",
                "PlayingSitar",
            },
            "Sports": {
                "BaseballPitch",
                "Basketball",
                "BenchPress",
                "Biking",
                "Billiards",
                "BreastStroke",
                "CleanAndJerk",
                "Diving",
                "Fencing",
                "GolfSwing",
                "HighJump",
                "HorseRace",
                "HorseRiding",
                "JavelinThrow",
                "Kayaking",
                "PoleVault",
                "PommelHorse",
                "Punch",
                "Rowing",
                "Skiing",
                "Skijet",
                "TennisSwing",
                "ThrowDiscus",
                "VolleyballSpiking",
                "Archery",
                "BalanceBeam",
                "BasketballDunk",
                "Bowling",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "CliffDiving",
                "CricketBowling",
                "CricketShot",
                "FieldHockeyPenalty",
                "FloorGymnastics",
                "FrisbeeCatch",
                "FrontCrawl",
                "HammerThrow",
                "IceDancing",
                "LongJump",
                "ParallelBars",
                "Rafting",
                "Shotput",
                "SkyDiving",
                "SoccerPenalty",
                "StillRings",
                "SumoWrestling",
                "Surfing",
                "TableTennisShot",
                "UnevenBars",
            },
        }
