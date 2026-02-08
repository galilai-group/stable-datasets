import os
import random

import datasets
import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm
from torchcodec.decoders import VideoDecoder
from torchcodec import Frame
import pytest

from stable_datasets.images.ucf101 import UCF101


DOWNLOAD_DIR = f"/cs/data/people/{os.getenv('USER')}/.stable_datasets/downloads"
PROCESSED_CACHE_DIR = f"/cs/data/people/{os.getenv('USER')}/.stable_datasets/processed"


def test_ucf101_action_recognition_01_dataset():
    # Test 1: Checks the length of the train split
    train_dataset = UCF101(
        config_name="action_recognition_01",
        split="train",
        download_dir=DOWNLOAD_DIR,
        processed_cache_dir=PROCESSED_CACHE_DIR,
    )
    assert len(train_dataset) == 9537, "Train dataset should have 9537 examples, got {len(train_dataset)}"

    # Test 2: Checks that the keys are correct
    sample = train_dataset[0]
    expected_keys = {"video", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Checks sample value types
    video = sample["video"]
    assert isinstance(video, VideoDecoder), f"Video should be a VideoDecoder, got {type(video)}"
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}"

    # Test 4: Checks sample value properties
    assert 0 <= label < 101, f"Label should be between 0 and 100, got {label}"
    met = video.metadata
    assert (met.width, met.height) == (320, 240), f"Video should have width 320 and height 240, got {met.width}x{met.height}"
    assert met.average_fps == 25, f"Video should have 25 fps, got {met.fps}"
    assert met.num_frames, f"Video should have more than 0 frames"

    # Test 5: Checks the first frame of the video
    frame = video.get_frame_at(0)
    assert isinstance(frame, Frame), f"Frame should be a torchcodec.Frame, got {type(frame)}"
    assert isinstance(frame.data, torch.Tensor), f"Frame data should be a torch.Tensor, got {type(frame.data)}"
    assert frame.data.shape == (3, 240, 320), f"Frame should have shape (3, 240, 320), got {frame.data.shape}"
    assert frame.data.dtype == torch.uint8, f"Frame should have dtype uint8, got {frame.data.dtype}"

    # Test 6: Checks the length of the test split
    test_dataset = UCF101(
        config_name="action_recognition_01",
        split="test",
        download_dir=DOWNLOAD_DIR,
        processed_cache_dir=PROCESSED_CACHE_DIR,
    )
    assert len(test_dataset) == 3783, f"Test dataset should have 3783 examples, got {len(test_dataset)}"

    logging.info("UCF-101 dataset tests for config action_recognition_01 passed successfully!")


def test_other_variants():
    for task in ["action_recognition", "action_detection"]:
        split_nums = range(1, 4) if task == "action_detection" else range(2, 4)
        for split_num in split_nums:
            config_name = f"{task}_{split_num:02d}"
            logging.info(f"Testing {config_name}")
            train_dataset = UCF101(
                config_name=config_name,
                split="train",
                download_dir=DOWNLOAD_DIR,
                processed_cache_dir=PROCESSED_CACHE_DIR,
            )
            test_dataset = UCF101(
                config_name=config_name,
                split="test",
                download_dir=DOWNLOAD_DIR,
                processed_cache_dir=PROCESSED_CACHE_DIR,
            )

            expected_num_examples = 13320 if task == "action_recognition" else 3207
            assert len(train_dataset) + len(test_dataset) == expected_num_examples, f"Dataset {config_name} should have {expected_num_examples} total examples, got {len(train_dataset) + len(test_dataset)}"
    
    logging.info("UCF-101 dataset tests for other variants passed successfully!")


def test_ucf101_config_enforcement():
    with pytest.raises(ValueError) as excinfo:
        _ = UCF101(split="train")
    assert "Config name is missing" in str(excinfo.value)
    logging.info("UCF-101 config enforcement test passed!")


if __name__ == "__main__":
    test_ucf101_action_recognition_01_dataset()
    test_other_variants()
    test_ucf101_config_enforcement()
