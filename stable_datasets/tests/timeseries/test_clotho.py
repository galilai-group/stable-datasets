import os
import re

from loguru import logger as logging
from torchcodec.decoders import AudioDecoder

from stable_datasets.timeseries.clotho import Clotho


def test_clotho_dataset():
    # Test 1: Checks number of samples
    download_dir = f"/cs/data/people/{os.getenv('USER')}/.stable_datasets/downloads"
    processed_cache_dir = f"/cs/data/people/{os.getenv('USER')}/.stable_datasets/processed"
    clotho_train = Clotho(
        split="train",
        download_dir=download_dir,
        processed_cache_dir=processed_cache_dir,
    )
    expected_num_samples = 2893
    assert len(clotho_train) == expected_num_samples, (
        f"Expected {expected_num_samples} training samples, got {len(clotho_train)}."
    )

    # Test 2: Checks sample keys
    sample = clotho_train[0]
    expected_keys = {
        "audio",
        "captions",
        "keywords",
        "freesound_id",
        "freesound_link",
        "start_sample",
        "end_sample",
        "manufacturer",
        "license",
    }
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Checks sample value types
    audio = sample["audio"]
    assert isinstance(audio, AudioDecoder), f"Audio field should be an AudioDecoder, got {type(audio)}."

    captions = sample["captions"]
    assert isinstance(captions, list), f"Captions field should be a list, got {type(captions)}."
    for caption in captions:
        assert isinstance(caption, str), f"Each caption should be a string, got {type(caption)}."

    keywords = sample["keywords"]
    assert isinstance(keywords, list), f"Keywords field should be a list, got {type(keywords)}."
    for keyword in keywords:
        assert isinstance(keyword, str), f"Each keyword should be a string, got {type(keyword)}."

    freesound_id = sample["freesound_id"]
    assert isinstance(freesound_id, int) or freesound_id is None, (
        f"Freesound ID field should be an integer or None, got {type(freesound_id)}."
    )
    freesound_link = sample["freesound_link"]
    assert isinstance(freesound_link, str) or freesound_link is None, (
        f"Freesound link field should be a string or None, got {type(freesound_link)}."
    )
    start_sample = sample["start_sample"]
    assert isinstance(start_sample, int), f"Start sample field should be an integer, got {type(start_sample)}."
    end_sample = sample["end_sample"]
    assert isinstance(end_sample, int), f"End sample field should be an integer, got {type(end_sample)}."
    manufacturer = sample["manufacturer"]
    assert isinstance(manufacturer, str), f"Manufacturer field should be a string, got {type(manufacturer)}."
    license = sample["license"]
    assert isinstance(license, str), f"License field should be a string, got {type(license)}."

    # Test 4: Checks sample value properties
    expected_sample_rate = 44100
    assert audio.metadata.sample_rate == expected_sample_rate, (
        f"Audio should be at {expected_sample_rate} Hz, got {audio.metadata.sample_rate}."
    )
    audio_data = audio.get_all_samples().data
    assert len(audio_data.shape) == 2, f"Audio data should have 2 dimensions, got {len(audio_data.shape)}."
    expected_num_channels = 1
    assert audio_data.shape[0] == expected_num_channels, (
        f"Audio data should have {expected_num_channels} channels, got {audio_data.shape[0]}."
    )
    assert (start_sample is None and end_sample is None) or (
        start_sample is not None and end_sample is not None and start_sample < end_sample
    ), f"Start sample should be less than end sample, got {start_sample} and {end_sample}."

    assert len(captions) == 5, f"Captions field should have 5 elements, got {len(captions)}."
    for caption in captions:
        num_words = len(caption.split())
        assert num_words >= 8 and num_words <= 20, f"Each caption should have between 8 and 20 words, got {num_words}."
    assert len(keywords) > 0, "Keywords field should not be empty."

    if not (freesound_id is None and freesound_link is None):
        assert isinstance(freesound_id, int), f"Freesound ID field should be an integer, got {type(freesound_id)}."
        assert isinstance(freesound_link, str), f"Freesound link field should be a string, got {type(freesound_link)}."
        pattern = rf"^https://freesound\.org/people/[^/]+/sounds/{freesound_id}$"
        assert re.match(pattern, freesound_link), f"Freesound link should match the pattern, got {freesound_link}."

    # Test 5: Checks number of samples in validation split
    clotho_validation = Clotho(
        split="validation",
        download_dir=download_dir,
        processed_cache_dir=processed_cache_dir,
    )
    expected_num_samples = 1045
    assert len(clotho_validation) == expected_num_samples, (
        f"Expected {expected_num_samples} validation samples, got {len(clotho_validation)}."
    )

    logging.info("All Clotho dataset tests passed successfully!")


if __name__ == "__main__":
    test_clotho_dataset()
