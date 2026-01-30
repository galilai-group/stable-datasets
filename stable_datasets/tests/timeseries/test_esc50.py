from datasets import Audio
from loguru import logger as logging
from torchcodec.decoders import AudioDecoder
from stable_datasets.timeseries.esc50 import ESC50

def test_esc50_dataset():
    # Test 1: Checks number of samples
    dataset = ESC50(split="test")
    expected_num_samples = 2000
    assert len(dataset) == expected_num_samples, f"Expected {expected_num_samples} samples, got {len(dataset)}."

    # Test 2: Checks sample keys
    sample = dataset[0]
    expected_keys = {"audio", "fold", "category", "major_category", "esc10", "clip_id", "take"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Checks sample value types
    audio = sample["audio"]
    assert isinstance(audio, AudioDecoder), f"Audio field should be an AudioDecoder, got {type(audio)}."
    assert isinstance(sample["fold"], int), f"Fold field should be an integer, got {type(sample['fold'])}."
    assert isinstance(sample["category"], int), f"Category field should be an integer, got {type(sample['category'])}."
    assert isinstance(sample["major_category"], int), f"Major category field should be an integer, got {type(sample['major_category'])}."
    assert isinstance(sample["esc10"], bool), f"ESC10 field should be a boolean, got {type(sample['esc10'])}."
    assert isinstance(sample["clip_id"], int), f"Freesound clip ID field should be an integer, got {type(sample['clip_id'])}."
    assert isinstance(sample["take"], str), f"Take field should be a string, got {type(sample['take'])}."

    # Test 4: Checks audio properties
    duration = 5.0
    sample_rate = 44100
    num_samples = sample_rate * duration
    num_channels = 1
    audio_data = audio.get_all_samples().data
    assert audio.metadata.duration_seconds == duration, f"Audio should be {duration} seconds long, got {audio.metadata.duration_seconds}."
    assert audio.metadata.sample_rate == sample_rate, f"Audio should be at {sample_rate} Hz, got {audio.metadata.sample_rate}."
    assert audio_data.shape == (num_channels, num_samples), f"Audio data shape should be ({num_channels}, {num_samples}), got {audio_data.shape}."

    # Test 5: Checks other values
    assert sample["fold"] in list(range(1, 6)), f"Fold should be in range [1, 5], got {sample['fold']}."
    assert sample["category"] in list(range(50)), f"Category should be in range [0, 49], got {sample['category']}."
    assert sample["major_category"] in list(range(5)), f"Major category should be in range [0, 4], got {sample['major_category']}."

    logging.info(f"All ESC-50 dataset tests passed successfully!")

if __name__ == "__main__":
    test_esc50_dataset()
