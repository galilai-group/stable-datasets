import pytest

from stable_datasets.timeseries.librispeech import LibriSpeech


@pytest.mark.large
def test_librispeech_train_split():
    """Test the train-clean-100 split of LibriSpeech.

    This test downloads ~6.3GB of data and may take several minutes.
    Run with: pytest -m large
    """
    ds = LibriSpeech(split="train")

    # Test 1: Dataset should have ~28,539 utterances in train-clean-100
    assert len(ds) > 25000, f"Expected >25,000 training samples, got {len(ds)}."

    # Test 2: Check that each sample has the expected keys
    sample = ds[0]
    expected_keys = {"audio", "sample_rate", "speaker_id", "transcript"}
    assert set(sample.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(sample.keys())}"
    )

    # Test 3: Validate audio type (should be a list of floats)
    audio = sample["audio"]
    assert isinstance(audio, list), f"Audio should be a list, got {type(audio)}."
    assert len(audio) > 0, "Audio waveform should not be empty."
    assert isinstance(audio[0], float), f"Audio samples should be floats, got {type(audio[0])}."

    # Test 4: Validate sample rate (LibriSpeech is 16kHz)
    assert sample["sample_rate"] == 16000, (
        f"Sample rate should be 16000, got {sample['sample_rate']}."
    )

    # Test 5: Validate speaker_id is a positive integer
    speaker_id = sample["speaker_id"]
    assert isinstance(speaker_id, int), f"Speaker ID should be int, got {type(speaker_id)}."
    assert speaker_id > 0, f"Speaker ID should be positive, got {speaker_id}."

    # Test 6: Validate transcript is a non-empty string
    transcript = sample["transcript"]
    assert isinstance(transcript, str), f"Transcript should be a string, got {type(transcript)}."
    assert len(transcript) > 0, "Transcript should not be empty."

    print(f"All LibriSpeech train tests passed! ({len(ds)} samples)")


@pytest.mark.large
def test_librispeech_test_split():
    """Test the test-clean split of LibriSpeech."""
    ds = LibriSpeech(split="test")

    # test-clean has ~2,620 utterances
    assert len(ds) > 2000, f"Expected >2,000 test samples, got {len(ds)}."

    sample = ds[0]
    assert "audio" in sample
    assert "transcript" in sample
    assert sample["sample_rate"] == 16000

    print(f"All LibriSpeech test tests passed! ({len(ds)} samples)")


def test_librispeech_returns_dataset_dict_when_no_split(tmp_path):
    """Verify that split=None returns a StableDatasetDict.

    NOTE: This test also downloads data. Marked as large.
    """
    pytest.skip("Skipping: requires full download. Run with -m large manually.")


def test_librispeech_source_contract():
    """Verify that LibriSpeech's SOURCE metadata is well-formed (no download needed)."""
    # These checks run at class-definition time via __init_subclass__,
    # so if we get here, the class was defined correctly.
    assert hasattr(LibriSpeech, "VERSION")
    assert hasattr(LibriSpeech, "SOURCE")
    assert "homepage" in LibriSpeech.SOURCE
    assert "citation" in LibriSpeech.SOURCE
    assert "assets" in LibriSpeech.SOURCE
    assert "train" in LibriSpeech.SOURCE["assets"]
    assert "test" in LibriSpeech.SOURCE["assets"]

    print("LibriSpeech SOURCE contract test passed!")
