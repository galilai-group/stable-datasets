"""Checks for persistent numeric cache contracts and training-path equivalence."""

import json

import numpy as np
import pytest
import torch

from profiling.signal_cache import SignalCache, SignalMetadata, write_signal_cache
from profiling.signal_training import TrainingConfig, run_comparison


@pytest.fixture
def cache_inputs():
    """Include negative gain and offset, preserving native int16 samples."""
    digital = np.random.default_rng(4).integers(-1000, 1000, size=(1024, 3), dtype=np.int16)
    gain = np.array([1e-6, -2e-6, 3e-6], dtype=np.float32)
    offset = np.array([0, 1e-6, -1e-6], dtype=np.float32)
    # MNE's transposed recording commonly arrives in Fortran memory order.
    physical = np.asfortranarray(digital.astype(np.float32) * gain + offset)
    metadata = SignalMetadata(("EEG0", "EEG1", "AUX"), 256.0, {"source": "test"})
    return physical, digital, gain, offset, metadata


def test_persistent_cache_preserves_order_duplicates_units_and_native_samples(tmp_path, cache_inputs):
    directory = tmp_path / "cache"
    write_signal_cache(directory, *cache_inputs)
    cache = SignalCache.open(directory)
    assert cache.metadata.channel_names == ("EEG0", "EEG1", "AUX")
    assert cache.manifest["physical_unit"] == "V"
    assert not cache.physical.flags.writeable and not cache.digital.flags.writeable
    assert cache.physical.flags.c_contiguous and cache.digital.flags.c_contiguous
    starts = np.array([1000, 0, 1000, 511])
    for representation, source in (("float32", cache_inputs[0]), ("int16", cache_inputs[1])):
        output = np.empty((4, 24, 3), dtype=source.dtype)
        cache.gather(representation, starts, output)
        np.testing.assert_array_equal(output, np.stack([source[start : start + 24] for start in starts]))
    with pytest.raises(FileExistsError):
        write_signal_cache(directory, *cache_inputs)


def test_invalid_windows_fail_before_mutating_output(tmp_path, cache_inputs):
    directory = tmp_path / "cache"
    write_signal_cache(directory, *cache_inputs)
    cache = SignalCache.open(directory)
    output = np.full((2, 128, 3), 9, dtype=np.float32)
    with pytest.raises(IndexError):
        cache.gather("float32", np.array([0, 1024]), output)
    assert np.all(output == 9)
    with pytest.raises(ValueError, match="matching dtype"):
        cache.gather("int16", np.array([0, 1]), output)


def test_mismatched_calibration_never_publishes_cache(tmp_path, cache_inputs):
    physical, digital, gain, offset, metadata = cache_inputs
    with pytest.raises(AssertionError):
        write_signal_cache(tmp_path / "invalid", physical, digital, gain * 2, offset, metadata)
    assert not (tmp_path / "invalid").exists()


def test_corrupted_file_and_edited_calibration_are_rejected(tmp_path, cache_inputs):
    directory = tmp_path / "cache"
    write_signal_cache(directory, *cache_inputs)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["gain"][0] *= 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        SignalCache.open(directory)
    manifest["gain"][0] /= 2
    manifest_path.write_text(json.dumps(manifest))
    with (directory / "int16.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        SignalCache.open(directory)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_serial_and_overlapped_training_match_gpu_resident_updates(tmp_path, cache_inputs):
    directory = tmp_path / "cache"
    write_signal_cache(directory, *cache_inputs)
    config = TrainingConfig(
        directory, tmp_path / "result.json", batch_size=2, length=128, steps=5, repeats=1, warmup=1
    )
    report = run_comparison(config)
    resident_loss = report["results"]["resident"]["trials"][0]["final_loss"]
    assert len(report["results"]) == 5
    for result in report["results"].values():
        trial = result["trials"][0]
        assert trial["steps_per_second"] > 0
        np.testing.assert_allclose(trial["final_loss"], resident_loss, rtol=2e-5, atol=1e-7)
