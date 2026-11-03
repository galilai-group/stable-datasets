import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import UWaveGestureLibrary


def test_uwave_gesture_library_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = UWaveGestureLibrary(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (120, 320)

    # Test 2: Check the sample fields, types and class names.
    first = test[0]
    assert set(first) == {"series", "label"}
    assert test.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert test.features["label"].names == ["1", "2", "3", "4", "5", "6", "7", "8"]

    # Test 3: Check the shapes and class balance: every series is 315 steps of 3 channels.
    shape = (UWaveGestureLibrary.SEQUENCE_LENGTH, UWaveGestureLibrary.NUM_CHANNELS)
    assert shape == (315, 3)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (120, *shape)
    assert series["test"].shape == (320, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [15, 15, 15, 15, 15, 15, 15, 15]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [40, 40, 40, 40, 40, 40, 40, 40]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(series["test"][0, 0], [-0.48569, -1.4682, -1.6373], rtol=1e-6)
    np.testing.assert_allclose(series["test"][-1, -1], [-0.063019, -0.087823, -0.13839], rtol=1e-6)
    assert (first["label"], test[-1]["label"]) == (0, 7)
    assert np.abs(series["train"]).sum() == pytest.approx(94895.523632, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(254237.709841, rel=1e-9)
