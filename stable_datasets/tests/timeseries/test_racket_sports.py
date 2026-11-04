import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import RacketSports


def test_racket_sports_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = RacketSports(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (151, 152)

    # Test 2: Check the sample fields, types and class names.
    first = test[0]
    assert set(first) == {"series", "label"}
    assert test.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert test.features["label"].names == [
        "badminton_clear",
        "badminton_smash",
        "squash_backhandboast",
        "squash_forehandboast",
    ]

    # Test 3: Check the shapes and class balance: every series is 30 steps of 6 channels.
    shape = (RacketSports.SEQUENCE_LENGTH, RacketSports.NUM_CHANNELS)
    assert shape == (30, 6)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (151, *shape)
    assert series["test"].shape == (152, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [43, 39, 34, 35]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [43, 40, 34, 35]

    # Test 4: Check the values parse correctly from the .ts file.
    np.testing.assert_allclose(
        series["test"][0, 0], [0.6661, -0.558194, -2.323993, -0.058594, -0.822982, -0.194426], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["test"][-1, -1], [2.484839, 9.713163, 2.377252, -0.370209, -0.207743, 0.207743], rtol=1e-6
    )
    assert (first["label"], test[-1]["label"]) == (1, 2)
    assert series["train"].sum() == pytest.approx(216.223868, abs=1e-3)
    assert series["test"].sum() == pytest.approx(2951.730445, abs=1e-3)
