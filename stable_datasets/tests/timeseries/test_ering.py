import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import ERing


def test_ering_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = ERing(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (30, 270)

    # Test 2: Check the sample fields, types and class names.
    first = test[0]
    assert set(first) == {"series", "label"}
    assert test.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert test.features["label"].names == ["1", "2", "3", "4", "5", "6"]

    # Test 3: Check the shapes and class balance: every series is 65 steps of 4 channels.
    shape = (ERing.SEQUENCE_LENGTH, ERing.NUM_CHANNELS)
    assert shape == (65, 4)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (30, *shape)
    assert series["test"].shape == (270, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [5, 5, 5, 5, 5, 5]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [45, 45, 45, 45, 45, 45]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(series["test"][0, 0], [0.00604275, 0.75090308, 1.25534393, 0.51499435], rtol=1e-6)
    np.testing.assert_allclose(series["test"][-1, -1], [0.24761949, 0.94448082, -0.91316669, -0.66816691], rtol=1e-6)
    assert (first["label"], test[-1]["label"]) == (1, 4)
    assert np.abs(series["train"]).sum() == pytest.approx(6706.771925, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(60107.716547, rel=1e-9)
