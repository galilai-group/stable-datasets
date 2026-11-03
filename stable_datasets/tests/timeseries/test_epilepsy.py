import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import Epilepsy


def test_epilepsy_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = Epilepsy(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (137, 138)

    # Test 2: Check the sample fields, types and class names.
    first = train[0]
    assert set(first) == {"series", "label"}
    assert train.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert train.features["label"].names == ["epilepsy", "running", "sawing", "walking"]

    # Test 3: Check the shapes and class balance: every series is 206 steps of 3 channels.
    shape = (Epilepsy.SEQUENCE_LENGTH, Epilepsy.NUM_CHANNELS)
    assert shape == (206, 3)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (137, *shape)
    assert series["test"].shape == (138, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [34, 36, 30, 37]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [34, 37, 30, 37]

    # Test 4: Check the values parse correctly from the .ts file.
    np.testing.assert_allclose(series["train"][0, 0], [0.35, 0.22, 0.55], rtol=1e-6)
    np.testing.assert_allclose(series["train"][-1, -1], [0.59, -0.99, -0.06], rtol=1e-6)
    assert (first["label"], train[-1]["label"]) == (0, 2)
    assert series["train"].sum() == pytest.approx(-9106.87001, abs=1e-3)
    assert series["test"].sum() == pytest.approx(-15162.600019, abs=1e-3)
