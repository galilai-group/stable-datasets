import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import Cricket


def test_cricket_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = Cricket(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (108, 72)

    # Test 2: Check the sample fields, types and class names.
    first = train[0]
    assert set(first) == {"series", "label"}
    assert train.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert train.features["label"].names == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    # Test 3: Check the shapes and class balance: every series is 1197 steps of 6 channels.
    shape = (Cricket.SEQUENCE_LENGTH, Cricket.NUM_CHANNELS)
    assert shape == (1197, 6)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (108, *shape)
    assert series["test"].shape == (72, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(
        series["train"][0, 0], [-0.97211, -0.23077, -0.46863, -0.57729, 0.29993, -0.44158], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["train"][-1, -1], [-0.66982, -0.62841, -0.69293, -0.68377, 0.84363, -0.77377], rtol=1e-6
    )
    assert (first["label"], train[-1]["label"]) == (0, 11)
    assert np.abs(series["train"]).sum() == pytest.approx(613041.842532, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(404907.290104, rel=1e-9)
