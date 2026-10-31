import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import HandMovementDirection


def test_hand_movement_direction_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = HandMovementDirection(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (160, 74)

    # Test 2: Check the sample fields, types and class names.
    first = test[0]
    assert set(first) == {"series", "label"}
    assert test.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert test.features["label"].names == ["backward", "forward", "left", "right"]

    # Test 3: Check the shapes and class balance: every series is 400 steps of 10 channels.
    shape = (HandMovementDirection.SEQUENCE_LENGTH, HandMovementDirection.NUM_CHANNELS)
    assert shape == (400, 10)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (160, *shape)
    assert series["test"].shape == (74, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [40, 40, 40, 40]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [15, 30, 15, 14]

    # Test 4: Check the values and labels parse correctly from the .ts file (first six channels).
    np.testing.assert_allclose(
        series["test"][0, 0, :6], [111.129743, 80.533307, -152.989498, 158.920473, -65.494423, 30.317042], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["test"][-1, -1, :6], [-74.928263, -38.381598, 83.923628, -26.018941, -65.92835, -104.720438], rtol=1e-6
    )
    assert (first["label"], test[-1]["label"]) == (3, 0)
    assert np.abs(series["train"]).sum() == pytest.approx(85553399.900532, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(39369878.437115, rel=1e-9)
