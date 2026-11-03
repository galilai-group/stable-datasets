import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import BasicMotions


def test_basic_motions_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = BasicMotions(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (40, 40)

    # Test 2: Check the sample fields, types and class names.
    first = train[0]
    assert set(first) == {"series", "label"}
    assert train.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert train.features["label"].names == ["badminton", "running", "standing", "walking"]

    # Test 3: Check the shapes and class balance: every series is 100 steps of 6 channels.
    shape = (BasicMotions.SEQUENCE_LENGTH, BasicMotions.NUM_CHANNELS)
    assert shape == (100, 6)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (40, *shape)
    assert series["test"].shape == (40, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [10, 10, 10, 10]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [10, 10, 10, 10]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(
        series["train"][0, 0], [0.079106, 0.394032, 0.551444, 0.351565, 0.02397, 0.633883], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["train"][-1, -1], [3.16927, 0.826934, -0.362036, -0.298298, 0.250357, 0.428803], rtol=1e-6
    )
    assert (first["label"], train[-1]["label"]) == (2, 0)
    assert np.abs(series["train"]).sum() == pytest.approx(61841.765635, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(58233.253639, rel=1e-9)
