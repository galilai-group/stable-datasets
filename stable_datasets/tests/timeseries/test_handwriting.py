import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import Handwriting


def test_handwriting_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = Handwriting(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (150, 850)

    # Test 2: Check the sample fields, types and class names.
    first = test[0]
    assert set(first) == {"series", "label"}
    assert test.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert test.features["label"].names == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
    ]

    # Test 3: Check the shapes and class balance: every series is 152 steps of 3 channels.
    shape = (Handwriting.SEQUENCE_LENGTH, Handwriting.NUM_CHANNELS)
    assert shape == (152, 3)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (150, *shape)
    assert series["test"].shape == (850, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [
        8,
        6,
        9,
        4,
        5,
        9,
        6,
        2,
        3,
        7,
        2,
        5,
        7,
        4,
        8,
        5,
        6,
        6,
        3,
        5,
        5,
        10,
        4,
        4,
        6,
        11,
    ]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [
        36,
        34,
        29,
        43,
        34,
        30,
        29,
        32,
        27,
        26,
        25,
        37,
        40,
        35,
        29,
        33,
        32,
        31,
        33,
        31,
        34,
        43,
        32,
        37,
        26,
        32,
    ]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(series["test"][0, 0], [1.197, -2.3502, -1.3336], rtol=1e-6)
    np.testing.assert_allclose(series["test"][-1, -1], [-0.099419, -0.11287, -0.093505], rtol=1e-6)
    assert (first["label"], test[-1]["label"]) == (14, 21)
    assert np.abs(series["train"]).sum() == pytest.approx(38147.461063, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(219712.929066, rel=1e-9)
