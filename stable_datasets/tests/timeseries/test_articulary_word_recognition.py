import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import ArticularyWordRecognition


def test_articulary_word_recognition_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = ArticularyWordRecognition(
        split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path)
    )
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (275, 300)

    # Test 2: Check the sample fields, types and class names.
    first = train[0]
    assert set(first) == {"series", "label"}
    assert train.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert train.features["label"].names == [
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
    ]

    # Test 3: Check the shapes and class balance: every series is 144 steps of 9 channels.
    shape = (ArticularyWordRecognition.SEQUENCE_LENGTH, ArticularyWordRecognition.NUM_CHANNELS)
    assert shape == (144, 9)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (275, *shape)
    assert series["test"].shape == (300, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
        11,
    ]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
    ]

    # Test 4: Check the values and labels parse correctly from the .ts file (first six channels).
    np.testing.assert_allclose(
        series["train"][0, 0, :6], [0.87159, 0.7453, 1.1533, 0.38622, 1.5156, -0.7155], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["train"][-1, -1, :6], [-0.68074, 0.10613, 0.023265, -0.41724, -0.38593, 0.55649], rtol=1e-6
    )
    assert (first["label"], train[-1]["label"]) == (0, 24)
    assert np.abs(series["train"]).sum() == pytest.approx(299803.527709, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(326320.685457, rel=1e-9)
