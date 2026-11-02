import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import Libras


def test_libras_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = Libras(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert (len(train), len(test)) == (180, 180)

    # Test 2: Check the sample fields, types and class names.
    first = train[0]
    assert set(first) == {"series", "label"}
    assert train.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert isinstance(first["series"], list)
    assert isinstance(first["label"], int)
    assert train.features["label"].names == [
        "curved_swing",
        "horizontal_swing",
        "vertical_swing",
        "anti_clockwise_arc",
        "clockwise_arc",
        "circle",
        "horizontal_straight_line",
        "vertical_straight_line",
        "horizontal_zigzag",
        "vertical_zigzag",
        "horizontal_wavy",
        "vertical_wavy",
        "face_up_curve",
        "face_down_curve",
        "tremble",
    ]

    # Test 3: Check the shapes and class balance: every series is 45 steps of 2 channels.
    shape = (Libras.SEQUENCE_LENGTH, Libras.NUM_CHANNELS)
    assert shape == (45, 2)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (180, *shape)
    assert series["test"].shape == (180, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [
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
    ]

    # Test 4: Check the values and labels parse correctly from the .ts file.
    np.testing.assert_allclose(series["train"][0, 0], [0.67892, 0.27315], rtol=1e-6)
    np.testing.assert_allclose(series["train"][-1, -1], [0.52031, 0.49306], rtol=1e-6)
    assert (first["label"], train[-1]["label"]) == (0, 14)
    assert np.abs(series["train"]).sum() == pytest.approx(8331.419185, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(8250.26288, rel=1e-9)
