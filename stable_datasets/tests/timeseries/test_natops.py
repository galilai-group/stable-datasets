import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import NATOPS


def test_natops_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = NATOPS(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
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
        "i_have_command",
        "all_clear",
        "not_clear",
        "spread_wings",
        "fold_wings",
        "lock_wings",
    ]

    # Test 3: Check the shapes and class balance: every series is 51 steps of 24 channels.
    shape = (NATOPS.SEQUENCE_LENGTH, NATOPS.NUM_CHANNELS)
    assert shape == (51, 24)
    series = {name: np.asarray(split.table.column("series").to_pylist(), np.float64) for name, split in ds.items()}
    assert series["train"].shape == (180, *shape)
    assert series["test"].shape == (180, *shape)
    assert all(np.isfinite(x).all() for x in series.values())
    assert np.bincount(train.table.column("label").to_numpy()).tolist() == [30, 30, 30, 30, 30, 30]
    assert np.bincount(test.table.column("label").to_numpy()).tolist() == [30, 30, 30, 30, 30, 30]

    # Test 4: Check the values and labels parse correctly from the .ts file (first six channels).
    np.testing.assert_allclose(
        series["train"][0, 0, :6], [-0.372758, -1.821679, -0.846321, 0.465208, -2.015072, -0.839242], rtol=1e-6
    )
    np.testing.assert_allclose(
        series["train"][-1, -1, :6], [-0.536957, -1.68946, -0.817575, 0.367535, -1.877205, -0.77226], rtol=1e-6
    )
    assert (first["label"], train[-1]["label"]) == (3, 3)
    assert np.abs(series["train"]).sum() == pytest.approx(175603.592946, rel=1e-9)
    assert np.abs(series["test"]).sum() == pytest.approx(176082.903899, rel=1e-9)
