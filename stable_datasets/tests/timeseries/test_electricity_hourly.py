import numpy as np
import pyarrow as pa

from stable_datasets.timeseries import ElectricityHourly


def test_electricity_hourly_dataset(tmp_path):
    # Test 1: Check the splits and their sizes.
    ds = ElectricityHourly(split=None, download_dir=str(tmp_path / "downloads"), processed_cache_dir=str(tmp_path))
    assert set(ds) == {"train", "test"}
    train, test = ds["train"], ds["test"]
    assert len(train) == len(test) == 321

    # Test 2: Check the sample fields and types.
    first = test[0]
    assert set(first) == {"series", "item_id", "start"}
    assert test.table.schema.field("series").type == pa.list_(pa.float32())
    assert isinstance(first["series"], list)
    assert isinstance(first["item_id"], str)
    assert isinstance(first["start"], str)
    assert (first["item_id"], first["start"]) == ("T1", "2012-01-01T00:00:01")
    assert test[-1]["item_id"] == "T321"

    # Test 3: Check the forecasting settings and series lengths: test is the full series, train drops the horizon.
    horizon = ElectricityHourly.PREDICTION_LENGTH
    assert (ElectricityHourly.FREQ, horizon, ElectricityHourly.SEASONALITY) == ("h", 24, 24)
    test_series = [np.asarray(s, np.float64) for s in test.table.column("series").to_pylist()]
    train_series = [np.asarray(s, np.float64) for s in train.table.column("series").to_pylist()]
    assert {len(s) for s in test_series} == {26304}
    assert all(np.array_equal(a, b[:-horizon]) for a, b in zip(train_series, test_series))

    # Test 4: Check the values parse correctly from the .tsf file.
    np.testing.assert_array_equal(test_series[0][:3], [14, 18, 21])
    np.testing.assert_array_equal(test_series[-1][-3:], [2650, 2719, 2640])
    assert sum(s.sum() for s in test_series) == 21_436_500_541
