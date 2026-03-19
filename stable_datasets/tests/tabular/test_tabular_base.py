import pyarrow as pa
import pytest

from stable_datasets.tabular.base import TabularDataset, TabularTaskInfo


def _make_info(**overrides) -> TabularTaskInfo:
    defaults = dict(
        task_id=0,
        task_name="test_task",
        problem_type="binary",
        target_col="target",
        n_rows=10,
        n_features=2,
        n_folds=0,
        n_repeats=0,
    )
    defaults.update(overrides)
    return TabularTaskInfo(**defaults)


def _make_dataset(n_rows: int = 10, with_splits: bool = False) -> TabularDataset:
    table = pa.table(
        {
            "feature_a": list(range(n_rows)),
            "feature_b": [float(i) * 0.5 for i in range(n_rows)],
            "target": [i % 2 for i in range(n_rows)],
        }
    )
    info = _make_info(n_rows=n_rows)

    if not with_splits:
        return TabularDataset(table, info)

    # Two folds, one repeat
    train_0 = list(range(0, 8))
    test_0 = list(range(8, 10))
    train_1 = list(range(2, 10))
    test_1 = list(range(0, 2))
    splits = {0: {0: (train_0, test_0), 1: (train_1, test_1)}}
    info_with_splits = _make_info(n_rows=n_rows, n_folds=2, n_repeats=1)
    return TabularDataset(table, info_with_splits, splits=splits)


def test_info_returns_task_info():
    ds = _make_dataset()
    assert isinstance(ds.info, TabularTaskInfo)


def test_metadata_properties():
    ds = _make_dataset()
    assert ds.task_id == 0
    assert ds.task_name == "test_task"
    assert ds.problem_type == "binary"
    assert ds.target_col == "target"
    assert ds.n_folds == 0
    assert ds.n_repeats == 0


def test_repr_contains_task_name():
    ds = _make_dataset()
    assert "test_task" in repr(ds)


def test_len():
    ds = _make_dataset(n_rows=10)
    assert len(ds) == 10


def test_table_is_arrow():
    ds = _make_dataset()
    assert isinstance(ds.table, pa.Table)


def test_to_pandas_shape():
    ds = _make_dataset(n_rows=10)
    df = ds.to_pandas()
    assert len(df) == 10
    assert set(df.columns) == {"feature_a", "feature_b", "target"}


def test_X_excludes_target():
    ds = _make_dataset()
    assert "target" not in ds.X.column_names
    assert "feature_a" in ds.X.column_names
    assert "feature_b" in ds.X.column_names


def test_X_row_count_matches():
    ds = _make_dataset(n_rows=10)
    assert ds.X.num_rows == 10


def test_y_is_chunked_array():
    ds = _make_dataset()
    assert isinstance(ds.y, pa.ChunkedArray)


def test_y_length_matches():
    ds = _make_dataset(n_rows=10)
    assert len(ds.y) == 10


def test_y_values_correct():
    ds = _make_dataset(n_rows=6)
    values = ds.y.to_pylist()
    assert values == [0, 1, 0, 1, 0, 1]


def test_getitem_int_returns_dict():
    ds = _make_dataset()
    row = ds[0]
    assert isinstance(row, dict)
    assert set(row.keys()) == {"feature_a", "feature_b", "target"}


def test_getitem_int_correct_values():
    ds = _make_dataset()
    row = ds[3]
    assert row["feature_a"] == 3
    assert row["feature_b"] == pytest.approx(1.5)
    assert row["target"] == 3 % 2


def test_getitem_negative_index():
    ds = _make_dataset(n_rows=10)
    assert ds[-1]["feature_a"] == ds[9]["feature_a"]


def test_getitem_out_of_range_raises():
    ds = _make_dataset(n_rows=10)
    with pytest.raises(IndexError):
        _ = ds[10]


def test_getitem_slice_returns_dataset():
    ds = _make_dataset(n_rows=10)
    subset = ds[2:5]
    assert isinstance(subset, TabularDataset)
    assert len(subset) == 3


def test_getitem_slice_correct_rows():
    ds = _make_dataset(n_rows=10)
    subset = ds[0:3]
    values = subset.table.column("feature_a").to_pylist()
    assert values == [0, 1, 2]


def test_getitem_wrong_type_raises():
    ds = _make_dataset()
    with pytest.raises(TypeError):
        _ = ds["bad_index"]


def test_iter_yields_dicts():
    ds = _make_dataset(n_rows=5)
    rows = list(ds)
    assert len(rows) == 5
    for row in rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == {"feature_a", "feature_b", "target"}


def test_iter_correct_values():
    ds = _make_dataset(n_rows=3)
    rows = list(ds)
    assert rows[0]["feature_a"] == 0
    assert rows[1]["feature_a"] == 1
    assert rows[2]["feature_a"] == 2


def test_get_fold_raises_when_no_splits():
    ds = _make_dataset(with_splits=False)
    with pytest.raises(ValueError, match="no pre-defined splits"):
        ds.get_fold(fold=0, repeat=0)


def test_iter_folds_raises_when_no_splits():
    ds = _make_dataset(with_splits=False)
    with pytest.raises(ValueError, match="no pre-defined splits"):
        list(ds.iter_folds())


def test_get_fold_returns_two_datasets():
    ds = _make_dataset(with_splits=True)
    train, test = ds.get_fold(fold=0, repeat=0)
    assert isinstance(train, TabularDataset)
    assert isinstance(test, TabularDataset)


def test_get_fold_sizes():
    ds = _make_dataset(n_rows=10, with_splits=True)
    train, test = ds.get_fold(fold=0, repeat=0)
    assert len(train) == 8
    assert len(test) == 2


def test_get_fold_no_overlap():
    ds = _make_dataset(n_rows=10, with_splits=True)
    train, test = ds.get_fold(fold=0, repeat=0)
    train_vals = set(train.table.column("feature_a").to_pylist())
    test_vals = set(test.table.column("feature_a").to_pylist())
    assert train_vals.isdisjoint(test_vals)


def test_get_fold_covers_all_rows():
    ds = _make_dataset(n_rows=10, with_splits=True)
    train, test = ds.get_fold(fold=0, repeat=0)
    all_vals = set(train.table.column("feature_a").to_pylist()) | set(test.table.column("feature_a").to_pylist())
    assert all_vals == set(range(10))


def test_get_fold_invalid_raises():
    ds = _make_dataset(with_splits=True)
    with pytest.raises(ValueError):
        ds.get_fold(fold=99, repeat=0)


def test_iter_folds_yields_correct_count():
    ds = _make_dataset(with_splits=True)
    folds = list(ds.iter_folds())
    # 2 folds × 1 repeat = 2 entries
    assert len(folds) == 2


def test_iter_folds_yields_correct_types():
    ds = _make_dataset(with_splits=True)
    for fold, repeat, train, test in ds.iter_folds():
        assert isinstance(fold, int)
        assert isinstance(repeat, int)
        assert isinstance(train, TabularDataset)
        assert isinstance(test, TabularDataset)


def test_fold_subsets_have_no_further_splits():
    """Subsets returned by get_fold cannot be split further."""
    ds = _make_dataset(with_splits=True)
    train, _ = ds.get_fold(fold=0, repeat=0)
    with pytest.raises(ValueError, match="no pre-defined splits"):
        train.get_fold(fold=0, repeat=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
