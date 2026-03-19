import pyarrow as pa

from stable_datasets.tabular import GitTables
from stable_datasets.tabular.base import TabularDataset, TabularTaskInfo


# Smallest zip in the Zenodo manifest (~11 MB) — used for all tests to
# avoid slow downloads in CI.
_TEST_ZIP = "beats_per_minute_tables_licensed.zip"


def test_zip_files_returns_manifest():
    """zip_files() returns a non-empty list of dicts with the expected keys."""
    zips = GitTables.zip_files()

    assert isinstance(zips, list)
    assert len(zips) > 0

    for entry in zips:
        assert "name" in entry
        assert "url" in entry
        assert "size" in entry
        assert entry["name"].endswith(".zip")


def test_list_tables_returns_parquet_names():
    """list_tables() returns a non-empty list of .parquet filenames."""
    tables = GitTables.list_tables(_TEST_ZIP)

    assert isinstance(tables, list)
    assert len(tables) > 0
    for name in tables:
        assert name.endswith(".parquet"), f"Expected .parquet, got {name!r}"


def test_list_tables_unknown_zip_raises():
    """list_tables() raises ValueError for an archive not in the manifest."""
    try:
        GitTables.list_tables("nonexistent_archive.zip")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "not found in Zenodo manifest" in str(e)


def test_load_returns_tabular_dataset():
    """load() returns a TabularDataset with correct types and non-empty data."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    assert isinstance(ds, TabularDataset)
    assert isinstance(ds.info, TabularTaskInfo)
    assert isinstance(ds.table, pa.Table)


def test_load_dataset_shape():
    """Loaded table has at least one row and one column."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    assert ds.info.n_rows > 0, f"Expected n_rows > 0, got {ds.info.n_rows}"
    assert ds.info.n_features > 0, f"Expected n_features > 0, got {ds.info.n_features}"
    assert len(ds) == ds.info.n_rows


def test_load_to_pandas():
    """to_pandas() returns a DataFrame consistent with the Arrow table."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])
    df = ds.to_pandas()

    assert len(df) == ds.info.n_rows
    assert len(df.columns) == ds.info.n_features


def test_load_task_info_fields():
    """TabularTaskInfo has correct default values for a GitTables table."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    assert ds.info.task_name == tables[0]
    assert ds.info.problem_type == "unknown"
    assert ds.info.target_col == ""
    assert ds.info.n_folds == 0
    assert ds.info.n_repeats == 0


def test_load_is_cached_on_second_call():
    """Calling load() twice returns equivalent datasets (second call hits cache)."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds1 = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])
    ds2 = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    assert ds1.info == ds2.info
    assert len(ds1) == len(ds2)
    assert ds1.table.schema == ds2.table.schema


def test_iter_tables_yields_tabular_datasets():
    """iter_tables() yields TabularDataset objects for a single archive."""
    seen = 0
    for ds in GitTables.iter_tables(zip_name=_TEST_ZIP, cache_tables=False):
        assert isinstance(ds, TabularDataset)
        assert isinstance(ds.table, pa.Table)
        assert ds.info.n_rows > 0
        assert ds.info.n_features > 0
        seen += 1
        if seen >= 3:  # only check first 3 tables — enough to validate, fast to run
            break

    assert seen == 3, f"Expected to iterate at least 3 tables, got {seen}"


def test_get_fold_raises_for_gittables():
    """GitTables datasets have no splits — get_fold() must raise ValueError."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    try:
        ds.get_fold(fold=0, repeat=0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "no pre-defined splits" in str(e)


def test_row_access():
    """Row access via __getitem__ returns a dict with the correct keys."""
    tables = GitTables.list_tables(_TEST_ZIP)
    ds = GitTables.load(zip_name=_TEST_ZIP, table_name=tables[0])

    row = ds[0]
    assert isinstance(row, dict)
    assert set(row.keys()) == set(ds.table.schema.names)


if __name__ == "__main__":
    test_zip_files_returns_manifest()
    test_list_tables_returns_parquet_names()
    test_list_tables_unknown_zip_raises()
    test_load_returns_tabular_dataset()
    test_load_dataset_shape()
    test_load_to_pandas()
    test_load_task_info_fields()
    test_load_is_cached_on_second_call()
    test_iter_tables_yields_tabular_datasets()
    test_get_fold_raises_for_gittables()
    test_row_access()
    print("All GitTables tests passed!")
