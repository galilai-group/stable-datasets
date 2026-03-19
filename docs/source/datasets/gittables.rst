GitTables
=========

Overview
--------

GitTables is a large-scale corpus of ~1.7 million relational tables extracted from CSV files hosted on GitHub. It is designed for pretraining, semantic type detection, table retrieval, and related tabular learning tasks. Each table is accompanied by rich metadata including semantic column type annotations derived from DBpedia and Schema.org ontologies.

Tables span a wide variety of domains and sizes, making GitTables a diverse resource for representation learning on tabular data.

- **Total tables**: ~1.7 million
- **Column annotations**: semantic types from DBpedia and Schema.org
- **File format**: Parquet, partitioned by topic
- **Source**: CSV files mined from public GitHub repositories

Data Structure
--------------

``GitTables.load()`` returns a ``TabularDataset`` instance wrapping a PyArrow table with all columns, plus associated metadata and semantic column type annotations.

**GitTablesTaskInfo metadata**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Type
     - Description
   * - ``table_id``
     - str
     - Unique identifier for the table (derived from source path)
   * - ``repo``
     - str
     - GitHub repository the CSV was extracted from
   * - ``file_path``
     - str
     - Path of the original CSV within the repository
   * - ``topic``
     - str
     - High-level topic partition (e.g. ``"sports"``, ``"finance"``)
   * - ``n_rows``
     - int
     - Number of rows in the table
   * - ``n_cols``
     - int
     - Number of columns in the table
   * - ``col_types``
     - dict
     - Mapping of column name to inferred semantic type (DBpedia/Schema.org)

**TabularDataset properties**

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Property / Method
     - Returns
     - Description
   * - ``ds.table``
     - ``pa.Table``
     - Full Arrow table — all columns
   * - ``ds.X``
     - ``pa.Table``
     - All feature columns (no designated target in this corpus)
   * - ``ds.to_pandas()``
     - ``pd.DataFrame``
     - Full table as a pandas DataFrame
   * - ``ds[i]``
     - dict
     - Single row as a plain Python dict
   * - ``ds[i:j]``
     - ``TabularDataset``
     - Row slice as a new in-memory dataset
   * - ``ds.col_types``
     - dict
     - Semantic column type annotations per column

Cache Layout
------------

Tables are cached under ``~/.stable-datasets/processed/gittables/`` (respects the ``STABLE_DATASETS_CACHE_DIR`` environment variable)::

    topic_<topic>/
    ├── data.parquet     Parquet file — full table, all columns
    └── metadata.json    GitTablesTaskInfo fields

Usage Example
-------------

**List all topic partitions**

.. code-block:: python

    from stable_datasets.tabular import GitTables

    # Fetches the topic list on first call, then caches in memory
    topics = GitTables.topics()
    print(f"GitTables contains {len(topics)} topic partitions")

**Load a single table by ID**

.. code-block:: python

    from stable_datasets.tabular import GitTables

    # Downloads and caches on first use; loads from cache on subsequent calls
    ds = GitTables.load(table_id="abc123")

    print(ds)
    # TabularDataset(table_id='abc123', repo='owner/repo', n_rows=500, n_cols=8)

    print(ds.repo)       # "owner/repo"
    print(ds.topic)      # "finance"
    print(ds.col_types)  # {"date": "xsd:date", "price": "dbo:Money", ...}

**Load tables by topic**

.. code-block:: python

    # Load all tables belonging to a specific topic partition
    for ds in GitTables.iter_topic(topic="sports"):
        df = ds.to_pandas()
        print(ds.table_id, df.shape)

**Access columns and semantic types**

.. code-block:: python

    from stable_datasets.tabular import GitTables

    ds = GitTables.load(table_id="abc123")

    # PyArrow table
    X = ds.X

    # Pandas DataFrame
    df = ds.to_pandas()
    print(df.shape)

    # Semantic column annotations
    for col, sem_type in ds.col_types.items():
        print(f"{col}: {sem_type}")

    # Row-level access
    row = ds[0]   # plain Python dict
    print(row.keys())

**Iterate the full corpus**

.. code-block:: python

    from stable_datasets.tabular import GitTables

    for ds in GitTables.iter_tables():
        df = ds.to_pandas()
        # ... process table
        print(ds.table_id, ds.n_rows, ds.n_cols)

**Subset by topic list**

.. code-block:: python

    from stable_datasets.tabular import GitTables

    # Iterate only over specific topic partitions
    for ds in GitTables.iter_tables(topics=["finance", "sports", "health"]):
        print(ds.table_id, ds.topic)

References
----------

- Paper: https://arxiv.org/abs/2106.07258
- Dataset: https://zenodo.org/record/6517052
- Code: https://github.com/madelonhulsebos/gittables

Citation
--------

.. code-block:: bibtex

    @article{hulsebos2023gittables,
    title={Gittables: A large-scale corpus of relational tables},
    author={Hulsebos, Madelon and Demiralp, {\c{C}}agatay and Groth, Paul},
    journal={Proceedings of the ACM on Management of Data},
    volume={1},
    number={1},
    pages={1--17},
    year={2023},
    publisher={ACM New York, NY, USA}
    }