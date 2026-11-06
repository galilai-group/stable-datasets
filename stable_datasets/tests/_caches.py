"""Small Lance caches for the tests."""

from stable_datasets.cache import open_cache, write_lance_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.schema import DatasetInfo


def cached_dataset(gen, features, cache_dir, *, rows_per_fragment=None, batch_size=1000, info=None):
    """Write the examples of ``gen`` as a Lance cache and open it (``rows_per_fragment``: several fragments)."""
    kw = {"max_rows_per_file": rows_per_fragment} if rows_per_fragment else {}
    write_lance_cache(gen, features, cache_dir, batch_size=batch_size, **kw)
    opened = open_cache(cache_dir, features)
    return StableDataset(
        features=features,
        info=info or DatasetInfo(features=features),
        backend=opened.backend,
        num_rows=opened.num_rows,
    )
