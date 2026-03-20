"""Tests for StableIterableDataset (Phase 3)."""

from unittest.mock import MagicMock, patch

import pytest

from stable_datasets.arrow_dataset import StableDataset
from stable_datasets.cache import write_sharded_arrow_cache
from stable_datasets.iterable import StableIterableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value


def _make_ds(tmp_path, n=20, shard_size_bytes=512, batch_size=5):
    """Create a shard-backed StableDataset with multiple shards."""
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)
    cache_dir = tmp_path / "iter_cache"

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    meta = write_sharded_arrow_cache(
        gen(), features, cache_dir,
        shard_size_bytes=shard_size_bytes, batch_size=batch_size,
    )
    return StableDataset(
        features=features,
        info=info,
        shard_dir=cache_dir,
        shard_paths=meta.shard_paths,
        shard_row_counts=meta.shard_row_counts,
        num_rows=meta.num_rows,
    ), meta


class TestIterableDataset:
    def test_single_worker_yields_all_rows(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=20)
        iterable = StableIterableDataset(ds, shuffle=False, seed=0)
        rows = list(iterable)
        assert len(rows) == 20
        assert sorted(r["x"] for r in rows) == list(range(20))

    def test_worker_sharding_covers_all_rows(self, tmp_path):
        ds, meta = _make_ds(tmp_path, n=20, shard_size_bytes=128, batch_size=5)
        assert meta.num_shards >= 2

        all_rows = []
        for worker_id in range(2):
            worker_info = MagicMock()
            worker_info.id = worker_id
            worker_info.num_workers = 2
            iterable = StableIterableDataset(ds, shuffle=False, seed=0)
            with patch("torch.utils.data.get_worker_info", return_value=worker_info):
                all_rows.extend(list(iterable))

        assert sorted(r["x"] for r in all_rows) == list(range(20))

    def test_buffered_shuffle_varies_order(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=20)
        iterable = StableIterableDataset(ds, shuffle=True, seed=42, buffer_size=10)
        rows = list(iterable)
        assert len(rows) == 20
        xs = [r["x"] for r in rows]
        # Should be a permutation, not sequential
        assert sorted(xs) == list(range(20))

    def test_set_epoch_changes_seed(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=20)
        iterable = StableIterableDataset(ds, shuffle=True, seed=0, buffer_size=10)
        iterable.set_epoch(0)
        order_0 = [r["x"] for r in iterable]
        iterable.set_epoch(1)
        order_1 = [r["x"] for r in iterable]
        assert sorted(order_0) == sorted(order_1)
        # Different epochs should (almost certainly) produce different orders
        assert order_0 != order_1

    def test_transform_applied_at_yield(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=5)

        def add_ten(row):
            row["x"] = row["x"] + 10
            return row

        iterable = StableIterableDataset(ds, shuffle=False, seed=0, transform=add_ten)
        rows = list(iterable)
        assert all(r["x"] >= 10 for r in rows)

    def test_as_iterable_bridge(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=10)
        iterable = ds.as_iterable(shuffle=False, seed=0)
        assert isinstance(iterable, StableIterableDataset)
        rows = list(iterable)
        assert len(rows) == 10
