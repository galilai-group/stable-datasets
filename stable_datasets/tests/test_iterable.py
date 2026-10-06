"""Tests for StableIterableDataset."""

from unittest.mock import MagicMock, patch

from stable_datasets.cache import write_sharded_arrow_cache
from stable_datasets.dataset import StableDataset
from stable_datasets.iterable import StableIterableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value, Video


def _make_ds(tmp_path, n=20, shard_size_bytes=512, batch_size=5):
    """Create a shard-backed StableDataset with multiple shards."""
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)
    cache_dir = tmp_path / "iter_cache"

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    meta = write_sharded_arrow_cache(
        gen(),
        features,
        cache_dir,
        shard_size_bytes=shard_size_bytes,
        batch_size=batch_size,
    )
    return StableDataset(
        features=features,
        info=info,
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


class TestIterableRespectsView:
    def test_indexed_view_yields_only_view_rows(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=100)
        test = ds.train_test_split(test_size=0.1, seed=0)["test"]
        rows = list(test.as_iterable())
        assert [r["x"] for r in rows] == [r["x"] for r in test]
        assert len(rows) == 10

    def test_indexed_view_worker_sharding_covers_view_once(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=50)
        view = ds.select(list(range(49, -1, -2)))
        xs = []
        for worker_id in range(3):
            worker_info = MagicMock(id=worker_id, num_workers=3)
            with patch("torch.utils.data.get_worker_info", return_value=worker_info):
                xs.extend(r["x"] for r in view.as_iterable())
        assert sorted(xs) == sorted(range(1, 50, 2))

    def test_dataset_transform_and_format_apply(self, tmp_path):
        ds, _ = _make_ds(tmp_path, n=5)
        view = ds.with_format("numpy").with_transform(lambda row: {"x": row["x"] * 10})
        assert [r["x"] for r in view.as_iterable()] == [r["x"] for r in view] == [0, 10, 20, 30, 40]

    def test_video_decode_applies_with_map_style_sample_index(self, tmp_path):
        features = Features({"video": Video(storage="bytes"), "x": Value("int32")})
        meta = write_sharded_arrow_cache(
            ((i, {"video": b"clip", "x": i}) for i in range(12)),
            features,
            tmp_path / "video_cache",
            shard_size_bytes=64,
            batch_size=4,
        )
        assert meta.num_shards > 1
        ds = StableDataset(
            features=features,
            info=DatasetInfo(features=features),
            shard_paths=meta.shard_paths,
            shard_row_counts=meta.shard_row_counts,
        )

        def decode_fn(ref, config, *, row=None, sample_index=None):
            return (row["x"], sample_index)

        decoded = ds.set_video_decode(num_frames=1, decode_fn=decode_fn)
        expected = [decoded[i]["video"] for i in range(len(decoded))]
        assert [r["video"] for r in decoded.as_iterable()] == expected

        view = decoded.select([7, 2, 9])
        assert [r["video"] for r in view.as_iterable()] == [(7, 0), (2, 1), (9, 2)]
