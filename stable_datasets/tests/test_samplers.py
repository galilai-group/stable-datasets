"""Tests for :mod:`stable_datasets.samplers`.

Covers:
* Coverage: every dataset index is yielded exactly once per epoch.
* Reproducibility: same seed + epoch -> same index order.
* Epoch variability: different epochs produce different orders.
* Shard structure: indices from shard k are all emitted before any
  index from shard k+1 in the emitted order (I/O locality).
* ``within_shard`` modes behave as documented.
* Pickle round-trip: sampler survives pickle for DataLoader worker
  boot under spawn context.
* Factory method ``StableDataset.make_sampler`` returns the right
  class.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from stable_datasets.dataset import StableDataset
from stable_datasets.arrow_backend import ArrowBackend
from stable_datasets.cache import write_lance_cache, write_sharded_arrow_cache
from stable_datasets.lance_backend import LanceBackend
from stable_datasets.samplers import ShardShuffleSampler
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value


def _make_multi_shard_arrow_ds(tmp_path, n: int = 40, batch_size: int = 10):
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    cache_dir = tmp_path / "arrow_cache"
    # Force multiple shards with a small shard_size_bytes
    meta = write_sharded_arrow_cache(
        gen(),
        features,
        cache_dir,
        batch_size=batch_size,
        shard_size_bytes=200,  # tiny -> forces multi-shard
    )
    backend = ArrowBackend(
        shard_paths=meta.shard_paths,
        shard_row_counts=meta.shard_row_counts,
        schema=features.to_arrow_schema(),
    )
    return (
        StableDataset(features=features, info=info, backend=backend, num_rows=meta.num_rows),
        meta,
    )


def _make_lance_ds(tmp_path, n: int = 40, batch_size: int = 10):
    features = Features({"x": Value("int32"), "label": ClassLabel(names=["a", "b"])})
    info = DatasetInfo(features=features)

    def gen():
        for i in range(n):
            yield i, {"x": i, "label": i % 2}

    lance_dir = tmp_path / "lance_cache"
    lance_meta = write_lance_cache(gen(), features, lance_dir, batch_size=batch_size)
    backend = LanceBackend(uri=lance_dir)
    return (
        StableDataset(
            features=features, info=info, backend=backend, num_rows=lance_meta.num_rows
        ),
        lance_meta,
    )


class TestShardShuffleSamplerArrow:
    def test_covers_every_index_once(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=40)
        sampler = ShardShuffleSampler(ds, seed=42, within_shard="random")
        emitted = list(sampler)
        assert len(emitted) == 40
        assert sorted(emitted) == list(range(40))
        assert len(sampler) == 40

    def test_reproducible_same_seed_same_epoch(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=40)
        s1 = ShardShuffleSampler(ds, seed=7)
        s2 = ShardShuffleSampler(ds, seed=7)
        assert list(s1) == list(s2)

    def test_different_seeds_differ(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=40)
        s1 = list(ShardShuffleSampler(ds, seed=1))
        s2 = list(ShardShuffleSampler(ds, seed=2))
        # Could match by coincidence but probability is astronomically low
        # with 40 indices.
        assert s1 != s2

    def test_set_epoch_changes_order(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=40)
        sampler = ShardShuffleSampler(ds, seed=42)
        e0 = list(sampler)
        sampler.set_epoch(1)
        e1 = list(sampler)
        assert e0 != e1

    def test_within_shard_random_permutes_inside_shard(self, tmp_path):
        ds, meta = _make_multi_shard_arrow_ds(tmp_path, n=40, batch_size=10)
        # With multiple shards, within_shard='random' should produce at
        # least one shard where the within-shard order isn't strictly
        # ascending.
        assert meta.num_shards > 1, "fixture should produce multiple shards"
        sampler = ShardShuffleSampler(ds, seed=0, within_shard="random")
        emitted = list(sampler)
        # Split the emitted stream back into per-shard chunks in emission
        # order and check at least one chunk is not a sorted range.
        ranges = [(0, 0)]
        total = 0
        for c in meta.shard_row_counts:
            ranges.append((total, total + c))
            total += c
        # We don't know which shard came first, but we know contiguous
        # runs in the emitted stream correspond to shards.
        # Group consecutive emitted indices by shard membership.
        def shard_of(i):
            for k, (s, e) in enumerate(ranges[1:]):
                if s <= i < e:
                    return k
            raise AssertionError

        groups: list[list[int]] = [[]]
        current_shard = shard_of(emitted[0])
        for idx in emitted:
            if shard_of(idx) == current_shard:
                groups[-1].append(idx)
            else:
                current_shard = shard_of(idx)
                groups.append([idx])
        any_non_sorted = any(g != sorted(g) for g in groups)
        assert any_non_sorted, "within_shard='random' should permute at least one shard"

    def test_within_shard_sequential_keeps_shard_ordering(self, tmp_path):
        ds, meta = _make_multi_shard_arrow_ds(tmp_path, n=40, batch_size=10)
        sampler = ShardShuffleSampler(ds, seed=0, within_shard="sequential")
        emitted = list(sampler)

        # Group emitted into per-shard chunks in emission order; each
        # chunk must be a contiguous ascending sequence.
        ranges = [(0, 0)]
        total = 0
        for c in meta.shard_row_counts:
            ranges.append((total, total + c))
            total += c

        def shard_of(i):
            for k, (s, e) in enumerate(ranges[1:]):
                if s <= i < e:
                    return k
            raise AssertionError

        groups: list[list[int]] = [[]]
        current_shard = shard_of(emitted[0])
        for idx in emitted:
            if shard_of(idx) == current_shard:
                groups[-1].append(idx)
            else:
                current_shard = shard_of(idx)
                groups.append([idx])
        for g in groups:
            assert g == sorted(g), (
                f"within_shard='sequential' expected ascending within-shard but got {g}"
            )

    def test_shard_block_structure(self, tmp_path):
        """All indices from any single shard are emitted contiguously."""
        ds, meta = _make_multi_shard_arrow_ds(tmp_path, n=40, batch_size=10)
        assert meta.num_shards > 1
        sampler = ShardShuffleSampler(ds, seed=3)
        emitted = list(sampler)

        ranges = [(0, 0)]
        total = 0
        for c in meta.shard_row_counts:
            ranges.append((total, total + c))
            total += c

        def shard_of(i):
            for k, (s, e) in enumerate(ranges[1:]):
                if s <= i < e:
                    return k
            raise AssertionError

        seen_shards: set[int] = set()
        current_shard = shard_of(emitted[0])
        seen_shards.add(current_shard)
        for idx in emitted[1:]:
            s = shard_of(idx)
            if s != current_shard:
                # Transition to a new shard -- we must not revisit.
                assert s not in seen_shards, f"shard {s} interleaved with {current_shard}"
                seen_shards.add(s)
                current_shard = s

    def test_pickle_roundtrip(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=40)
        sampler = ShardShuffleSampler(ds, seed=123)
        loaded = pickle.loads(pickle.dumps(sampler))
        # Order must be identical after pickle (sampler holds only
        # plain state, no live backend handle).
        assert list(sampler) == list(loaded)

    def test_invalid_within_shard_raises(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=10)
        with pytest.raises(ValueError, match="within_shard"):
            ShardShuffleSampler(ds, within_shard="neither")


class TestShardShuffleSamplerLance:
    def test_covers_every_index_once(self, tmp_path):
        ds, _ = _make_lance_ds(tmp_path, n=40)
        sampler = ShardShuffleSampler(ds, seed=42)
        emitted = list(sampler)
        assert sorted(emitted) == list(range(40))

    def test_pickle_roundtrip(self, tmp_path):
        ds, _ = _make_lance_ds(tmp_path, n=40)
        sampler = ShardShuffleSampler(ds, seed=7)
        loaded = pickle.loads(pickle.dumps(sampler))
        assert list(sampler) == list(loaded)


class TestFactoryMethod:
    def test_make_sampler_returns_shard_shuffle_sampler(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=20)
        s = ds.make_sampler("shard_shuffle", seed=42)
        assert isinstance(s, ShardShuffleSampler)
        assert sorted(list(s)) == list(range(20))

    def test_make_sampler_forwards_kwargs(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=20)
        s_random = ds.make_sampler("shard_shuffle", seed=0, within_shard="random")
        s_sequential = ds.make_sampler("shard_shuffle", seed=0, within_shard="sequential")
        # Both cover indices but with potentially different orderings.
        assert sorted(list(s_random)) == list(range(20))
        assert sorted(list(s_sequential)) == list(range(20))

    def test_make_sampler_unknown_kind_raises(self, tmp_path):
        ds, _ = _make_multi_shard_arrow_ds(tmp_path, n=10)
        with pytest.raises(ValueError, match="Unknown sampler kind"):
            ds.make_sampler("nonexistent")
