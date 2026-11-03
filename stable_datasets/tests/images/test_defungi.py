"""Integration coverage for the public DeFungi builder and grouped splits."""

import hashlib
from collections import Counter

from PIL import Image

from stable_datasets.images import DeFungi


def test_real_data(tmp_path):
    """Download all real patches and validate decoding, labels and split isolation."""
    cache = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    expected_counts = {
        "train": [2976, 1646, 635, 574, 537],
        "validation": [734, 377, 50, 109, 83],
        "test": [694, 311, 134, 135, 119],
    }
    class_names = ["H1", "H2", "H3", "H5", "H6"]
    image_ids = set()
    source_groups = {}
    linked_groups = {}
    hashes = {}
    for split, expected in expected_counts.items():
        dataset = DeFungi(split=split, **cache)
        assert len(dataset) == sum(expected)
        assert dataset.features["label"].names == class_names
        counts = Counter()
        sources = set()
        linked = set()
        for sample in dataset:
            assert set(sample) == {"image", "label", "image_id", "source_group", "split_group"}
            image = sample["image"]
            assert isinstance(image, Image.Image)
            image.load()
            assert image.size == (500, 500)
            assert image.mode == "RGB"
            label = sample["label"]
            assert isinstance(label, int)
            assert 0 <= label < 5
            for field in ("image_id", "source_group", "split_group"):
                assert isinstance(sample[field], str)
                assert sample[field]
            assert sample["image_id"].startswith(class_names[label] + "/")
            assert sample["source_group"].startswith(class_names[label] + "_")
            assert sample["split_group"].startswith(class_names[label] + "_")
            assert sample["image_id"] not in image_ids
            image_ids.add(sample["image_id"])
            counts[label] += 1
            sources.add(sample["source_group"])
            linked.add(sample["split_group"])
        assert [counts[label] for label in range(5)] == expected
        source_groups[split] = sources
        linked_groups[split] = linked
        # Inspect original JPEG bytes through the public dataset view, not the generator.
        hashes[split] = {hashlib.sha256(sample["image"]).digest() for sample in dataset.set_decode(False)}

    assert len(image_ids) == 9114
    assert [len(source_groups[split]) for split in expected_counts] == [312, 64, 71]
    assert [len(linked_groups[split]) for split in expected_counts] == [306, 64, 70]
    for groups in (source_groups, linked_groups, hashes):
        assert groups["train"].isdisjoint(groups["validation"])
        assert groups["train"].isdisjoint(groups["test"])
        assert groups["validation"].isdisjoint(groups["test"])
    assert sum(len(values) for values in hashes.values()) == 9055
    cached = DeFungi(**cache)
    assert {split: len(dataset) for split, dataset in cached.items()} == {
        split: sum(counts) for split, counts in expected_counts.items()
    }
    assert cached["test"][0]["image"].size == (500, 500)
