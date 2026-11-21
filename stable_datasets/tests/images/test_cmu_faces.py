"""Integration coverage for the public CMU Face Images builder."""

from PIL import Image

from stable_datasets.images import CMUFaces


def test_real_data(tmp_path):
    """Download real images, validate every split and reload the public cache."""
    cache = {"download_dir": tmp_path / "downloads", "processed_cache_dir": tmp_path / "processed"}
    expected_sizes = {"train": 378, "validation": 121, "test": 125}
    subjects = {}
    annotations = set()
    for split, expected_size in expected_sizes.items():
        dataset = CMUFaces(split=split, **cache)
        assert len(dataset) == expected_size
        assert dataset.features["label"].names == ["straight", "left", "right", "up"]
        split_subjects = set()
        labels = set()
        for sample in dataset:
            assert set(sample) == {"image", "label", "subject", "expression", "eyes"}
            image = sample["image"]
            assert isinstance(image, Image.Image)
            image.load()
            assert image.size == (32, 30)
            assert image.mode == "L"
            for field, limit in (("label", 4), ("expression", 4), ("eyes", 2)):
                assert isinstance(sample[field], int)
                assert 0 <= sample[field] < limit
            assert isinstance(sample["subject"], str)
            key = (sample["subject"], sample["label"], sample["expression"], sample["eyes"])
            assert key not in annotations
            annotations.add(key)
            split_subjects.add(sample["subject"])
            labels.add(sample["label"])
        assert labels == {0, 1, 2, 3}
        subjects[split] = split_subjects

    assert len(annotations) == 624
    assert [len(subjects[split]) for split in expected_sizes] == [12, 4, 4]
    assert subjects["train"].isdisjoint(subjects["validation"])
    assert subjects["train"].isdisjoint(subjects["test"])
    assert subjects["validation"].isdisjoint(subjects["test"])
    cached = CMUFaces(**cache)
    assert {split: len(dataset) for split, dataset in cached.items()} == expected_sizes
    assert cached["train"][0]["image"].size == (32, 30)
