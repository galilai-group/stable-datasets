import numpy as np
import pytest
from PIL import Image
from scipy.io import loadmat

from stable_datasets.images import Flowers17


@pytest.mark.parametrize("partition", [1, 2, 3])
def test_flowers17_dataset(partition, tmp_path, monkeypatch):
    split_specs = {"train": ("trn", 680), "validation": ("val", 340), "test": ("tst", 340)}
    generated_ids = {}
    expected_ids = {}
    generate_examples = Flowers17._generate_examples

    # Observe real generated IDs without changing the examples sent to the cache.
    def record_examples(builder, path_map, split):
        prefix, _ = split_specs[split]
        expected_ids[split] = set(loadmat(path_map["splits"])[f"{prefix}{partition}"].ravel())
        generated_ids[split] = set()
        for image_id, example in generate_examples(builder, path_map, split):
            generated_ids[split].add(image_id)
            yield image_id, example

    monkeypatch.setattr(Flowers17, "_generate_examples", record_examples)
    # Fresh processed caches ensure the generator runs; raw downloads can be reused.
    flowers = Flowers17(config_name=f"partition{partition}", processed_cache_dir=tmp_path / "processed")
    assert set(flowers) == set(split_specs)

    for split, (_, expected_count) in split_specs.items():
        dataset = flowers[split]
        assert len(dataset) == expected_count
        assert generated_ids[split] == expected_ids[split]

        sample = dataset[0]
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        image_np = np.array(sample["image"])
        assert image_np.ndim == 3
        assert image_np.shape[2] == 3
        assert image_np.dtype == np.uint8
        assert isinstance(sample["label"], int)
        assert 0 <= sample["label"] < 17

    train, validation, test = (generated_ids[split] for split in split_specs)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == set(range(1, 1361))
