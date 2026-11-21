from collections import Counter

from PIL import Image

from stable_datasets.images import COIL20


def test_coil20_dataset():
    dataset = COIL20(split="train")

    assert len(dataset) == 1440

    sample = dataset[0]
    assert set(sample) == {"image", "label", "view_id"}
    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].size == (128, 128)
    assert sample["image"].mode == "L"

    label_counts = Counter()
    object_views = set()

    for index in range(len(dataset)):
        item = dataset[index]
        label = item["label"]
        view_id = item["view_id"]

        assert 0 <= label < 20
        assert isinstance(view_id, int)
        label_counts[label] += 1
        object_views.add((label, view_id))

    assert len(label_counts) == 20
    assert all(count == 72 for count in label_counts.values())
    assert len(object_views) == 1440
