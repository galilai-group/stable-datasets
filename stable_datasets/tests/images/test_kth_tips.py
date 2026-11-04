from collections import Counter

from PIL import Image

from stable_datasets.images import KTHTIPS


def test_kth_tips_dataset():
    dataset = KTHTIPS(split="train")

    assert len(dataset) == 810

    sample = dataset[0]
    assert set(sample) == {"image", "label", "scale", "image_id"}
    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].size == (200, 200)
    assert sample["image"].mode == "RGB"

    label_counts = Counter()
    combinations = set()

    for index in range(len(dataset)):
        item = dataset[index]
        label = item["label"]
        scale = item["scale"]
        image_id = item["image_id"]

        assert 0 <= label < 10
        assert 1 <= scale <= 9
        assert 1 <= image_id <= 9

        label_counts[label] += 1
        combinations.add((label, scale, image_id))

    assert len(label_counts) == 10
    assert all(count == 81 for count in label_counts.values())
    assert len(combinations) == 810
