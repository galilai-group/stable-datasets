from unittest.mock import patch

import numpy as np
from PIL import Image

from stable_datasets.images import WeizmannSegmentation
from stable_datasets.images.weizmann_seg import _marks_to_mask, decode_mask, encode_mask


def test_weizmann_mask_roundtrip():
    # Offline: encoding then decoding a label map must give back exactly the same values.
    mask = np.random.default_rng(0).integers(0, 3, size=(225, 300)).astype(np.uint8)
    decoded = decode_mask(encode_mask(mask))
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, mask)


def test_weizmann_marks_to_mask():
    # Offline: red marks -> 1, blue marks -> 2, gray (and slightly tinted gray) -> 0,
    # including the near-red / near-blue shades found in the real annotations.
    pixels = {
        (128, 128, 128): 0,  # gray background
        (255, 252, 252): 0,  # slightly tinted gray
        (255, 0, 0): 1,  # red
        (248, 9, 9): 1,  # near-red
        (0, 0, 255): 2,  # blue
        (36, 0, 255): 2,  # near-blue
    }
    marked = np.array([list(pixels)], dtype=np.uint8)  # shape (1, 6, 3)
    assert _marks_to_mask(marked).tolist() == [list(pixels.values())]


def test_weizmann_dataset():
    # WeizmannSegmentation(split=None) automatically downloads and loads both splits
    ds = WeizmannSegmentation(split=None)

    # Test 1: Both splits have 100 images, and no image appears in both
    assert set(ds.keys()) == {"one_object", "two_objects"}, f"Unexpected splits {set(ds.keys())}."
    ids = {}
    for split in ds:
        assert len(ds[split]) == 100, f"Expected 100 {split} samples, got {len(ds[split])}."
        ids[split] = {sample["image_id"] for sample in ds[split]}
        assert len(ids[split]) == 100, f"Duplicate image ids in {split} split."
    assert not ids["one_object"] & ids["two_objects"], "Some images appear in both splits."

    # Test 2: Each sample has the expected keys; color and grayscale images match in size
    sample = ds["one_object"][0]
    expected_keys = {"image", "image_bw", "image_id", "segmentations"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."
    assert isinstance(sample["image"], Image.Image), f"Image should be a PIL image, got {type(sample['image'])}."

    # Test 3: Per image: color is RGB, grayscale is L, number of segmentations is as documented,
    # every mask has the image's size and contains exactly the expected objects
    expected = {"one_object": ({3}, {0, 1}), "two_objects": ({1, 2, 3}, {0, 1, 2})}
    for split, (allowed_counts, labels) in expected.items():
        for sample in ds[split]:
            image, image_bw, image_id = sample["image"], sample["image_bw"], sample["image_id"]
            assert image.mode == "RGB", f"{image_id}: color image should be RGB, got {image.mode}."
            assert image_bw.mode == "L", f"{image_id}: grayscale image should be L, got {image_bw.mode}."
            assert image.size == image_bw.size, f"{image_id}: color and grayscale sizes differ."
            assert len(sample["segmentations"]) in allowed_counts, (
                f"{image_id}: {len(sample['segmentations'])} segmentations, expected one of {allowed_counts}."
            )
            for seg in sample["segmentations"]:
                mask = decode_mask(seg)
                assert mask.shape == (image.height, image.width), f"{image_id}: mask size mismatch."
                assert set(np.unique(mask).tolist()) == labels, f"{image_id}: mask labels {np.unique(mask)}."

    # Test 4: A second load comes from the cache and does not download again
    with patch("stable_datasets.utils.bulk_download", side_effect=AssertionError("Dataset was re-downloaded.")):
        cached = WeizmannSegmentation(split="one_object")
    assert len(cached) == 100

    print("All Weizmann segmentation dataset tests passed successfully!")
