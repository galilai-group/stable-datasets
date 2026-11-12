from unittest.mock import patch

import numpy as np
from PIL import Image

from stable_datasets.images import BSDS500
from stable_datasets.images.bsds500 import decode_mask, encode_mask


def test_bsds500_mask_roundtrip():
    # Offline: encoding then decoding a mask must give back exactly the same values,
    # including region labels above 255 (stored as 16-bit PNG).
    mask = np.random.default_rng(0).integers(0, 1000, size=(321, 481)).astype(np.uint16)
    decoded = decode_mask(encode_mask(mask))
    assert decoded.dtype == np.uint16
    assert np.array_equal(decoded, mask)


def test_bsds500_dataset():
    # BSDS500(split=None) automatically downloads and loads all splits
    ds = BSDS500(split=None)

    # Test 1: Official split sizes, and no image appears in more than one split
    expected_sizes = {"train": 200, "validation": 100, "test": 200}
    assert set(ds.keys()) == set(expected_sizes), f"Expected splits {set(expected_sizes)}, got {set(ds.keys())}."
    ids = {}
    for split, n in expected_sizes.items():
        assert len(ds[split]) == n, f"Expected {n} {split} samples, got {len(ds[split])}."
        ids[split] = [sample["image_id"] for sample in ds[split]]
        assert len(set(ids[split])) == n, f"Duplicate image ids in {split} split."
    all_ids = ids["train"] + ids["validation"] + ids["test"]
    assert len(set(all_ids)) == len(all_ids), "Some images appear in more than one split."

    # Test 2: Each sample has the expected keys and the image is an RGB picture
    sample = ds["train"][0]
    expected_keys = {"image", "image_id", "segmentations", "boundaries"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."
    assert isinstance(sample["image"], Image.Image), f"Image should be a PIL image, got {type(sample['image'])}."
    assert sample["image"].mode == "RGB", f"Image should be RGB, got {sample['image'].mode}."

    # Test 3: Every image has at least one annotation, with matching segmentation/boundary counts,
    # and every decoded mask has the same height and width as its image
    for split in expected_sizes:
        for sample in ds[split]:
            height, width = np.array(sample["image"]).shape[:2]
            segmentations, boundaries = sample["segmentations"], sample["boundaries"]
            assert len(segmentations) >= 1, f"Image {sample['image_id']} has no annotations."
            assert len(segmentations) == len(boundaries), (
                f"Image {sample['image_id']}: {len(segmentations)} segmentations but {len(boundaries)} boundaries."
            )
            for seg, bdry in zip(segmentations, boundaries):
                assert decode_mask(seg).shape == (height, width), f"Segmentation size mismatch: {sample['image_id']}."
                assert decode_mask(bdry).shape == (height, width), f"Boundary size mismatch: {sample['image_id']}."

    # Test 4: Values make sense: region labels start at 1, boundaries are 0/1
    seg = decode_mask(ds["train"][0]["segmentations"][0])
    bdry = decode_mask(ds["train"][0]["boundaries"][0])
    assert seg.min() >= 1, f"Region labels should start at 1, got minimum {seg.min()}."
    assert set(np.unique(bdry)) <= {0, 1}, f"Boundaries should be 0/1, got {np.unique(bdry)}."

    # Test 5: A second load comes from the cache and does not download again
    with patch("stable_datasets.utils.bulk_download", side_effect=AssertionError("Dataset was re-downloaded.")):
        cached = BSDS500(split="train")
    assert len(cached) == expected_sizes["train"]

    print("All BSDS500 dataset tests passed successfully!")
