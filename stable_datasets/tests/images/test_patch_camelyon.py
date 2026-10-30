import gzip
import io

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import stable_datasets.images.patch_camelyon as patch_camelyon
from stable_datasets.images import PatchCamelyon


PREFIX = "camelyonpatch_level_2_split_"


def _write_gzipped_h5(path, name, array):
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset(name, data=array)
    path.write_bytes(gzip.compress(buf.getvalue()))


def _write_split(src, tag, n, seed):
    """Write a tiny split in the release format: gzipped HDF5 images/labels and a metadata CSV."""
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    labels = (np.arange(n) % 2).astype(np.uint8)
    meta = pd.DataFrame(
        {
            "coord_y": rng.integers(0, 100_000, size=n),
            "coord_x": rng.integers(0, 100_000, size=n),
            # Patch 0 has tumor outside its center region, so it is labeled negative.
            "tumor_patch": labels.astype(bool) | (np.arange(n) == 0),
            "center_tumor_patch": labels.astype(bool),
            "wsi": [f"camelyon16_{tag}_{i % 2:03d}" for i in range(n)],
        }
    )
    _write_gzipped_h5(src / f"{PREFIX}{tag}_x.h5.gz", "x", images)
    _write_gzipped_h5(src / f"{PREFIX}{tag}_y.h5.gz", "y", labels.reshape(n, 1, 1, 1))
    meta.to_csv(src / f"{PREFIX}{tag}_meta.csv")  # leading unnamed index column, as in the release
    return images, meta


def _patch_downloads(monkeypatch, src):
    def fake_bulk_download(infos, dest_folder):
        return [src / info.url.rsplit("/", 1)[1] for info in infos]

    monkeypatch.setattr(patch_camelyon, "bulk_download", fake_bulk_download)


def test_patch_camelyon_builds_splits_from_release_files(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    written = {
        tag: _write_split(src, tag, n, seed) for tag, n, seed in (("train", 5, 0), ("valid", 3, 1), ("test", 2, 2))
    }
    _patch_downloads(monkeypatch, src)
    cache_kwargs = {"processed_cache_dir": tmp_path / "processed", "download_dir": tmp_path / "raw"}

    ds = PatchCamelyon(**cache_kwargs)
    assert {name: len(split) for name, split in ds.items()} == {"train": 5, "validation": 3, "test": 2}

    train = ds["train"]
    assert train.features["label"].names == ["negative", "positive"]
    images, meta = written["train"]
    for idx in range(5):
        row = train[idx]
        assert isinstance(row["image"], Image.Image)
        np.testing.assert_array_equal(np.array(row["image"]), images[idx])
        assert row["label"] == idx % 2
        assert row["wsi"] == meta["wsi"][idx]
        assert (row["coord_x"], row["coord_y"]) == (meta["coord_x"][idx], meta["coord_y"][idx])
        assert row["tumor_patch"] == bool(meta["tumor_patch"][idx])
    assert train[0]["tumor_patch"] and train[0]["label"] == 0

    validation_images, _ = written["valid"]
    np.testing.assert_array_equal(np.array(ds["validation"][2]["image"]), validation_images[2])

    # The decompressed temporary image files are removed after the caches are built.
    assert not list(src.glob("*.tmp"))

    # Cached splits are found without touching the downloads.
    def no_download(*args, **kwargs):
        raise AssertionError("cached splits should not trigger downloads")

    monkeypatch.setattr(patch_camelyon, "bulk_download", no_download)
    assert len(PatchCamelyon(split="test", **cache_kwargs)) == 2


def test_patch_camelyon_dataset():
    train = PatchCamelyon(split="train")
    assert len(train) == 262144, f"Expected 262144 training samples, got {len(train)}."

    sample = train[0]
    assert set(sample.keys()) == {"image", "label", "wsi", "coord_x", "coord_y", "tumor_patch"}
    assert isinstance(sample["image"], Image.Image)
    image_np = np.array(sample["image"])
    assert image_np.shape == (96, 96, 3), f"Image should have shape (96, 96, 3), got {image_np.shape}."
    assert image_np.dtype == np.uint8
    assert sample["label"] in (0, 1)

    expected = {"train": (262144, 131072), "validation": (32768, 16369), "test": (32768, 16377)}
    slides = {}
    for split, (num_rows, num_positive) in expected.items():
        table = PatchCamelyon(split=split).table
        assert table.num_rows == num_rows, f"{split}: expected {num_rows} rows, got {table.num_rows}"
        assert int(table.column("label").to_numpy().sum()) == num_positive, split
        slides[split] = set(table.column("wsi").to_pylist())

    # The official splits come from disjoint sets of slides.
    assert {split: len(names) for split, names in slides.items()} == {"train": 216, "validation": 54, "test": 129}
    assert not slides["train"] & slides["validation"]
    assert not slides["train"] & slides["test"]
    assert not slides["validation"] & slides["test"]

    assert tuple(train.with_format("torch")[0]["image"].shape) == (3, 96, 96)
