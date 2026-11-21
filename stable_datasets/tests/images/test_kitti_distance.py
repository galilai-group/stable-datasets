import hashlib
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import KITTIDistance
from stable_datasets.images.kitti_distance import _build_splits, _closest_vehicle_distance, distance_to_label


# ---------------------------------------------------------------------------
# Offline unit tests (synthetic KITTI-format archives, no network)
# ---------------------------------------------------------------------------


def test_distance_to_label_matches_vtab_bins():
    # VTAB: thrs = [-100, 8, 20, 999]; label = max(where(thrs - dist < 0)); boundaries go to the lower bin.
    assert distance_to_label(0.5) == 0
    assert distance_to_label(8.0) == 0
    assert distance_to_label(8.01) == 1
    assert distance_to_label(20.0) == 1
    assert distance_to_label(55.3) == 2
    assert distance_to_label(1000.0) == 3


def test_closest_vehicle_distance_ignores_non_vehicles():
    lines = [
        "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 3.10 0.01",
        "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57",
        "Van 0.00 1 -1.6 100.0 150.0 200.0 250.0 2.0 1.9 4.5 3.0 1.5 12.25 -1.5",
        "DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10",
    ]
    assert _closest_vehicle_distance(lines) == pytest.approx(12.25)
    assert _closest_vehicle_distance(lines[:1]) == 1000.0  # pedestrians only -> "no vehicle"


def _write_fake_kitti(download_dir: Path, n_drives=30, frames_per_drive=5, seed=0):
    """Write tiny KITTI-layout zips under the names `bulk_download` would cache them as."""
    rng = np.random.default_rng(seed)
    n = n_drives * frames_per_drive
    mapping = [
        f"2011_09_26 2011_09_26_drive_{d:04d}_sync {f:010d}" for d in range(n_drives) for f in range(frames_per_drive)
    ]
    rand = rng.permutation(n) + 1  # frame i -> 1-based mapping line

    def cached(url):
        p = Path(os.path.basename(urlparse(url).path))
        return download_dir / f"{p.stem}.{hashlib.sha256(url.encode()).hexdigest()[:10]}{p.suffix}"

    assets = KITTIDistance.SOURCE["assets"]
    download_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cached(assets["devkit"].url), "w") as z:
        z.writestr("mapping/train_mapping.txt", "\n".join(mapping) + "\n")
        z.writestr("mapping/train_rand.txt", ",".join(map(str, rand)))

    expected = {}
    with (
        zipfile.ZipFile(cached(assets["labels"].url), "w") as zl,
        zipfile.ZipFile(cached(assets["images"].url), "w") as zi,
    ):
        for i in range(n):
            label = int(rng.integers(4))
            z_m = [4.0, 14.0, 40.0, None][label]
            rows = ["Pedestrian 0.00 0 -0.2 1 1 2 2 1.8 0.5 1.2 1.0 1.5 2.0 0.0"]
            if z_m is not None:
                rows.append(f"Car 0.00 0 0.0 1 1 2 2 1.5 1.6 3.9 0.0 1.6 {z_m:.2f} 0.0")
            zl.writestr(f"training/label_2/{i:06d}.txt", "\n".join(rows) + "\n")
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 255, (24, 80, 3), dtype=np.uint8)).save(buf, format="PNG")
            zi.writestr(f"training/image_2/{i:06d}.png", buf.getvalue())
            expected[i] = label
    return cached(assets["devkit"].url), expected


def _tfds_reference_splits(rand_ids, mapping_lines):
    """Verbatim logic of TFDS kitti `_build_splits` (global np.random.seed)."""
    import collections

    video_to_image = collections.defaultdict(list)
    for image_id, line_id in enumerate(rand_ids):
        video_to_image[mapping_lines[line_id].split(" ")[1]].append(image_id)
    state = np.random.get_state()
    np.random.seed(seed=123)
    nt = max(1, 10 * len(video_to_image) // 100)
    nv = max(1, 10 * len(video_to_image) // 100)
    test_v = set(np.random.choice(sorted(video_to_image), nt, replace=False))
    val_v = set(np.random.choice(sorted(set(video_to_image) - test_v), nv, replace=False))
    tr, va, te = [], [], []
    for k, v in video_to_image.items():
        (te if k in test_v else va if k in val_v else tr).extend(v)
    np.random.set_state(state)
    return tr, va, te


def test_splits_match_tfds_and_are_drive_disjoint(tmp_path):
    devkit, _ = _write_fake_kitti(tmp_path)
    ours = _build_splits(devkit)
    with zipfile.ZipFile(devkit) as z:
        mapping = z.read("mapping/train_mapping.txt").decode().splitlines()
        rand = [int(x) - 1 for x in z.read("mapping/train_rand.txt").decode().split(",")]
    assert ours == _tfds_reference_splits(rand, mapping)

    drive = {i: mapping[rand[i]].split(" ")[1] for i in range(len(rand))}
    drives = [{drive[i] for i in s} for s in ours]
    assert not (drives[0] & drives[1]) and not (drives[0] & drives[2]) and not (drives[1] & drives[2])
    assert sum(map(len, ours)) == len(rand)


def test_kitti_distance_builder_on_fixture(tmp_path):
    _, expected = _write_fake_kitti(tmp_path / "dl")
    ds = KITTIDistance(download_dir=tmp_path / "dl", processed_cache_dir=tmp_path / "proc")
    sizes = {k: len(v) for k, v in ds.items()}
    assert set(sizes) == {"train", "validation", "test"}
    assert sum(sizes.values()) == len(expected)

    sample = ds["train"][0]
    assert set(sample) == {"image", "label", "distance"}
    assert isinstance(sample["image"], Image.Image) and np.array(sample["image"]).shape == (24, 80, 3)
    assert ds["train"].info.features["label"].names == ["<=8m", "8-20m", ">20m", "no_vehicle"]
    labels = [ds[s][i]["label"] for s in ds for i in range(len(ds[s]))]
    assert sorted(labels) == sorted(expected.values())


# ---------------------------------------------------------------------------
# Full download test (~12 GB of images), excluded from CI like other large datasets
# ---------------------------------------------------------------------------


@pytest.mark.large
def test_kitti_distance_dataset():
    ds = KITTIDistance()
    # TFDS kitti v3.x split sizes used by VTAB.
    assert {k: len(v) for k, v in ds.items()} == {"train": 6347, "validation": 423, "test": 711}

    sample = ds["train"][0]
    image = np.array(sample["image"])
    assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    assert 0 <= sample["label"] < 4
    assert distance_to_label(sample["distance"]) == sample["label"]
