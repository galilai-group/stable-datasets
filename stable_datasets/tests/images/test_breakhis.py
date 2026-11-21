import re
from collections import Counter

import numpy as np
from PIL import Image

from stable_datasets.images.breakhis import _IMAGE_RE, FOLDS, MAGNIFICATIONS, TUMOR_TYPES, BreakHis, _read_fold
from stable_datasets.schema import Version
from stable_datasets.utils import download


# (train, test) images in each official fold.
FOLD_SIZES = {
    "fold1": (5005, 2904),
    "fold2": (5506, 2403),
    "fold3": (5332, 2577),
    "fold4": (5211, 2698),
    "fold5": (4826, 3083),
}

# (train, test) images per tumour type in fold 1, in label order.
FOLD1_CLASS_COUNTS = [(253, 191), (683, 331), (218, 235), (318, 251), (2313, 1138), (302, 324), (547, 245), (371, 189)]


def test_breakhis_info():
    """Test dataset info, features, configs and source provenance without downloading."""
    builder = object.__new__(BreakHis)
    builder.__init__()
    info = builder.info

    assert info.description is not None
    assert set(info.features) == {"image", "label", "malignant", "magnification", "patient_id"}
    assert info.features["label"].names == TUMOR_TYPES
    assert info.features["label"].num_classes == 8
    assert info.features["malignant"].names == ["benign", "malignant"]
    assert info.features["magnification"].names == MAGNIFICATIONS
    assert info.license == "CC BY 4.0"
    assert BreakHis.VERSION == Version("1.0.0")
    assert [config.name for config in BreakHis.BUILDER_CONFIGS] == FOLDS
    assert builder.config.name == "fold1"
    assert set(builder.SOURCE["assets"]) == {"images", "folds"}
    assert all(asset.checksum.startswith("sha256:") for asset in builder.SOURCE["assets"].values())


def test_breakhis_path_parsing():
    """Archive paths parse into tumour type, magnification, file name and patient ID."""
    match = _IMAGE_RE.match(
        "BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998EF/400X/SOB_B_F-14-21998EF-400-013.png"
    )
    assert match is not None
    assert match["tumor_type"] == "fibroadenoma"
    assert match["magnification"] == "400X"
    assert match["file_name"] == "SOB_B_F-14-21998EF-400-013.png"
    assert match["patient_id"] == "14-21998EF"

    assert _IMAGE_RE.match("BreaKHis_v1/histology_slides/breast/benign/process_db_stat.py") is None


def test_breakhis_official_folds(tmp_path):
    """Download the authors' fold lists and check each fold's sizes and that no slide is in both splits."""
    folds_path = download(BreakHis.SOURCE["assets"]["folds"], dest_folder=tmp_path, progress_bar=False)
    for fold, (n_train, n_test) in FOLD_SIZES.items():
        groups = _read_fold(folds_path, fold)
        assert len(groups) == 7909
        assert Counter(groups.values()) == {"train": n_train, "test": n_test}, fold

        # A slide is one tumour type and patient ID; 14-13412 has a ductal and a lobular carcinoma slide.
        slides = {}
        for name, split in groups.items():
            slide = re.match(r"SOB_[BM]_([A-Z]+-\d+-\d+[A-Z]*)-", name).group(1)
            assert slides.setdefault(slide, split) == split, f"{fold}: slide {slide} is in both splits."
        assert len(slides) == 82


def test_breakhis_dataset():
    """Download BreakHis and check fold 1's split sizes, per-class counts, sample format and slide-disjointness."""
    ds_train = BreakHis(split="train")
    ds_test = BreakHis(split="test")
    assert len(ds_train) == 5005, f"Expected 5005 train samples, got {len(ds_train)}."
    assert len(ds_test) == 2904, f"Expected 2904 test samples, got {len(ds_test)}."

    train_counts = Counter(ds_train.table.column("label").to_pylist())
    test_counts = Counter(ds_test.table.column("label").to_pylist())
    assert [(train_counts[i], test_counts[i]) for i in range(8)] == FOLD1_CLASS_COUNTS

    sample = ds_test[0]
    expected_keys = {"image", "label", "malignant", "magnification", "patient_id"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    img = sample["image"]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}."
    assert img.mode == "RGB", f"Expected RGB image, got {img.mode}."
    assert img.size in {(700, 460), (700, 456)}, f"Expected a 700x460 or 700x456 image, got {img.size}."
    img_np = np.array(img)
    assert img_np.dtype == np.uint8, f"Expected uint8 dtype, got {img_np.dtype}."

    assert 0 <= sample["label"] < 8, f"Expected label in [0, 8), got {sample['label']}."
    assert sample["malignant"] == int(sample["label"] >= 4)
    assert 0 <= sample["magnification"] < 4

    def slides(ds):
        return set(zip(ds.table.column("label").to_pylist(), ds.table.column("patient_id").to_pylist()))

    train_slides, test_slides = slides(ds_train), slides(ds_test)
    assert len(train_slides) + len(test_slides) == 82
    assert train_slides.isdisjoint(test_slides)

    ds_torch = ds_test.with_format("torch")
    torch_sample = ds_torch[0]
    assert torch_sample["image"].shape[0] == 3, f"Expected (3, H, W) torch tensor, got {torch_sample['image'].shape}."
