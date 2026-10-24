import csv
import io
import zipfile

import numpy as np
from PIL import Image

import stable_datasets.utils as utils
from stable_datasets.images import GroceryStore


def _jpeg_bytes(color=(200, 40, 40)):
    buff = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buff, format="JPEG")
    return buff.getvalue()


def _create_grocery_zip(path):
    # Same layout as the GitHub archive: everything sits under "<repo>-<commit>/".
    root = "GroceryStoreDataset-0123abc/dataset/"
    split_files = {
        "train": [
            "train/Fruit/Apple/Granny-Smith/Granny-Smith_001.jpg, 1, 0",
            "train/Fruit/Banana/Banana_001.jpg, 6, 2",
            "train/Vegetables/Zucchini/Zucchini_001.jpg, 80, 42",
        ],
        "val": ["val/Fruit/Banana/Banana_001.jpg, 6, 2"],
        "test": [
            "test/Packages/Milk/Arla-Standard-Milk/Arla-Standard-Milk_001.jpg, 41, 20",
            "test/Fruit/Apple/Pink-Lady/Pink-Lady_001.jpg, 2, 0",
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root + "classes.csv", "Class Name (str),Class ID (int),Coarse Class Name (str)\n")
        for split, lines in split_files.items():
            archive.writestr(f"{root}{split}.txt", "\n".join(lines) + "\n")
            for line in lines:
                archive.writestr(root + line.split(",")[0], _jpeg_bytes())


def test_grocery_store_local_archive(tmp_path, monkeypatch):
    zip_path = tmp_path / "GroceryStoreDataset.zip"
    _create_grocery_zip(zip_path)
    requested = []

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        urls = list(urls)
        requested.extend(urls)
        return [zip_path] * len(urls)

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)
    ds = GroceryStore(processed_cache_dir=str(tmp_path / "processed"), download_dir=str(tmp_path / "downloads"))

    # All three splits come from the same archive, so it should only be fetched once.
    assert len(requested) == 1
    assert {split: len(ds[split]) for split in ds} == {"train": 3, "validation": 1, "test": 2}

    features = ds["test"].features
    rows = sorted(
        (features["label"].int2str(ex["label"]), features["coarse_label"].int2str(ex["coarse_label"]))
        for ex in ds["test"]
    )
    assert rows == [("Arla-Standard-Milk", "Milk"), ("Pink-Lady", "Apple")]

    sample = ds["train"][0]
    assert set(sample.keys()) == {"image", "label", "coarse_label"}
    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].mode == "RGB"


def test_grocery_store_dataset():
    train = GroceryStore(split="train")
    val = GroceryStore(split="validation")
    test = GroceryStore(split="test")

    # Split sizes from train.txt, val.txt and test.txt in the dataset repository.
    assert len(train) == 2640, f"Expected 2640 training samples, got {len(train)}."
    assert len(val) == 296, f"Expected 296 validation samples, got {len(val)}."
    assert len(test) == 2485, f"Expected 2485 test samples, got {len(test)}."

    sample = train[0]
    expected_keys = {"image", "label", "coarse_label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Expected an RGB image, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    assert isinstance(sample["label"], int), f"Label should be int, got {type(sample['label'])}."
    assert 0 <= sample["label"] < 81, f"Label should be in [0, 80], got {sample['label']}."
    assert 0 <= sample["coarse_label"] < 43, f"Coarse label should be in [0, 42], got {sample['coarse_label']}."

    fine = train.table.column("label").to_pylist()
    coarse = train.table.column("coarse_label").to_pylist()
    assert len(set(fine)) == 81, f"Expected all 81 fine-grained classes in train, got {len(set(fine))}."
    assert len(set(coarse)) == 43, f"Expected all 43 coarse-grained classes in train, got {len(set(coarse))}."

    # Every fine-grained class belongs to exactly one coarse-grained class.
    pairs = set(zip(fine, coarse))
    assert len(pairs) == 81, f"Expected one coarse class per fine class, got {len(pairs)} pairs."
    names = train.features["label"].names
    coarse_names = train.features["coarse_label"].names
    assert (names.index("Granny-Smith"), coarse_names.index("Apple")) in pairs
    assert (names.index("Oatly-Oat-Milk"), coarse_names.index("Oat-Milk")) in pairs

    # The class names in the builder are copied from dataset/classes.csv, so compare them with it.
    with zipfile.ZipFile(utils.download(GroceryStore.SOURCE["assets"]["train"])) as archive:
        csv_name = next(n for n in archive.namelist() if n.endswith("dataset/classes.csv"))
        rows = list(csv.DictReader(io.StringIO(archive.read(csv_name).decode("utf-8"))))
    fine_from_csv = {int(r["Class ID (int)"]): r["Class Name (str)"] for r in rows}
    coarse_from_csv = {int(r["Coarse Class ID (int)"]): r["Coarse Class Name (str)"] for r in rows}
    assert names == [fine_from_csv[i] for i in range(81)]
    assert coarse_names == [coarse_from_csv[i] for i in range(43)]

    # The validation split only has images for 60 of the 81 classes.
    assert len(set(val.table.column("label").to_pylist())) == 60
    assert len(set(test.table.column("label").to_pylist())) == 81
