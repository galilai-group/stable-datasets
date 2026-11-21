import io
import zipfile

import numpy as np
from PIL import Image

import stable_datasets.utils as utils
from stable_datasets.images import PlantDoc


def _image_bytes(mode, fmt):
    buff = io.BytesIO()
    Image.new(mode, (8, 8)).save(buff, format=fmt)
    return buff.getvalue()


def _rotated_jpeg_bytes():
    # An 8x4 JPEG whose EXIF orientation (6) says it should be shown rotated by 90 degrees.
    exif = Image.Exif()
    exif[0x0112] = 6
    buff = io.BytesIO()
    Image.new("RGB", (8, 4)).save(buff, format="JPEG", exif=exif.tobytes())
    return buff.getvalue()


def _create_plantdoc_zip(path):
    # Same layout as the GitHub archive: "<repo>-<commit>/<split>/<class>/<file>".
    root = "PlantDoc-Dataset-0123abc/"
    images = {
        "train/Apple leaf/apple_1.jpg": _image_bytes("RGB", "JPEG"),
        # The real data has a few PNG, CMYK and grayscale files with a .jpg name.
        "train/Apple leaf/apple_2.jpg": _image_bytes("RGBA", "PNG"),
        "train/Tomato leaf/tomato_1.jpg": _image_bytes("CMYK", "JPEG"),
        "test/Corn rust leaf/corn_1.jpg": _image_bytes("L", "JPEG"),
        "test/grape leaf/grape_1.jpg": _image_bytes("RGB", "JPEG"),
        "test/grape leaf/grape_2.jpg": _rotated_jpeg_bytes(),
    }
    folders = {"train", "test"} | {name.rsplit("/", 1)[0] for name in images}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root, "")
        archive.writestr(root + "README.md", "PlantDoc")
        for folder in sorted(folders):
            archive.writestr(f"{root}{folder}/", "")
        for name, data in images.items():
            archive.writestr(root + name, data)


def test_plantdoc_local_archive(tmp_path, monkeypatch):
    zip_path = tmp_path / "PlantDoc-Dataset.zip"
    _create_plantdoc_zip(zip_path)

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        return [zip_path] * len(list(urls))

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)
    ds = PlantDoc(processed_cache_dir=str(tmp_path / "processed"), download_dir=str(tmp_path / "downloads"))

    assert {split: len(ds[split]) for split in ds} == {"train": 3, "test": 3}

    label = ds["train"].features["label"]
    assert sorted(label.int2str(ex["label"]) for ex in ds["train"]) == ["Apple leaf", "Apple leaf", "Tomato leaf"]
    assert sorted(label.int2str(ex["label"]) for ex in ds["test"]) == ["Corn rust leaf", "grape leaf", "grape leaf"]

    # Every image comes back as RGB, whatever its format and color mode in the archive.
    for split in ds:
        for ex in ds[split]:
            assert isinstance(ex["image"], Image.Image)
            assert ex["image"].mode == "RGB"

    # The photo with an EXIF rotation is stored upright: 8x4 becomes 4x8.
    assert sorted(ex["image"].size for ex in ds["test"]) == [(4, 8), (8, 8), (8, 8)]


def test_plantdoc_dataset():
    train = PlantDoc(split="train")
    test = PlantDoc(split="test")

    # Number of images in the train/ and test/ folders of the dataset repository.
    assert len(train) == 2342, f"Expected 2342 training samples, got {len(train)}."
    assert len(test) == 236, f"Expected 236 test samples, got {len(test)}."

    sample = train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Expected an RGB image, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    assert isinstance(sample["label"], int), f"Label should be int, got {type(sample['label'])}."
    assert 0 <= sample["label"] < 28, f"Label should be in [0, 27], got {sample['label']}."

    train_labels = train.table.column("label").to_pylist()
    test_labels = test.table.column("label").to_pylist()
    assert len(set(train_labels)) == 28, f"Expected all 28 classes in train, got {len(set(train_labels))}."

    # "Tomato two spotted spider mites leaf" only has 2 train images and none in test.
    mites = train.features["label"].str2int("Tomato two spotted spider mites leaf")
    assert train_labels.count(mites) == 2
    assert set(range(28)) - set(test_labels) == {mites}

    # Every stored image is RGB with no EXIF rotation left. Reading only the file headers keeps
    # this fast, so all images are checked, including the ones the builder converted.
    for split_ds in (train, test):
        for i, row in enumerate(split_ds.set_decode(False)):
            with Image.open(io.BytesIO(row["image"])) as img:
                assert img.mode == "RGB", f"Image {i} has mode {img.mode}."
                assert img.getexif().get(0x0112, 1) <= 1, f"Image {i} still has an EXIF rotation."
