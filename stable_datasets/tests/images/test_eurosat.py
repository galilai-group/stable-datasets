import io
import zipfile

import numpy as np
from PIL import Image

from stable_datasets.images import EuroSAT
from stable_datasets.schema import ClassLabel
from stable_datasets.schema import Image as ImageFeature


def _builder():
    # Bypass __new__ so no download/cache build happens.
    inst = object.__new__(EuroSAT)
    inst.__init__()
    return inst


def test_eurosat_metadata():
    builder = _builder()
    info = builder.info

    assert str(EuroSAT.VERSION) == "1.0.0"
    assert EuroSAT.SOURCE["homepage"] == "https://github.com/phelber/EuroSAT"
    assert set(EuroSAT.SOURCE["assets"].keys()) == {"train"}

    assert list(info.features.keys()) == ["image", "label"]
    assert isinstance(info.features["image"], ImageFeature)
    assert isinstance(info.features["label"], ClassLabel)
    assert info.features["label"].names == EuroSAT._labels()
    assert len(info.features["label"].names) == 10
    assert info.supervised_keys == ("image", "label")


def test_eurosat_generate_examples_matches_info(tmp_path):
    """Smoke-test _generate_examples on a tiny synthetic archive mirroring the real layout."""
    builder = _builder()
    names = builder.info.features["label"].names

    zip_path = tmp_path / "EuroSAT.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for name in names[:3]:
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="JPEG")
            z.writestr(f"2750/{name}/{name}_1.jpg", buf.getvalue())
        z.writestr("2750/README.txt", "not an image")
        z.writestr("2750/UnknownClass/UnknownClass_1.jpg", b"ignored")

    examples = list(builder._generate_examples(data_path=zip_path, split="train"))

    assert len(examples) == 3
    for key, example in examples:
        assert set(example.keys()) == set(builder.info.features.keys())
        assert isinstance(example["image"], Image.Image)
        assert example["image"].mode == "RGB"
        assert isinstance(example["label"], int)
        assert names[example["label"]] == key.split("/")[-2]


def test_eurosat_dataset():
    # EuroSAT ships as one archive with no official split; the builder
    # exposes it as a single "train" split (same pattern as AWA2).
    eurosat = EuroSAT(split="train")

    # Test 1: Check the dataset size matches the known total (27,000 images).
    assert len(eurosat) == 27000, f"Expected 27000 images, got {len(eurosat)}."

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = eurosat[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"EuroSAT images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"EuroSAT images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range (10 classes)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < 10, f"Label should be between 0 and 9 (10 classes), got {label}."

    # Test 5: Validate that all 10 classes are represented.
    labels_set = {eurosat[i]["label"] for i in range(0, len(eurosat), 97)}
    assert len(labels_set) == 10, f"Expected all 10 classes to be represented, found {len(labels_set)}."
