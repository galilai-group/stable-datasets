import io
import tarfile

import numpy as np
from PIL import Image

from stable_datasets.images import OxfordIIITPet
from stable_datasets.schema import ClassLabel
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split


def _builder():
    # Bypass __new__ so no download/cache build happens.
    inst = object.__new__(OxfordIIITPet)
    inst.__init__()
    return inst


def _add_bytes(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


def test_oxford_iiit_pet_metadata():
    builder = _builder()
    info = builder.info

    assert str(OxfordIIITPet.VERSION) == "1.0.0"
    assert OxfordIIITPet.SOURCE["homepage"] == "https://www.robots.ox.ac.uk/~vgg/data/pets/"
    assert set(OxfordIIITPet.SOURCE["assets"].keys()) == {"images", "annotations"}
    # Asset keys aren't split names, so cache lookup must use the real splits.
    assert builder._candidate_splits() == [Split.TRAIN, Split.TEST]

    assert list(info.features.keys()) == ["image", "label"]
    assert isinstance(info.features["image"], ImageFeature)
    assert isinstance(info.features["label"], ClassLabel)
    assert info.features["label"].names == OxfordIIITPet._labels()
    assert len(info.features["label"].names) == 37
    # Index i must be official class id i + 1 (annotations/list.txt).
    names = info.features["label"].names
    assert names[0] == "Abyssinian"
    assert names[3] == "basset_hound"
    assert names[5] == "Bengal"
    assert names[26] == "Ragdoll"
    assert names[36] == "yorkshire_terrier"
    assert info.supervised_keys == ("image", "label")


def test_oxford_iiit_pet_generate_examples_matches_info(tmp_path):
    """Smoke-test _generate_examples on tiny synthetic archives mirroring the real layout."""
    builder = _builder()
    names = builder.info.features["label"].names

    images_path = tmp_path / "images.tar.gz"
    with tarfile.open(images_path, "w:gz") as tar:
        for image_id in ["Abyssinian_1", "basset_hound_2", "yorkshire_terrier_3"]:
            buf = io.BytesIO()
            Image.new("RGB", (32, 24), color=(10, 20, 30)).save(buf, format="JPEG")
            _add_bytes(tar, f"images/{image_id}.jpg", buf.getvalue())
        _add_bytes(tar, "images/broken_4.jpg", b"not a jpeg")

    annotations_path = tmp_path / "annotations.tar.gz"
    with tarfile.open(annotations_path, "w:gz") as tar:
        _add_bytes(tar, "annotations/trainval.txt", b"#comment\nAbyssinian_1 1 1 1\nbasset_hound_2 4 2 3\n")
        _add_bytes(tar, "annotations/test.txt", b"yorkshire_terrier_3 37 2 25\n")

    path_map = {"images": images_path, "annotations": annotations_path}
    train = list(builder._generate_examples(path_map=path_map, split="trainval"))
    test = list(builder._generate_examples(path_map=path_map, split="test"))

    assert [key for key, _ in train] == ["Abyssinian_1", "basset_hound_2"]
    assert [key for key, _ in test] == ["yorkshire_terrier_3"]
    for key, example in train + test:
        assert set(example.keys()) == set(builder.info.features.keys())
        assert isinstance(example["image"], Image.Image)
        assert example["image"].mode == "RGB"
        assert isinstance(example["label"], int)
        assert names[example["label"]] == key.rsplit("_", 1)[0]


def test_oxford_iiit_pet_dataset():
    pets_train = OxfordIIITPet(split="train")

    # Test 1: Exact official trainval size.
    assert len(pets_train) == 3680, f"Expected 3680 training images, got {len(pets_train)}."

    # Test 2: Check sample keys
    sample = pets_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}."

    # Test 3: Validate image type
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Pet images should be HxWxC, got shape {image_np.shape}."
    assert image_np.shape[2] == 3, f"Pet images should have 3 channels (RGB), got {image_np.shape[2]}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."

    # Test 4: Validate label type and range (37 breeds)
    label = sample["label"]
    assert isinstance(label, int), f"Label should be int, got {type(label)}."
    assert 0 <= label < 37, f"Label should be in range [0, 36], got {label}."

    # Test 5: Exact official test size.
    pets_test = OxfordIIITPet(split="test")
    assert len(pets_test) == 3669, f"Expected 3669 test images, got {len(pets_test)}."
