import io
import tarfile

from PIL import Image

from stable_datasets.images.imagenet_1k import ImageNet1K
from stable_datasets.images.imagenet_10 import Imagenette
from stable_datasets.images.imagenet_100 import ImageNet100


def _jpeg_bytes(color=(255, 0, 0)):
    arr = Image.new("RGB", (8, 8), color=color)
    buff = io.BytesIO()
    arr.save(buff, format="JPEG")
    return buff.getvalue()


def _create_imagenet_train_tar(path, num_classes=3, images_per_class=2):
    with tarfile.open(path, "w") as outer:
        for idx in range(num_classes):
            class_name = f"n{idx:08d}"
            class_buf = io.BytesIO()
            with tarfile.open(fileobj=class_buf, mode="w") as inner:
                for j in range(images_per_class):
                    img = _jpeg_bytes(color=(idx, j, 0))
                    info = tarfile.TarInfo(name=f"{class_name}_{j}.JPEG")
                    info.size = len(img)
                    inner.addfile(info, io.BytesIO(img))
            class_bytes = class_buf.getvalue()
            outer_info = tarfile.TarInfo(name=f"{class_name}.tar")
            outer_info.size = len(class_bytes)
            outer.addfile(outer_info, io.BytesIO(class_bytes))


def _create_imagenette_tar(path):
    with tarfile.open(path, "w:gz") as archive:
        classes = ["n01440764", "n02102040"]
        for split in ["train", "val"]:
            for cls in classes:
                img = _jpeg_bytes()
                name = f"imagenette2/{split}/{cls}/{cls}_{split}.JPEG"
                info = tarfile.TarInfo(name=name)
                info.size = len(img)
                archive.addfile(info, io.BytesIO(img))


def test_imagenet_1k_streaming_integration(tmp_path, monkeypatch):
    tar_path = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(tar_path, num_classes=2, images_per_class=2)

    monkeypatch.setattr("stable_datasets.images.imagenet_1k.download", lambda *args, **kwargs: tar_path)

    dataset = ImageNet1K(split="train", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(dataset) == 4
    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label"}
    assert isinstance(sample["image"], Image.Image)
    assert isinstance(sample["label"], int)


def test_imagenet_100_streaming_integration(tmp_path, monkeypatch):
    tar_path = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(tar_path, num_classes=120, images_per_class=1)

    monkeypatch.setattr("stable_datasets.images.imagenet_100.download", lambda *args, **kwargs: tar_path)

    dataset = ImageNet100(split="train", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(dataset) == 100
    labels = {dataset[i]["label"] for i in range(len(dataset))}
    assert labels == set(range(100))


def test_imagenet_10_integration(tmp_path, monkeypatch):
    tar_path = tmp_path / "imagenette2.tgz"
    _create_imagenette_tar(tar_path)

    monkeypatch.setattr("stable_datasets.images.imagenet_10.download", lambda *args, **kwargs: tar_path)

    train = Imagenette(split="train", streaming=False, processed_cache_dir=tmp_path / "processed")
    test = Imagenette(split="test", streaming=False, processed_cache_dir=tmp_path / "processed")

    assert len(train) == 2
    assert len(test) == 2
    assert isinstance(train[0]["image"], Image.Image)
    assert 0 <= train[0]["label"] < 10
