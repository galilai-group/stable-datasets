import io
import tarfile

import numpy as np
from PIL import Image

from stable_datasets.images._imagenet_wnids import IN1K_CLASSES, WNID_TO_IDX
from stable_datasets.images.imagenet_1k import ImageNet1K
from stable_datasets.images.imagenet_10 import Imagenette
from stable_datasets.images.imagenet_100 import IN100_CLASSES, ImageNet100


def _jpeg_bytes(color=(255, 0, 0)):
    arr = Image.new("RGB", (8, 8), color=color)
    buff = io.BytesIO()
    arr.save(buff, format="JPEG")
    return buff.getvalue()


def _create_imagenet_train_tar(path, wnids, images_per_class=2):
    """Create a synthetic train tar containing one inner tar per wnid."""
    with tarfile.open(path, "w") as outer:
        for idx, wnid in enumerate(wnids):
            class_buf = io.BytesIO()
            with tarfile.open(fileobj=class_buf, mode="w") as inner:
                for j in range(images_per_class):
                    img = _jpeg_bytes(color=(idx % 256, j, 0))
                    info = tarfile.TarInfo(name=f"{wnid}_{j}.JPEG")
                    info.size = len(img)
                    inner.addfile(info, io.BytesIO(img))
            class_bytes = class_buf.getvalue()
            outer_info = tarfile.TarInfo(name=f"{wnid}.tar")
            outer_info.size = len(class_bytes)
            outer.addfile(outer_info, io.BytesIO(class_bytes))


def _create_imagenet_val_tar(path, num_files):
    """Create a flat val tar with ILSVRC2012_val_NNNNNNNN.JPEG entries (1-indexed)."""
    with tarfile.open(path, "w") as outer:
        for i in range(1, num_files + 1):
            img = _jpeg_bytes(color=(i % 256, 0, 0))
            info = tarfile.TarInfo(name=f"ILSVRC2012_val_{i:08d}.JPEG")
            info.size = len(img)
            outer.addfile(info, io.BytesIO(img))


def _create_imagenet_devkit_tar_gz(path, ilsvrc_id_to_wnid, val_ground_truth_ids):
    """Create a minimal devkit .tar.gz with meta.mat + validation ground-truth.

    ilsvrc_id_to_wnid: dict[int, str] — ILSVRC2012_ID → wnid.
    val_ground_truth_ids: list[int] — one ILSVRC2012_ID per val image (1-indexed order).
    """
    from scipy.io import savemat

    # Build a synthetic synsets struct array.
    n = len(ilsvrc_id_to_wnid)
    ids = sorted(ilsvrc_id_to_wnid.keys())
    synsets = np.zeros(
        (n,),
        dtype=[
            ("ILSVRC2012_ID", "O"),
            ("WNID", "O"),
            ("words", "O"),
            ("gloss", "O"),
            ("num_children", "O"),
            ("children", "O"),
            ("wordnet_height", "O"),
            ("num_train_images", "O"),
        ],
    )
    for i, sid in enumerate(ids):
        synsets[i]["ILSVRC2012_ID"] = sid
        synsets[i]["WNID"] = ilsvrc_id_to_wnid[sid]
        synsets[i]["words"] = ""
        synsets[i]["gloss"] = ""
        synsets[i]["num_children"] = 0
        synsets[i]["children"] = np.zeros((0,), dtype=np.int32)
        synsets[i]["wordnet_height"] = 0
        synsets[i]["num_train_images"] = 0

    meta_buf = io.BytesIO()
    savemat(meta_buf, {"synsets": synsets})

    gt_text = "\n".join(str(x) for x in val_ground_truth_ids).encode()

    with tarfile.open(path, "w:gz") as tf:
        meta_info = tarfile.TarInfo(name="ILSVRC2012_devkit_t12/data/meta.mat")
        meta_info.size = len(meta_buf.getvalue())
        tf.addfile(meta_info, io.BytesIO(meta_buf.getvalue()))

        gt_info = tarfile.TarInfo(name="ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        gt_info.size = len(gt_text)
        tf.addfile(gt_info, io.BytesIO(gt_text))


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


def _patch_imagenet_downloads(monkeypatch, module, asset_to_path):
    """Patch the module's `download` to dispatch by DownloadInfo.url suffix."""

    def fake_download(info, dest_folder=None, **kwargs):
        url = info["url"] if isinstance(info, dict) else getattr(info, "url", str(info))
        for asset_name, path in asset_to_path.items():
            if asset_name in url:
                return path
        raise AssertionError(f"unexpected download url: {url}")

    monkeypatch.setattr(f"{module}.download", fake_download)


def test_imagenet_1k_train_streaming(tmp_path, monkeypatch):
    wnids = ["n01440764", "n01443537"]  # first two canonical wnids
    train_tar = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(train_tar, wnids=wnids, images_per_class=2)

    # Train-only — stub val/devkit with placeholder unused paths.
    val_tar = tmp_path / "val.tar"
    _create_imagenet_val_tar(val_tar, num_files=0)
    devkit = tmp_path / "devkit.tar.gz"
    _create_imagenet_devkit_tar_gz(devkit, {1: "n01440764"}, val_ground_truth_ids=[])

    _patch_imagenet_downloads(
        monkeypatch,
        "stable_datasets.images.imagenet_1k",
        {"ILSVRC2012_img_train": train_tar, "ILSVRC2012_img_val": val_tar, "devkit": devkit},
    )

    ds = ImageNet1K(split="train", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 4
    sample = ds[0]
    assert set(sample.keys()) == {"image", "label"}
    assert isinstance(sample["image"], Image.Image)

    # Labels must equal the canonical sorted-wnid index, regardless of tar order.
    labels = {ds[i]["label"] for i in range(len(ds))}
    assert labels == {WNID_TO_IDX["n01440764"], WNID_TO_IDX["n01443537"]} == {0, 1}


def test_imagenet_1k_val_streaming(tmp_path, monkeypatch):
    val_wnids = ["n01440764", "n01443537", "n01484850", "n01491361"]
    train_tar = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(train_tar, wnids=val_wnids[:1])  # not used for val

    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    _create_imagenet_val_tar(val_tar, num_files=len(val_wnids))

    # Pick arbitrary ILSVRC IDs (not 0..N-1, to ensure ID parsing works correctly).
    ilsvrc_id_to_wnid = {17: val_wnids[0], 88: val_wnids[1], 213: val_wnids[2], 999: val_wnids[3]}
    val_ground_truth_ids = [17, 88, 213, 999]

    devkit = tmp_path / "devkit.tar.gz"
    _create_imagenet_devkit_tar_gz(devkit, ilsvrc_id_to_wnid, val_ground_truth_ids)

    _patch_imagenet_downloads(
        monkeypatch,
        "stable_datasets.images.imagenet_1k",
        {"ILSVRC2012_img_train": train_tar, "ILSVRC2012_img_val": val_tar, "devkit": devkit},
    )

    ds = ImageNet1K(split="validation", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 4
    labels = sorted(ds[i]["label"] for i in range(len(ds)))
    assert labels == sorted(WNID_TO_IDX[w] for w in val_wnids)


def test_imagenet_100_train_streaming(tmp_path, monkeypatch):
    # Include some in-bucket and some out-of-bucket wnids in the train archive.
    in_bucket = IN100_CLASSES[:3]
    out_bucket = [IN1K_CLASSES[100], IN1K_CLASSES[500]]
    train_tar = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(train_tar, wnids=in_bucket + out_bucket, images_per_class=1)

    val_tar = tmp_path / "val.tar"
    _create_imagenet_val_tar(val_tar, num_files=0)
    devkit = tmp_path / "devkit.tar.gz"
    _create_imagenet_devkit_tar_gz(devkit, {1: in_bucket[0]}, val_ground_truth_ids=[])

    # ImageNet100 inherits _split_generators from ImageNet1K, so `download` is
    # called from the imagenet_1k module.
    _patch_imagenet_downloads(
        monkeypatch,
        "stable_datasets.images.imagenet_1k",
        {"ILSVRC2012_img_train": train_tar, "ILSVRC2012_img_val": val_tar, "devkit": devkit},
    )

    ds = ImageNet100(split="train", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 3  # only in-bucket wnids contribute
    labels = {ds[i]["label"] for i in range(len(ds))}
    assert labels == {0, 1, 2}
    assert all(0 <= ds[i]["label"] < 100 for i in range(len(ds)))


def test_imagenet_100_val_streaming(tmp_path, monkeypatch):
    in_bucket = IN100_CLASSES[:3]
    out_bucket_wnid = IN1K_CLASSES[500]
    val_wnids = in_bucket + [out_bucket_wnid]  # last one should be filtered out

    train_tar = tmp_path / "ILSVRC2012_img_train.tar"
    _create_imagenet_train_tar(train_tar, wnids=in_bucket[:1])

    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    _create_imagenet_val_tar(val_tar, num_files=len(val_wnids))

    ilsvrc_id_to_wnid = {i + 1: w for i, w in enumerate(val_wnids)}
    val_ground_truth_ids = list(range(1, len(val_wnids) + 1))

    devkit = tmp_path / "devkit.tar.gz"
    _create_imagenet_devkit_tar_gz(devkit, ilsvrc_id_to_wnid, val_ground_truth_ids)

    _patch_imagenet_downloads(
        monkeypatch,
        "stable_datasets.images.imagenet_1k",
        {"ILSVRC2012_img_train": train_tar, "ILSVRC2012_img_val": val_tar, "devkit": devkit},
    )

    ds = ImageNet100(split="validation", streaming=True, processed_cache_dir=tmp_path / "processed")
    assert len(ds) == 3  # out-of-bucket wnid excluded
    labels = sorted(ds[i]["label"] for i in range(len(ds)))
    assert labels == [0, 1, 2]


def test_imagenet_10_validation(tmp_path, monkeypatch):
    tar_path = tmp_path / "imagenette2.tgz"
    _create_imagenette_tar(tar_path)

    monkeypatch.setattr("stable_datasets.images.imagenet_10.download", lambda *a, **k: tar_path)

    train = Imagenette(split="train", streaming=False, processed_cache_dir=tmp_path / "processed_train")
    val = Imagenette(split="validation", streaming=False, processed_cache_dir=tmp_path / "processed_val")

    assert len(train) == 2
    assert len(val) == 2
    assert isinstance(train[0]["image"], Image.Image)
    assert 0 <= train[0]["label"] < 10
