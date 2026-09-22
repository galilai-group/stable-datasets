import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.flavia import Flavia


# Published class order is not filename order, and the original label 13 is absent.
CLASS_RANGES = [
    (1001, 1059, "pubescent_bamboo"),
    (1060, 1122, "chinese_horse_chestnut"),
    (1552, 1616, "anhui_barberry"),
    (1123, 1194, "chinese_redbud"),
    (1195, 1267, "true_indigo"),
    (1268, 1323, "japanese_maple"),
    (1324, 1385, "nanmu"),
    (1386, 1437, "castor_aralia"),
    (1497, 1551, "chinese_cinnamon"),
    (1438, 1496, "goldenrain_tree"),
    (2001, 2050, "big-fruited_holly"),
    (2051, 2113, "japanese_cheesewood"),
    (2114, 2165, "wintersweet"),
    (2166, 2230, "camphortree"),
    (2231, 2290, "japan_arrowwood"),
    (2291, 2346, "sweet_osmanthus"),
    (2347, 2423, "deodar"),
    (2424, 2485, "ginkgo"),
    (2486, 2546, "crape_myrtle"),
    (2547, 2612, "oleander"),
    (2616, 2675, "yew_plum_pine"),
    (3001, 3055, "japanese_flowering_cherry"),
    (3056, 3110, "glossy_privet"),
    (3111, 3175, "chinese_toon"),
    (3176, 3229, "peach"),
    (3230, 3281, "ford_woodlotus"),
    (3282, 3334, "trident_maple"),
    (3335, 3389, "beale's_barberry"),
    (3390, 3446, "southern_magnolia"),
    (3447, 3510, "canadian_poplar"),
    (3511, 3563, "chinese_tulip_tree"),
    (3566, 3621, "tangerine"),
]


def _builder():
    builder = object.__new__(Flavia)
    builder.__init__()
    return builder


def _write_archive(path, image_ids, mode="RGB", include_metadata=False):
    with io.BytesIO() as buffer:
        Image.new(mode, (12, 8), color=100).save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
    with tarfile.open(path, "w:bz2") as archive:
        directory = tarfile.TarInfo("Leaves")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for image_id in image_ids:
            member = tarfile.TarInfo(f"Leaves/{image_id:04d}.jpg")
            member.size = len(image_bytes)
            archive.addfile(member, io.BytesIO(image_bytes))
        if include_metadata:
            for name in ("Leaves/README.txt", "Leaves/.DS_Store", "__MACOSX/Leaves/._1001.jpg"):
                data = b"Not an image"
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return path


def test_flavia_schema_and_class_order():
    info = _builder().info
    assert set(info.features) == {"image", "label"}
    assert info.supervised_keys == ("image", "label")
    assert info.features["label"].names == [name for _, _, name in CLASS_RANGES]
    assert len(info.features["label"].names) == 32
    assert info.features["label"].int2str(2) == "anhui_barberry"
    assert info.features["label"].int2str(12) == "wintersweet"
    assert info.homepage == "https://flavia.sourceforge.net/"
    assert "A Leaf Recognition Algorithm" in info.citation
    assert "GPL" not in (info.license or "")


def test_flavia_archive_maps_all_class_boundaries(tmp_path):
    image_ids = [image_id for first, last, _ in CLASS_RANGES for image_id in (first, last)]
    archive = _write_archive(tmp_path / "Leaves.tar.bz2", reversed(image_ids), include_metadata=True)

    examples = list(_builder()._generate_examples(archive, split="train"))

    assert len(examples) == 64
    assert len({key for key, _ in examples}) == 64
    expected = {
        f"Leaves/{image_id}.jpg": label
        for label, (first, last, _) in enumerate(CLASS_RANGES)
        for image_id in (first, last)
    }
    assert {key: sample["label"] for key, sample in examples} == expected
    for _, sample in examples:
        assert set(sample) == {"image", "label"}
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        assert sample["image"].size == (12, 8)
        assert np.asarray(sample["image"]).dtype == np.uint8
        assert isinstance(sample["label"], int)


@pytest.mark.parametrize("mode", ["L", "CMYK"])
def test_flavia_archive_converts_images_to_rgb(tmp_path, mode):
    archive = _write_archive(tmp_path / "Leaves.tar.bz2", [1001], mode=mode)
    _, sample = next(_builder()._generate_examples(archive, split="train"))
    assert sample["image"].mode == "RGB"
    assert np.asarray(sample["image"]).shape == (8, 12, 3)


@pytest.mark.parametrize("image_id", [1000, 1617, 1999, 2613, 2614, 2615, 2676, 3564, 3565, 3622])
def test_flavia_rejects_unmapped_image_ids(tmp_path, image_id):
    archive = _write_archive(tmp_path / "Leaves.tar.bz2", [image_id])
    with pytest.raises(ValueError, match=f"{image_id}"):
        list(_builder()._generate_examples(archive, split="train"))


def test_flavia_offline_cache_and_single_split(tmp_path, monkeypatch):
    archive = _write_archive(tmp_path / "Leaves.tar.bz2", [1552, 1123, 2114, 3621])
    downloads = []

    def local_download(urls, dest_folder):
        downloads.append((list(urls), dest_folder))
        return [archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", local_download)
    kwargs = {"processed_cache_dir": tmp_path / "processed", "download_dir": tmp_path / "downloads"}
    datasets = Flavia(split=None, **kwargs)
    assert set(datasets) == {"train"}
    dataset = datasets["train"]
    assert len(dataset) == 4
    assert [dataset[i]["label"] for i in range(len(dataset))] == [2, 3, 12, 31]
    assert isinstance(dataset[0]["image"], Image.Image)
    assert dataset[0]["image"].mode == "RGB"
    assert dataset.features["label"].names == [name for _, _, name in CLASS_RANGES]
    assert downloads == [([Flavia.SOURCE["assets"]["train"]], tmp_path / "downloads")]

    def unexpected_regeneration(*args, **kwargs):
        pytest.fail("Prepared data should be loaded from its Arrow cache")

    monkeypatch.setattr(Flavia, "_generate_examples", unexpected_regeneration)
    cached = Flavia(split="train", **kwargs)
    assert len(cached) == 4
    assert cached[2]["label"] == 12
    with pytest.raises(ValueError, match="Split 'test' not found"):
        Flavia(split="test", **kwargs)


def test_flavia_full_dataset_contract_with_offline_archive(tmp_path, monkeypatch):
    image_ids = [image_id for first, last, _ in CLASS_RANGES for image_id in range(first, last + 1)]
    archive = _write_archive(tmp_path / "Leaves.tar.bz2", image_ids)
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda urls, dest_folder: [archive])
    monkeypatch.setenv("STABLE_DATASETS_CACHE_DIR", str(tmp_path / "cache"))
    test_flavia_dataset()


@pytest.mark.large
def test_flavia_dataset():
    dataset = Flavia(split="train")
    assert len(dataset) == 1907
    assert dataset.features["label"].names == [name for _, _, name in CLASS_RANGES]
    sample = dataset[0]
    assert set(sample) == {"image", "label"}
    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].mode == "RGB"
    pixels = np.asarray(sample["image"])
    assert pixels.ndim == 3
    assert pixels.shape[2] == 3
    assert pixels.dtype == np.uint8
    labels = np.fromiter((row["label"] for row in dataset), dtype=np.int64, count=len(dataset))
    assert isinstance(sample["label"], int)
    assert set(labels) == set(range(32))
    assert np.bincount(labels).tolist() == [last - first + 1 for first, last, _ in CLASS_RANGES]
