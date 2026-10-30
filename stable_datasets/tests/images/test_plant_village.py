import io
import json
import zipfile
from pathlib import PurePosixPath

import pytest
from PIL import Image

from stable_datasets.images import PlantVillage
from stable_datasets.splits import Split


def make_image_bytes():
    image = Image.new("RGB", (8, 8), color=(30, 150, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def plant_village_archive(tmp_path):
    archive_path = tmp_path / "plantvillage.zip"
    image_bytes = make_image_bytes()
    labels = PlantVillage._labels()[:2]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for label in labels:
            for index in range(20):
                archive.writestr(
                    f"PlantVillage-Dataset-master/raw/color/{label}/leaf_{index // 2}_view_{index % 2}.jpg",
                    image_bytes,
                )
        archive.writestr(
            f"PlantVillage-Dataset-master/raw/grayscale/{labels[0]}/ignored.jpg",
            image_bytes,
        )
        archive.writestr(
            f"PlantVillage-Dataset-master/raw/color/{labels[0]}/notes.txt",
            "not an image",
        )
    return archive_path


@pytest.fixture
def split_manifests(tmp_path):
    manifests = {}
    for split, indices in ((Split.TRAIN, range(16)), (Split.TEST, range(16, 20))):
        paths = [
            f"raw/color/{label}/leaf_{index // 2}_view_{index % 2}.jpg"
            for label in PlantVillage._labels()[:2]
            for index in indices
        ]
        path = tmp_path / f"color_{split}.txt"
        path.write_text("\n".join(reversed(paths)) + "\n", encoding="utf-8")
        manifests[split] = path
    return manifests


def make_builder():
    builder = object.__new__(PlantVillage)
    builder._dataset_info = builder._info()
    return builder


def test_label_count():
    labels = PlantVillage._labels()
    assert len(labels) == 38
    assert len(set(labels)) == 38


def test_default_color_configuration():
    builder = object.__new__(PlantVillage)
    PlantVillage.__init__(builder)
    assert builder.config.name == "color"
    assert builder.info.config_name == "color"


def test_candidate_splits():
    assert make_builder()._candidate_splits() == [Split.TRAIN, Split.TEST]


def test_exact_manifest_membership(plant_village_archive, split_manifests):
    with zipfile.ZipFile(plant_village_archive) as archive:
        splits = {split: PlantVillage._split_members(archive, path) for split, path in split_manifests.items()}
    assert {split: len(names) for split, names in splits.items()} == {Split.TRAIN: 32, Split.TEST: 8}
    for split, names in splits.items():
        expected = set(split_manifests[split].read_text().splitlines())
        assert {name.split("/", 1)[1] for name in names} == expected
        assert names == sorted(names)
    train, test = (set(splits[split]) for split in (Split.TRAIN, Split.TEST))
    assert train.isdisjoint(test)
    assert len(train | test) == 40
    # Two views of each synthetic leaf must follow its manifest assignment.
    leaf_groups = [
        {(PurePosixPath(name).parent.name, PurePosixPath(name).stem.split("_view_")[0]) for name in names}
        for names in (train, test)
    ]
    assert leaf_groups[0].isdisjoint(leaf_groups[1])


def test_archive_prefix_does_not_change_membership(plant_village_archive, split_manifests, tmp_path):
    alternate_path = tmp_path / "different-prefix.zip"
    with zipfile.ZipFile(plant_village_archive) as source, zipfile.ZipFile(alternate_path, "w") as target:
        for name in reversed(source.namelist()):
            target.writestr(name.split("/", 1)[1], source.read(name))
    for manifest in split_manifests.values():
        with zipfile.ZipFile(plant_village_archive) as source, zipfile.ZipFile(alternate_path) as target:
            original = PlantVillage._split_members(source, manifest)
            alternate = PlantVillage._split_members(target, manifest)
        assert [name.split("/", 1)[1] for name in original] == alternate


@pytest.mark.parametrize(("split", "expected_count"), [(Split.TRAIN, 32), (Split.TEST, 8)])
def test_generate_examples(plant_village_archive, split_manifests, tmp_path, monkeypatch, split, expected_count):
    builder = make_builder()
    builder._raw_download_dir = tmp_path
    monkeypatch.setattr(
        "stable_datasets.images.plant_village.bulk_download", lambda *args, **kwargs: [plant_village_archive]
    )
    examples = list(builder._generate_examples(data_path=split_manifests[split], split=split))
    assert len(examples) == expected_count
    for key, example in examples:
        assert key.endswith(".jpg")
        assert isinstance(example["image"], bytes)
        assert 0 <= example["label"] < 38
        with Image.open(io.BytesIO(example["image"])) as image:
            assert image.size == (8, 8)
            assert image.mode == "RGB"


@pytest.mark.parametrize("split", ["invalid", Split.VALIDATION])
def test_unknown_split_raises(plant_village_archive, split):
    with pytest.raises(ValueError, match="Unknown PlantVillage split"):
        list(make_builder()._generate_examples(data_path=plant_village_archive, split=split))


def test_missing_image_raises(plant_village_archive, tmp_path):
    manifest = tmp_path / "missing.txt"
    manifest.write_text("raw/color/Apple___healthy/missing.jpg\n")
    with zipfile.ZipFile(plant_village_archive) as archive, pytest.raises(ValueError, match="missing color image"):
        PlantVillage._split_members(archive, manifest)


def test_duplicate_manifest_entry_raises(plant_village_archive, split_manifests):
    manifest = split_manifests[Split.TRAIN]
    manifest.write_text(manifest.read_text() * 2)
    with zipfile.ZipFile(plant_village_archive) as archive, pytest.raises(ValueError, match="Duplicate image paths"):
        PlantVillage._split_members(archive, manifest)


def test_empty_manifest_raises(plant_village_archive, tmp_path, monkeypatch):
    manifest = tmp_path / "empty.txt"
    manifest.write_text("")
    builder = make_builder()
    builder._raw_download_dir = tmp_path
    monkeypatch.setattr(
        "stable_datasets.images.plant_village.bulk_download", lambda *args, **kwargs: [plant_village_archive]
    )
    with pytest.raises(ValueError, match="No PlantVillage color images"):
        list(builder._generate_examples(data_path=manifest, split=Split.TRAIN))


@pytest.fixture(scope="module")
def real_plant_village_dataset(tmp_path_factory):
    """Exercise the public loader, official manifests and fresh processed caches."""
    return PlantVillage(
        split=None,
        download_dir=tmp_path_factory.mktemp("plant-village-downloads"),
        processed_cache_dir=tmp_path_factory.mktemp("plant-village-processed"),
    )


def test_real_plant_village_dataset(real_plant_village_dataset):
    expected_counts = {Split.TRAIN: 43596, Split.TEST: 10709}
    assert set(real_plant_village_dataset) == set(expected_counts)
    for split, count in expected_counts.items():
        dataset = real_plant_village_dataset[split]
        assert len(dataset) == count
        assert dataset.info.config_name == "color"
        assert dataset.info.features["label"].names == PlantVillage._labels()
        assert set(dataset.table["label"].to_pylist()) == set(range(38))
        for index in (0, count // 2, count - 1):
            sample = dataset[index]
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], Image.Image)
            assert sample["image"].mode == "RGB"
            assert min(sample["image"].size) > 0
            assert isinstance(sample["label"], int)
            assert 0 <= sample["label"] < 38

    download_dir = real_plant_village_dataset._stable_datasets_download_dir
    archives = list(download_dir.glob("master.*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        color_members = {
            name
            for name in archive.namelist()
            if "/raw/color/" in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        splits = {}
        for split in expected_counts:
            manifests = list(download_dir.glob(f"plant-village-color-{split}.*.txt"))
            assert len(manifests) == 1
            splits[split] = set(PlantVillage._split_members(archive, manifests[0]))
            # Independently compare with every path in the authors' manifest.
            assert {name.split("/", 1)[1] for name in splits[split]} == set(manifests[0].read_text().splitlines())
            assert len(splits[split]) == expected_counts[split]
        leaf_map = json.loads(archive.read("PlantVillage-Dataset-master/leaf_grouping/leaf-map.json"))
    train, test = (splits[split] for split in expected_counts)
    assert len(color_members) == 54305
    assert train | test == color_members
    assert train.isdisjoint(test)

    # Check separation for leaves identified by the authors' metadata.
    leaf_ids = {}
    for split, members in splits.items():
        identified = set()
        for name in members:
            path = PurePosixPath(name)
            identifier = path.name.split("___")[-1].split("copy")[0]
            for suffix in (".jpg", ".JPG", ".png", ".PNG"):
                identifier = identifier.removesuffix(suffix)
            identifier = identifier.strip().lower()
            suggestions = leaf_map.get(identifier, [])
            if len(suggestions) == 1:
                identified.add(suggestions[0])
            else:
                identified.update(suggestion for suggestion in suggestions if path.parent.name in suggestion)
        assert identified
        leaf_ids[split] = identified
    assert leaf_ids[Split.TRAIN].isdisjoint(leaf_ids[Split.TEST])


def test_real_plant_village_cache_reopens_offline(real_plant_village_dataset, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("A processed-cache hit should not download the dataset")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", unexpected_download)
    monkeypatch.setattr("stable_datasets.images.plant_village.bulk_download", unexpected_download)
    reloaded = PlantVillage(
        split=Split.TEST,
        processed_cache_dir=real_plant_village_dataset._stable_datasets_processed_cache_dir,
    )
    assert len(reloaded) == 10709
    assert reloaded[0]["image"].mode == "RGB"
