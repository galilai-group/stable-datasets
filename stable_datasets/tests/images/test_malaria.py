"""Offline fixture tests and full-download validation for Malaria."""

# Tests call builder internals that are not part of the public dataset API.
# pylint: disable=protected-access

import csv
import io
from collections import Counter
from zipfile import ZipFile

import pytest
from PIL import Image

from stable_datasets import StableDataset, StableDatasetDict
from stable_datasets.images import Malaria
from stable_datasets.schema import ClassLabel, DatasetSource, DownloadInfo


def _png_bytes(mode="RGB", size=(8, 6)):
    """Create a PNG bytes object for a PIL image."""
    with Image.new(mode, size) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _write_mapping(path, rows, *, quoted=False):
    """Write a patient mapping CSV file."""

    with path.open("w", encoding="utf-8", newline="") as file:
        for patient_id, filenames in rows:
            if quoted:
                csv.writer(file).writerow([patient_id, repr(filenames), "", ""])
            else:
                file.write(f"{patient_id},{filenames!r},,,\n")


def _require_dataset(value: object) -> StableDataset:
    """Narrow a builder result to the single split it returns."""
    if not isinstance(value, StableDataset):
        raise TypeError("Expected a StableDataset")
    # Lookup keeps Pylint from treating the result as the Malaria builder.
    return vars()["value"]


def _assert_rgb_image(image: object, size: tuple[int, int]) -> None:
    """Check that a decoded cell crop is an RGB image of the expected size."""
    # Grayscale and RGBA archive bytes are decoded to a PIL image.
    assert isinstance(image, Image.Image)
    # Every cell is stored as RGB, including crops that were not RGB on disk.
    assert image.mode == "RGB"
    # The fixture sizes are the bytes written into the zip, after conversion.
    assert image.size == size


def _require_dataset_dict(value: object) -> StableDatasetDict:
    """Narrow a builder result to the split mapping it returns."""
    if not isinstance(value, StableDatasetDict):
        raise TypeError("Expected a StableDatasetDict")
    # Lookup keeps Pylint from treating the result as the Malaria builder.
    return vars()["value"]


@pytest.fixture(name="malaria_assets")
def malaria_asset_paths(tmp_path):
    """Build a tiny cell-image archive and the two patient-mapping files."""
    parasitized = "C33P1thinF_IMG_20150619_114756a_cell_179.png"
    uninfected = "C33P1thinF_IMG_20150619_114756a_cell_180.png"
    other_parasitized = "C100P61ThinF_IMG_20150918_144104_cell_162.png"
    other_uninfected = "C100P61ThinF_IMG_20150918_144104_cell_128.png"
    members = {
        f"cell_images/Uninfected/{uninfected}": _png_bytes(mode="RGBA", size=(9, 7)),
        "cell_images/Uninfected/Thumbs.db": b"not an image",
        "cell_images/Parasitized/": b"",
        f"cell_images/Parasitized/{parasitized}": _png_bytes(mode="L", size=(11, 5)),
        f"other/Parasitized/{parasitized}": b"outside the dataset",
        "cell_images/Parasitized/Thumbs.db": b"not an image",
        "cell_images/Unknown/ignored.png": b"unknown class",
        f"cell_images/Parasitized/nested/{parasitized}": b"unexpected nesting",
        f"cell_images/Uninfected/{other_uninfected}": _png_bytes(),
        f"cell_images/Parasitized/{other_parasitized}": _png_bytes(),
    }
    path_map = {
        "images": tmp_path / "cell_images.zip",
        "patient_mapping_parasitized": tmp_path / "parasitized.csv",
        "patient_mapping_uninfected": tmp_path / "uninfected.csv",
    }
    with ZipFile(path_map["images"], "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    _write_mapping(
        path_map["patient_mapping_parasitized"],
        [("C33P1thinF", [parasitized]), ("C100P61ThinF", [other_parasitized])],
    )
    _write_mapping(
        path_map["patient_mapping_uninfected"],
        [("C33P1thinF", [uninfected]), ("C100P61ThinF", [other_uninfected])],
        quoted=True,
    )
    return path_map


def _builder():
    """Create a `Malaria` builder instance."""
    builder = object.__new__(Malaria)
    Malaria.__init__(builder)
    return builder


def test_malaria_fixture_loads_and_reuses_cache(tmp_path, monkeypatch, malaria_assets):
    """Load the fixture once, then reuse the processed cache with no second download."""
    cache_dir = tmp_path / "processed"
    download_dir = tmp_path / "downloads"
    calls = []

    def fake_bulk_download(specs, dest_folder):
        source = Malaria.SOURCE
        # Asset metadata lives on DatasetSource, not on the untyped mapping lookup.
        assert isinstance(source, DatasetSource)
        # The images zip and both patient mappings are requested, in source order.
        assert specs == list(source.assets.values())
        for spec in specs:
            # Each asset is a download spec rather than a bare URL string.
            assert isinstance(spec, DownloadInfo)
            # The builder refuses an asset that has no checksum.
            assert spec.checksum is not None
            # Checksums are sha256 digests, including the algorithm prefix.
            assert spec.checksum.startswith("sha256:")
        # Raw files go to the directory passed into Malaria(), not the default cache.
        assert dest_folder == download_dir
        calls.append(specs)
        return list(malaria_assets.values())

    monkeypatch.setattr(
        "stable_datasets.images.malaria.bulk_download", fake_bulk_download
    )
    ds = _require_dataset(
        Malaria(
            split="train",
            processed_cache_dir=cache_dir,
            download_dir=download_dir,
        )
    )
    # Thumbs.db, unknown classes, nested paths, and non-png members are skipped.
    assert len(ds) == 4
    # Building the split downloads the three assets once.
    assert len(calls) == 1
    # Labels are a ClassLabel, so stored values are class indices rather than names.
    assert isinstance(ds.features["label"], ClassLabel)
    # Index 0 is parasitized and index 1 is uninfected.
    assert ds.features["label"].names == ["parasitized", "uninfected"]
    # The supervised task is cell image to infection label.
    assert ds.info.supervised_keys == ("image", "label")

    rows = list(ds)
    # Archive order is by path: both parasitized crops, then both uninfected crops.
    assert [row["label"] for row in rows] == [0, 0, 1, 1]
    # Within each class, C100 sorts before C33, and that order repeats for uninfected.
    assert [row["patient_id"] for row in rows] == ["C100P61ThinF", "C33P1thinF"] * 2
    for row, size in zip(rows, [(8, 6), (11, 5), (8, 6), (9, 7)]):
        # Every example carries the image, its class, and the `Malaria` identity fields.
        assert set(row) == {
            "image",
            "label",
            "filename",
            "patient_id",
            "source_image_id",
        }
        _assert_rgb_image(row["image"], size)
    # The C33 parasitized crop records the microscopy image it was cut from.
    assert rows[1]["source_image_id"] == "C33P1thinF_IMG_20150619_114756a"
    # The matching C33 uninfected crop came from that same microscopy image.
    assert rows[1]["source_image_id"] == rows[3]["source_image_id"]
    for row in ds.with_format("raw"):
        raw_image = row["image"]
        # Raw rows keep the encoded PNG instead of a decoded PIL image.
        assert isinstance(raw_image, bytes)
        # The stored bytes are a PNG, including crops converted to RGB.
        assert raw_image.startswith(b"\x89PNG\r\n\x1a\n")

    def unexpected_download(*_args, **_kwargs):
        pytest.fail(
            "A cached `Malaria` dataset must not download its auxiliary assets again"
        )

    monkeypatch.setattr(
        "stable_datasets.images.malaria.bulk_download", unexpected_download
    )
    cached = _require_dataset_dict(
        Malaria(processed_cache_dir=cache_dir, download_dir=download_dir),
    )
    # A second load with no split still exposes only train.
    assert list(cached) == ["train"]
    # The cached train split is the same raw rows, without downloading again.
    assert list(cached["train"].with_format("raw")) == list(ds.with_format("raw"))
    for split in ("test", "validation", "patient_mapping_parasitized"):
        # Test, validation, and the mapping asset names are not dataset splits.
        with pytest.raises(ValueError, match="not found"):
            Malaria(
                split=split, processed_cache_dir=cache_dir, download_dir=download_dir
            )


def test_malaria_order_is_independent_of_zip_order(tmp_path, malaria_assets):
    """Yield the same examples when the zip member order is reversed."""
    builder = _builder()
    original = list(builder._generate_examples(malaria_assets, "train"))
    reversed_path = tmp_path / "reversed.zip"
    with (
        ZipFile(malaria_assets["images"]) as source,
        ZipFile(reversed_path, "w") as target,
    ):
        for member in reversed(source.infolist()):
            target.writestr(member, source.read(member))
    reversed_examples = list(
        builder._generate_examples(
            {**malaria_assets, "images": reversed_path},
            "train",
        )
    )
    # Examples are emitted in filename order, not in the order stored in the zip.
    assert [key for key, _ in original] == sorted(key for key, _ in original)
    # Reversing the zip members does not change that filename order.
    assert [key for key, _ in original] == [key for key, _ in reversed_examples]
    for (_, expected), (_, actual) in zip(original, reversed_examples):
        # Pixel values match for the crop that shares this filename.
        assert expected["image"].tobytes() == actual["image"].tobytes()
        # Label, filename, patient id, and source image id match as well.
        assert {key: value for key, value in expected.items() if key != "image"} == {
            key: value for key, value in actual.items() if key != "image"
        }


@pytest.mark.parametrize("quoted", [False, True])
def test_malaria_mapping_reads_padded_lists_and_preserves_ids(tmp_path, quoted):
    """Read padded and quoted `Malaria` filename lists without changing patient ids."""
    path = tmp_path / "mapping.csv"
    _write_mapping(
        path,
        [("C33P1thinF", ["first.png", "second.png"])],
        quoted=quoted,
    )
    # Padding cells and either quoting style still map each filename to its patient.
    assert Malaria._read_patient_mapping(path) == {
        "first.png": "C33P1thinF",
        "second.png": "C33P1thinF",
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("patient,not a list\n", "Invalid filename list"),
        ("patient,('first.png',)\n", "Invalid patient mapping"),
        (",['first.png']\n", "Invalid patient mapping"),
        ("patient,[123]\n", "Invalid cell filename"),
        ("patient,['']\n", "Invalid cell filename"),
        ("first,['cell.png']\nsecond,['cell.png']\n", "Conflicting patient mappings"),
    ],
)
def test_malaria_mapping_rejects_invalid_or_conflicting_rows(
    tmp_path, content, message
):
    """Reject filename lists, patient ids, and duplicate cells that `Malaria` would not emit."""
    path = tmp_path / "mapping.csv"
    path.write_text(content, encoding="utf-8")
    # Malformed lists, empty ids, non-string filenames, and conflicting ids are rejected.
    with pytest.raises(ValueError, match=message):
        Malaria._read_patient_mapping(path)


def test_malaria_rejects_missing_patient_mapping(malaria_assets):
    """Fail when a cell image has no patient id in the mapping file for `Malaria`."""
    _write_mapping(malaria_assets["patient_mapping_parasitized"], [])
    # A parasitized crop with no mapping row cannot be assigned a patient id.
    with pytest.raises(ValueError, match="Missing patient mapping.*cell_162.png"):
        list(_builder()._generate_examples(malaria_assets, "train"))


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("unexpected.png", _png_bytes(), "Cannot identify source image"),
        (
            "C33P1thinF_IMG_20150619_114756a_cell_179.png",
            b"broken PNG",
            "Cannot decode Malaria image",
        ),
    ],
)
def test_malaria_rejects_invalid_images(
    tmp_path, malaria_assets, filename, content, message
):
    """Reject cell filenames and image bytes that do not match the `Malaria` archive."""
    archive_path = tmp_path / "invalid.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"cell_images/Parasitized/{filename}", content)
    # Names that are not `Malaria` cell crops, and PNGs that do not decode, are rejected.
    with pytest.raises(ValueError, match=message):
        list(
            _builder()._generate_examples(
                {**malaria_assets, "images": archive_path},
                "train",
            )
        )


@pytest.mark.large
def test_malaria_dataset(tmp_path):
    """Test the `Malaria` dataset.

    This test is large and should be skipped in the CI.
    """
    ds_all = StableDatasetDict(
        Malaria(
            processed_cache_dir=tmp_path / "processed",
            download_dir=tmp_path / "downloads",
        )
    )
    # The data source publishes no official train/test split.
    # The patient-mapping CSVs are metadata, so loading every split exposes only "train".
    assert list(ds_all) == [
        "train"
    ], f"Expected only a train split, got {list(ds_all)}."
    ds = ds_all["train"]
    # The cell-image archive contains 27,558 red blood cell crops.
    assert len(ds) == 27558, f"Expected 27558 cell crops, got {len(ds)}."
    # ClassLabel stores names in index order: 0 is parasitized, 1 is uninfected.
    assert ds.features["label"].names == [
        "parasitized",
        "uninfected",
    ], f"Expected label names ['parasitized', 'uninfected'], got {ds.features['label'].names}."

    counts = Counter()
    filenames = set()
    sample_indices = {}
    for index, row in enumerate(ds.with_format("raw")):
        counts[row["label"]] += 1
        filenames.add(row["filename"])
        sample_indices.setdefault(row["label"], index)
        # Raw rows keep the encoded image bytes rather than a decoded PIL image.
        assert isinstance(
            row["image"], bytes
        ), f"Raw image at index {index} should be bytes, got {type(row['image'])}."
        # Filename, patient ID, and source microscopy ID are required non-empty strings.
        metadata = {
            key: row[key] for key in ("filename", "patient_id", "source_image_id")
        }
        assert all(
            isinstance(row[key], str) and row[key] for key in metadata
        ), f"Row {index} has missing string metadata: {metadata}."
    # The release is balanced: 13,779 cells in each class, using the label indices above.
    assert counts == {
        0: 13779,
        1: 13779,
    }, f"Expected 13779 examples per class, got {dict(counts)}."
    # Every cell crop has its own filename, so the filename set matches the row count.
    assert len(filenames) == len(
        ds
    ), f"Expected {len(ds)} unique filenames, got {len(filenames)}."

    # Decode one example from each class. Crops have variable resolution, so only
    # RGB mode and a positive size are required.
    for index in sample_indices.values():
        image = ds[index]["image"]
        assert isinstance(
            image, Image.Image
        ), f"Decoded image at index {index} should be a PIL image, got {type(image)}."
        mode_message = (
            f"Decoded image at index {index} should be RGB, got {image.mode}."
        )
        assert image.mode == "RGB", mode_message
        assert (
            image.width > 0 and image.height > 0
        ), f"Decoded image at index {index} should have a positive size, got {image.size}."
