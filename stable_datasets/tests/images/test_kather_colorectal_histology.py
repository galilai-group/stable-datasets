"""Offline fixtures and full-download validation for Kather histology tiles."""

# Tests exercise builder internals as well as the public dataset API.
# pylint: disable=protected-access

import io
from collections import Counter
from zipfile import ZipFile

import pytest
from PIL import Image

from stable_datasets import StableDataset, StableDatasetDict
from stable_datasets.images import KatherColorectalHistology
from stable_datasets.schema import ClassLabel, DatasetSource, DownloadInfo


LABELS = [
    "tumor",
    "stroma",
    "complex",
    "lympho",
    "debris",
    "mucosa",
    "adipose",
    "empty",
]
ROOT = "Kather_texture_2016_image_tiles_5000"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _tiff_bytes(mode="RGB", size=(150, 150)):
    """Create a non-uniform TIFF so pixel preservation is observable."""
    color = {"RGB": (17, 35, 91), "L": 73, "RGBA": (17, 35, 91, 127)}[mode]
    with Image.new(mode, size, color=color) as image:
        image.putpixel(
            (0, 0), {"RGB": (255, 0, 0), "L": 19, "RGBA": (255, 0, 0, 42)}[mode]
        )
        buffer = io.BytesIO()
        image.save(buffer, format="TIFF")
        return buffer.getvalue()


def _builder():
    """Initialize a builder without triggering downloads in __new__."""
    builder = object.__new__(KatherColorectalHistology)
    KatherColorectalHistology.__init__(builder)
    return builder


@pytest.fixture(name="kather_archive")
def kather_archive_path(tmp_path):
    """Include every class, filename variant, and several irrelevant members."""
    archive_path = tmp_path / "tiles.zip"
    with ZipFile(archive_path, "w") as archive:
        for index, label in reversed(list(enumerate(LABELS, start=1))):
            source_id = ("CRC-Prim-HE-03_009", "CRC-Prim-HE-02_003b", "CRC-Prim-HE-07")[
                (index - 1) % 3
            ]
            filename = f"{10000 + index}_{source_id}.tif_Row_301_Col_151.tif"
            mode = ("RGB", "L", "RGBA")[(index - 1) % 3]
            archive.writestr(
                f"{ROOT}/{index:02d}_{label.upper()}/{filename}", _tiff_bytes(mode)
            )
        archive.writestr(f"{ROOT}/01_TUMOR/", b"")
        archive.writestr(f"{ROOT}/01_TUMOR/Thumbs.db", b"not an image")
        archive.writestr(f"{ROOT}/01_TUMOR/readme.txt", b"not an image")
        archive.writestr(f"{ROOT}/09_UNKNOWN/ignored.tif", b"unknown class")
        archive.writestr(f"{ROOT}/01_TUMOR/nested/ignored.tif", b"unexpected nesting")
        archive.writestr("other/01_TUMOR/ignored.tif", b"outside the dataset")
        archive.writestr(f"__MACOSX/{ROOT}/01_TUMOR/._ignored.tif", b"resource fork")
    return archive_path


def test_kather_fixture_loads_and_reuses_cache(tmp_path, monkeypatch, kather_archive):
    """Check download specs, encoded schema, train-only loading, and cache reuse."""
    cache_dir = tmp_path / "processed"
    download_dir = tmp_path / "downloads"
    calls = []

    def fake_bulk_download(specs, dest_folder):
        source = KatherColorectalHistology.SOURCE
        # Provenance and assets are exposed through the typed DatasetSource metadata.
        assert isinstance(source, DatasetSource)
        # The builder requests exactly the assets declared by its source.
        assert specs == list(source.assets.values())
        # Only the tile archive is downloaded, not the separate larger-image archive.
        assert len(specs) == 1
        # The archive request carries download metadata rather than a bare URL.
        assert isinstance(specs[0], DownloadInfo)
        # The download spec pins Zenodo's published MD5 digest for the tile archive.
        assert specs[0].checksum == "md5:0ddbebfc56344752028fda72602aaade"
        # The URL targets the 5,000-tile archive rather than another Zenodo asset.
        assert specs[0].url.endswith("/Kather_texture_2016_image_tiles_5000.zip")
        # Raw files use the caller's download directory, not the default cache.
        assert dest_folder == download_dir
        calls.append(specs)
        return [kather_archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", fake_bulk_download)
    datasets = StableDatasetDict(
        KatherColorectalHistology(
            processed_cache_dir=cache_dir, download_dir=download_dir
        )
    )
    # The release has no official test or validation split, so only train is exposed.
    assert list(datasets) == ["train"]
    dataset = datasets["train"]
    # All eight fixture tiles survive; directories, sidecars, and unrelated paths are skipped.
    assert len(dataset) == 8
    # Preparing the train split requests the archive only once.
    assert len(calls) == 1
    # Tissue labels use ClassLabel encoding rather than unrestricted strings.
    assert isinstance(dataset.features["label"], ClassLabel)
    # Class indices follow the fixed tumor-through-empty label order.
    assert dataset.features["label"].names == LABELS
    # The label feature exposes exactly eight tissue categories.
    assert dataset.features["label"].num_classes == 8
    # The supervised task maps a tile image to its tissue label.
    assert dataset.info.supervised_keys == ("image", "label")
    # Dataset metadata retains the source's attribution license.
    assert dataset.info.license == "CC BY 4.0"
    # The homepage points to the original Zenodo dataset record.
    assert dataset.info.homepage == "https://zenodo.org/records/53169"
    # The citation identifies the original Kather texture-analysis paper.
    assert "10.1038/srep27988" in dataset.info.citation
    rows = list(dataset.with_format("raw"))
    # Sorting archive paths emits one encoded class index per tile in class-directory order.
    assert [row["label"] for row in rows] == list(range(8))
    # Source IDs retain numbered and letter-suffixed image names, including names without a suffix.
    assert [row["source_image_id"] for row in rows[:3]] == [
        "CRC-Prim-HE-03_009",
        "CRC-Prim-HE-02_003b",
        "CRC-Prim-HE-07",
    ]
    # Slide groups keep the numeric CRC-Prim-HE prefix instead of the source-image suffix.
    assert [row["slide_id"] for row in rows[:3]] == [
        "CRC-Prim-HE-03",
        "CRC-Prim-HE-02",
        "CRC-Prim-HE-07",
    ]
    # A source group can occur in different tissue classes; don't include the class in its ID.
    assert rows[0]["source_image_id"] == rows[3]["source_image_id"]
    for row in rows:
        # Every example includes the image, label, original filename, and both grouping IDs.
        assert set(row) == {"image", "label", "filename", "source_image_id", "slide_id"}
        # Filename metadata preserves the original TIFF name after the image is re-encoded.
        assert row["filename"].endswith(".tif")
        # Raw image bytes contain a PNG, not the source TIFF encoding.
        assert row["image"].startswith(PNG_SIGNATURE)
    for row in dataset:
        image = row["image"]
        # Normal dataset access decodes the stored bytes to a PIL image.
        assert isinstance(image, Image.Image)
        # RGB, grayscale, and RGBA fixture inputs all decode as RGB tiles.
        assert image.mode == "RGB"
        # The conversion preserves the release's 150x150 tile dimensions.
        assert image.size == (150, 150)

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("Cached Kather tiles must not be downloaded again")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", unexpected_download)
    cached = StableDatasetDict(
        KatherColorectalHistology(
            processed_cache_dir=cache_dir, download_dir=download_dir
        )
    )
    # Loading the processed cache still exposes only the train split.
    assert list(cached) == ["train"]
    # Cached bytes, labels, metadata, and ordering match the original build without downloading.
    assert list(cached["train"].with_format("raw")) == rows
    # Selecting train directly returns a single StableDataset rather than a split mapping.
    assert isinstance(
        KatherColorectalHistology(
            split="train", processed_cache_dir=cache_dir, download_dir=download_dir
        ),
        StableDataset,
    )
    for split in ("test", "validation"):
        # Test and validation requests fail instead of creating synthetic splits.
        with pytest.raises(ValueError, match="not found"):
            KatherColorectalHistology(
                split=split, processed_cache_dir=cache_dir, download_dir=download_dir
            )


def test_kather_png_preserves_pixels(kather_archive):
    """RGB, grayscale, and RGBA TIFFs retain their RGB pixels without resizing."""
    with ZipFile(kather_archive) as archive:
        for key, example in _builder()._generate_examples(
            data_path=kather_archive, split="train"
        ):
            with (
                Image.open(io.BytesIO(archive.read(key))) as original,
                Image.open(io.BytesIO(example["image"])) as encoded,
            ):
                # The conversion fixture really starts with TIFF data.
                assert original.format == "TIFF"
                # The yielded bytes are encoded as PNG rather than merely renamed TIFFs.
                assert encoded.format == "PNG"
                # Conversion normalizes grayscale and RGBA inputs to three-channel RGB.
                assert encoded.mode == "RGB"
                # PNG dimensions exactly match the original 150x150 TIFF, with no resizing.
                assert encoded.size == original.size == (150, 150)
                # Every RGB pixel, including the distinct corner pixel, survives losslessly.
                assert encoded.tobytes() == original.convert("RGB").tobytes()


def test_kather_requires_archive_path():
    """A missing archive path must fail explicitly."""
    # Calling the generator without an archive reports the missing argument clearly.
    with pytest.raises(TypeError, match="data_path is required"):
        list(_builder()._generate_examples())


@pytest.mark.parametrize(
    ("source_image_id", "slide_id"),
    [
        ("CRC-Prim-HE-01b", "CRC-Prim-HE-01"),
        ("CRC-Prim-HE-10c", "CRC-Prim-HE-10"),
        ("CRC-Prim-HE-02_copy", "CRC-Prim-HE-02"),
        ("CRC-Prim-HE-07_001_copy", "CRC-Prim-HE-07"),
        ("CRC-Prim-HE-10_002c", "CRC-Prim-HE-10"),
    ],
)
def test_kather_source_filename_variants(tmp_path, source_image_id, slide_id):
    """Preserve source suffixes while grouping by the numeric slide prefix."""
    filename = f"10647_{source_image_id}.tif_Row_1_Col_1.tif"
    archive_path = tmp_path / "source_variant.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{ROOT}/01_TUMOR/{filename}", _tiff_bytes())
    examples = list(
        _builder()._generate_examples(data_path=archive_path, split="train")
    )
    # Each supported filename variant yields exactly one tile instead of being skipped.
    assert len(examples) == 1
    _, example = examples[0]
    # Metadata retains the complete original tile filename, including its source suffix.
    assert example["filename"] == filename
    # Source IDs preserve letter and copy suffixes rather than merging distinct source names.
    assert example["source_image_id"] == source_image_id
    # Related source-name variants share the expected numeric slide-prefix group.
    assert example["slide_id"] == slide_id


def test_kather_order_is_independent_of_zip_order(tmp_path, kather_archive):
    """ZIP insertion order must not affect keys, encoded pixels, or metadata."""
    reversed_path = tmp_path / "reversed.zip"
    with ZipFile(kather_archive) as source, ZipFile(reversed_path, "w") as target:
        for member in reversed(source.infolist()):
            target.writestr(member, source.read(member))
    original = list(
        _builder()._generate_examples(data_path=kather_archive, split="train")
    )
    reversed_examples = list(
        _builder()._generate_examples(data_path=reversed_path, split="train")
    )
    # Reversing ZIP insertion order leaves example keys, PNG bytes, labels, and metadata unchanged.
    assert original == reversed_examples
    # Examples are emitted in sorted archive-path order, independent of ZIP insertion order.
    assert [key for key, _ in original] == sorted(key for key, _ in original)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("invalid.tif", _tiff_bytes(), "Cannot identify source image"),
        (
            "10009_CRC-Prim-HE-03_009.tif_Row_301_Col_151.tif",
            b"broken TIFF",
            "Cannot decode Kather image",
        ),
        (
            "10009_CRC-Prim-HE-03_009.tif_Row_301_Col_151.tif",
            _tiff_bytes(size=(149, 150)),
            "Expected a 150x150 Kather tile",
        ),
    ],
    ids=["invalid-filename", "corrupt-tiff", "wrong-dimensions"],
)
def test_kather_rejects_invalid_tiles(tmp_path, filename, content, message):
    """Reject invalid tile names, corrupt images, and unexpected dimensions."""
    archive_path = tmp_path / "invalid.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{ROOT}/01_TUMOR/{filename}", content)
    # Bad names, corrupt TIFFs, and wrong-sized tiles each raise their expected validation error.
    with pytest.raises(ValueError, match=message):
        list(_builder()._generate_examples(data_path=archive_path, split="train"))


@pytest.mark.large
def test_kather_colorectal_histology_dataset(tmp_path):
    """Validate all 5,000 real tiles; requires downloading the 246 MiB archive."""
    datasets = StableDatasetDict(
        KatherColorectalHistology(
            processed_cache_dir=tmp_path / "processed",
            download_dir=tmp_path / "downloads",
        )
    )
    # The real release exposes train only, without invented test or validation splits.
    assert list(datasets) == ["train"]
    dataset = datasets["train"]
    # The complete tile archive contains exactly 5,000 examples.
    assert len(dataset) == 5000
    # The real dataset uses the same categorical label encoding as the fixtures.
    assert isinstance(dataset.features["label"], ClassLabel)
    # The released classes retain their fixed names and index order.
    assert dataset.features["label"].names == LABELS
    # All eight tissue categories are represented by the label feature.
    assert dataset.features["label"].num_classes == 8

    counts = Counter()
    filenames = set()
    sample_indices = {}
    # Count and inspect metadata using raw rows, not feature-level image decoding.
    for index, row in enumerate(dataset.with_format("raw")):
        counts[row["label"]] += 1
        filenames.add(row["filename"])
        sample_indices.setdefault(row["label"], index)
        # Every tile has non-empty string metadata for its filename, source image, and slide group.
        assert all(
            isinstance(row[key], str) and row[key]
            for key in ("filename", "source_image_id", "slide_id")
        )
        # Original filenames remain TIFF names even though stored image bytes are PNGs.
        assert row["filename"].endswith(".tif")
        # Each source-image ID belongs to the numeric prefix recorded as its slide group.
        assert row["source_image_id"].startswith(row["slide_id"])
        # The source-image ID is taken from the tile filename, not fabricated independently.
        assert f"_{row['source_image_id']}.tif_Row_" in row["filename"]
        # Every stored image starts with the PNG signature rather than TIFF bytes.
        assert row["image"].startswith(PNG_SIGNATURE)
        # Decode every tile to verify its pixels are readable, RGB, and unresized.
        with Image.open(io.BytesIO(row["image"])) as image:
            image.load()
            # Pillow recognizes the stored bytes as a real PNG image.
            assert image.format == "PNG"
            # Every released tile is stored with three RGB channels.
            assert image.mode == "RGB"
            # Every released tile retains its 150x150 dimensions.
            assert image.size == (150, 150)
    # Raw label counts confirm the balanced release: 625 tiles for each of eight class indices.
    assert counts == dict.fromkeys(range(8), 625)
    # All 5,000 examples retain distinct original tile filenames.
    assert len(filenames) == 5000
    for index in sample_indices.values():
        image = dataset[index]["image"]
        # Feature-level decoding returns a PIL image for a sample from every tissue class.
        assert isinstance(image, Image.Image)
        # The public decoding path preserves the stored RGB mode.
        assert image.mode == "RGB"
        # The public decoding path also preserves the original tile dimensions.
        assert image.size == (150, 150)
