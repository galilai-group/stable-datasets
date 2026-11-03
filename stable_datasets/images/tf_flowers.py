"""TensorFlow Flowers with original image labels and photographer attribution."""

import gzip
import hashlib
import io
import re
import tarfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Preserve the TFDS label order, which is not alphabetical.
TFFLOWERS_CLASS_NAMES = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]
_CLASS_TO_LABEL = {name: index for index, name in enumerate(TFFLOWERS_CLASS_NAMES)}
_LICENSE_DECLARATION = (
    "All images in this archive are licensed under the Creative Commons By-Attribution License, available at:\n"
    "https://creativecommons.org/licenses/by/2.0/\n"
    "The photographers are listed below, thanks to all of them for making their work available, "
    "and please be sure to credit them for any use as per the license."
)
_CREDIT_BODY = re.compile(r"(?:by .+ - https?://\S+|https?://\S+ - by .+)")


def _verify_archive_checksum(archive_path, checksum):
    digest = hashlib.sha256()
    with open(archive_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if checksum != "sha256:" + digest.hexdigest():
        raise ValueError(f"TFFlowers archive checksum mismatch: expected {checksum}, got sha256:{digest.hexdigest()}")


def _path_parts(name, *, context):
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 for char in name)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"TFFlowers {context}: unsafe path {name!r}")
    return parts


def _image_id(parts, *, context):
    if not parts or parts[0] not in _CLASS_TO_LABEL:
        raise ValueError(f"TFFlowers {context}: unknown class")
    if len(parts) != 2 or PurePosixPath(parts[1]).suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError(f"TFFlowers {context}: expected class/image.jpg")
    return "/".join(parts)


def _classify_member(member):
    """Return a canonical member name and optional image ID; never extract."""
    context = f"archive member {member.name!r}"
    name = member.name.removeprefix("./")
    if member.isdir():
        name = name.removesuffix("/")
    parts = _path_parts(name, context=context)
    if parts[0] != "flower_photos":
        raise ValueError(f"TFFlowers {context}: expected flower_photos archive prefix")
    if member.issym() or member.islnk():
        raise ValueError(f"TFFlowers {context}: archive links are not allowed")
    if member.isdir():
        if len(parts) > 2 or (len(parts) == 2 and parts[1] not in _CLASS_TO_LABEL):
            raise ValueError(f"TFFlowers {context}: unsupported archive directory")
        return name, None
    if not member.isfile():
        raise ValueError(f"TFFlowers {context}: unsupported archive entry type")
    if name == "flower_photos/LICENSE.txt":
        return name, None
    if len(parts) == 2 and PurePosixPath(parts[-1]).suffix.lower() not in (".jpg", ".jpeg"):
        # Root metadata never becomes an image sample.
        return name, None
    if len(parts) == 3 and parts[1] in _CLASS_TO_LABEL and parts[-1] in (".DS_Store", "Thumbs.db"):
        return name, None
    return name, _image_id(parts[1:], context=context)


def _read_credits(content):
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("TFFlowers LICENSE.txt is not UTF-8") from exc
    if "\n".join(lines[:3]) != _LICENSE_DECLARATION:
        raise ValueError("TFFlowers LICENSE.txt: unexpected license declaration")
    credits = {}
    for line_number, line in enumerate(lines[3:], start=4):
        if not line:
            continue
        context = f"LICENSE.txt:{line_number}"
        image_id, separator, body = line.partition(" CC-BY ")
        if not separator or _CREDIT_BODY.fullmatch(body) is None:
            raise ValueError(f"TFFlowers {context}: invalid photographer attribution")
        image_id = _image_id(_path_parts(image_id, context=context), context=context)
        if image_id in credits:
            raise ValueError(f"TFFlowers {context}: duplicate attribution for {image_id!r}")
        # Retain the exact line, including author spacing, Unicode, and the
        # original photo or profile URL. Do not manufacture missing photo URLs.
        credits[image_id] = line
    return credits


def _read_archive_metadata(archive_path):
    seen, photos = set(), set()
    credits = None
    # LICENSE.txt occurs in the middle of the published tar. A metadata pass
    # avoids retaining all preceding full-resolution photos in memory.
    with gzip.open(archive_path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                name, image_id = _classify_member(member)
                if name in seen:
                    raise ValueError(f"TFFlowers archive: duplicate canonical member {name!r}")
                seen.add(name)
                if image_id is not None:
                    photos.add(image_id)
                if name == "flower_photos/LICENSE.txt" and member.isfile():
                    with archive.extractfile(member) as content:
                        credits = _read_credits(content.read())
        # Tar stops at its end markers; drain gzip too to validate its trailer.
        while stream.read(1024 * 1024):
            pass
    if credits is None:
        raise FileNotFoundError("TFFlowers archive is missing flower_photos/LICENSE.txt")
    if not photos:
        raise ValueError("TFFlowers archive contains no photos")
    missing = photos - credits.keys()
    if missing:
        raise ValueError(f"TFFlowers LICENSE.txt: missing attribution for {sorted(missing)[:5]}")
    orphaned = credits.keys() - photos
    if orphaned:
        raise ValueError(f"TFFlowers LICENSE.txt: attribution without a photo: {sorted(orphaned)[:5]}")
    return credits


class TFFlowers(BaseDatasetBuilder):
    """Load all 3,670 released photographs in the unsplit ``train`` container.

    Original labels and duplicate records are retained. The exact photographer
    attribution line accompanies each sample. The original ``LICENSE.txt``
    remains at ``flower_photos/LICENSE.txt`` inside the downloaded archive;
    its opening declaration is also available as ``dataset.info.license``.
    Evaluation grouping and splits are the caller's responsibility.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://www.tensorflow.org/datasets/catalog/tf_flowers",
        citation=(
            "TensorFlow, TensorFlow Flowers (tf_flowers). "
            "https://www.tensorflow.org/datasets/catalog/tf_flowers. "
            "Individual photographers are credited in the archive's LICENSE.txt."
        ),
        license=_LICENSE_DECLARATION,
        assets={
            "train": DownloadInfo(
                url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
                # Published in the TFDS checksum list at commit
                # 2dff5ebbf41953c3f8eeb892c6354f98ee4ef564 and verified against
                # the complete HTTPS archive during the source preflight.
                checksum="sha256:4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c",
            )
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "TensorFlow Flowers: 3,670 photographs in five classes with original labels and photographer "
                "attribution. The train container is the complete collection, not an official training split."
            ),
            features=Features(
                {
                    "image": ImageFeature(encode_format="PNG"),
                    "label": ClassLabel(names=TFFLOWERS_CLASS_NAMES),
                    "image_id": Value("string"),
                    "attribution": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        # The shared downloader returns completed cache hits without rehashing;
        # every cold native build verifies the raw archive before parsing it.
        _verify_archive_checksum(data_path, self.SOURCE.assets["train"].checksum)
        credits = _read_archive_metadata(data_path)
        seen = set()
        # Second sequential pass follows the fixed tar order and decodes one
        # photograph at a time. No per-image archive seeks or extraction.
        with tarfile.open(data_path, mode="r|gz") as archive:
            for member in archive:
                _, image_id = _classify_member(member)
                if image_id is None:
                    continue
                if image_id in seen or image_id not in credits:
                    raise ValueError(f"TFFlowers archive membership changed between passes: {image_id!r}")
                seen.add(image_id)
                try:
                    with archive.extractfile(member) as stream, PILImage.open(io.BytesIO(stream.read())) as source:
                        image = source.convert("RGB")
                        image.load()
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
                    raise ValueError(f"TFFlowers cannot decode image {image_id!r}: {exc}") from exc
                yield (
                    image_id,
                    {
                        "image": buffer.getvalue(),
                        "label": _CLASS_TO_LABEL[image_id.split("/")[0]],
                        "image_id": image_id,
                        "attribution": credits[image_id],
                    },
                )
        if seen != credits.keys():
            raise ValueError("TFFlowers archive membership changed between passes")
