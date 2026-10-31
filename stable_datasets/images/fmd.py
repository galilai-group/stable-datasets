"""Flickr Material Database with the released region masks preserved losslessly."""

import io
import stat
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


FMD_CLASS_NAMES = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
_CLASS_TO_LABEL = {name: index for index, name in enumerate(FMD_CLASS_NAMES)}
_ANCILLARY_NAMES = {"normalizeImage.m", "normalizeImage.asv", "Thumbs.db", ".DS_Store"}


def _index_members(archive):
    """Build exact image/mask pairs without extracting anything to disk."""
    indexed = {"image": {}, "mask": {}}
    seen = set()
    for member in archive.infolist():
        # orig_filename retains a NUL that ZipInfo.filename would truncate.
        name = member.orig_filename
        parts = name.removesuffix("/").split("/") if member.is_dir() else name.split("/")
        if (
            not name
            or "\\" in name
            or ":" in name
            or any(ord(char) < 32 for char in name)
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise ValueError(f"FMD archive member {name!r}: unsafe path")
        if name in seen:
            raise ValueError(f"FMD archive: duplicate member {name!r}")
        seen.add(name)
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"FMD archive member {name!r}: links and special entries are not allowed")
        if member.is_dir():
            continue
        # Only the two audited roots contain samples. Metadata, thumbnails and
        # platform files outside them never become classification examples.
        if parts[0] == "__MACOSX" or parts[-1] in _ANCILLARY_NAMES or parts[0] not in indexed:
            continue
        if len(parts) != 3 or PurePosixPath(parts[-1]).suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"FMD archive member {name!r}: expected {parts[0]}/class/image.jpg")
        if parts[1] not in _CLASS_TO_LABEL:
            raise ValueError(f"FMD archive member {name!r}: unknown class {parts[1]!r}")
        image_id = "/".join(parts[1:])
        indexed[parts[0]][image_id] = member
    photos, masks = indexed["image"], indexed["mask"]
    if not photos:
        raise ValueError("FMD archive contains no photos below image/<class>/")
    missing = photos.keys() - masks.keys()
    if missing:
        raise FileNotFoundError(f"FMD archive: missing masks for {sorted(missing)[:5]}")
    orphaned = masks.keys() - photos.keys()
    if orphaned:
        raise ValueError(f"FMD archive: masks without matching photos: {sorted(orphaned)[:5]}")
    return photos, masks


def _decode_member(archive, member, *, is_mask):
    field = "mask" if is_mask else "image"
    try:
        with archive.open(member) as stream, PILImage.open(stream) as source:
            source.load()
            if is_mask:
                if source.mode not in ("L", "RGB"):
                    raise ValueError(f"unsupported mask mode {source.mode!r}; expected original L or RGB")
                return source.copy()
            return source.convert("RGB")
    except (OSError, ValueError, zipfile.BadZipFile, PILImage.DecompressionBombError) as exc:
        raise ValueError(f"FMD cannot decode {field} {member.filename!r}: {exc}") from exc


def _png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FMD(BaseDatasetBuilder):
    """Load all 1,000 image/mask pairs in the unsplit ``train`` container.

    The released JPEG masks decode as L or RGB and need not be binary. Their
    original decoded mode, size, and pixels are preserved as PNG, without
    grayscale conversion, thresholding, or resizing. Classification uses image
    and label only; experimental train/validation/test splits belong downstream.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://people.csail.mit.edu/lavanya/fmd.html",
        citation="""@article{sharan2014accuracy,
  title={Accuracy and speed of material categorization in real-world images},
  author={Sharan, Lavanya and Rosenholtz, Ruth and Adelson, Edward H.},
  journal={Journal of Vision},
  volume={14},
  number={9},
  pages={12},
  year={2014},
  doi={10.1167/14.9.12}
}""",
        license="See the source page for Creative Commons terms, image credits, and differently licensed exceptions.",
        assets={
            "train": DownloadInfo(
                url="https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip",
                # Locally computed fingerprint of the release retrieved on
                # 2026-09-30, not a checksum published by the dataset authors.
                checksum="sha256:254475ddb37bf0fe1383ff9ea61eb71042db33af1f9a9aa8747391a72135b8f8",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Flickr Material Database: 1,000 photographs in ten material classes, 100 per class, "
                "with aligned original RGB/L region masks. The train container is not an official training split."
            ),
            features=Features(
                {
                    "image": ImageFeature(encode_format="PNG"),
                    "label": ClassLabel(names=FMD_CLASS_NAMES),
                    "image_id": Value("string"),
                    "mask": ImageFeature(encode_format="PNG"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            photos, masks = _index_members(archive)
            for image_id in sorted(photos):
                image = _decode_member(archive, photos[image_id], is_mask=False)
                mask = _decode_member(archive, masks[image_id], is_mask=True)
                if mask.size != image.size:
                    raise ValueError(
                        f"FMD {image_id!r}: image/mask size mismatch: image={image.size}, mask={mask.size}"
                    )
                # Explicit PNG bytes bypass the Image codec's source-format
                # shortcut, so neither field can be encoded as JPEG again.
                yield (
                    image_id,
                    {
                        "image": _png_bytes(image),
                        "label": _CLASS_TO_LABEL[image_id.split("/")[0]],
                        "image_id": image_id,
                        "mask": _png_bytes(mask),
                    },
                )
