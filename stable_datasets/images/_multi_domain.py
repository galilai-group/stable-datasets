"""Helpers shared by the multi-domain image datasets (PACS, Office-Home).

Every image in these datasets has a class label and a *domain* (for example
``photo`` vs. ``sketch``), and the same classes appear in every domain.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence

from PIL import Image as PILImage

from stable_datasets.schema import BuilderConfig, ClassLabel, Features
from stable_datasets.schema import Image as ImageFeature


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def google_drive_url(file_id: str, resource_key: str | None = None) -> str:
    """Direct download URL for a public Google Drive file.

    ``confirm=t`` skips the virus-scan page Drive shows for large files, and
    ``resource_key`` is needed for files shared before Drive's 2021 security update.
    """
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return f"{url}&resourcekey={resource_key}" if resource_key else url


def normalize_name(name: str) -> str:
    """Map a folder name to a snake_case identifier: ``"Real World"`` -> ``"real_world"``."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def multi_domain_features(class_names: Sequence[str], domains: Sequence[str]) -> Features:
    """Schema shared by the multi-domain datasets: an RGB image, its class and its domain."""
    return Features(
        {
            "image": ImageFeature(),
            "label": ClassLabel(names=list(class_names)),
            "domain": ClassLabel(names=list(domains)),
        }
    )


def leave_one_domain_out_configs(dataset_name: str, domains: Sequence[str], description: str) -> list[BuilderConfig]:
    """One config per domain, named after the domain held out for testing."""
    return [
        BuilderConfig(name=domain, description=description.format(dataset=dataset_name, domain=domain))
        for domain in domains
    ]


def image_members(archive: zipfile.ZipFile, domains: Iterable[str]) -> dict[str, str]:
    """Map ``"<domain>/<class>/<file>"`` to the archive member holding that image.

    The domain folder is matched after :func:`normalize_name`, so the archive's top-level
    folder and spelling (``Real World`` vs. ``real_world``) don't matter.
    """
    domains = set(domains)
    members = {}
    for name in archive.namelist():
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        parts = name.split("/")
        for i, part in enumerate(parts[:-2]):
            if normalize_name(part) in domains:
                members["/".join([normalize_name(part), *parts[i + 1 :]])] = name
                break
    return members


def ensure_rgb(image_bytes: bytes) -> bytes:
    """Return the bytes unchanged if the image is already RGB, otherwise an RGB PNG.

    Transparent pixels are composited onto white, so transparent sketches don't turn black.
    """
    with PILImage.open(io.BytesIO(image_bytes)) as img:
        if img.mode == "RGB":
            return image_bytes
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            rgb = PILImage.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return buf.getvalue()


def num_test_examples(group_size: int, test_fraction: float) -> int:
    """Test examples for one (domain, class) cell, keeping at least one on each side when possible."""
    if group_size < 2:
        return 0
    return min(max(1, round(group_size * test_fraction)), group_size - 1)


def stratified_test_keys(items: Iterable[tuple[str, bytes, tuple]], test_fraction: float) -> set[str]:
    """Pick test keys stratified by group, from ``(key, image_bytes, group)`` items.

    Images are ranked by a SHA-256 of their bytes, so the split is reproducible and does
    not depend on file order.
    """
    groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for key, image_bytes, group in items:
        groups[group].append((hashlib.sha256(image_bytes).hexdigest(), key))
    test_keys = set()
    for members in groups.values():
        members.sort()
        test_keys.update(key for _, key in members[: num_test_examples(len(members), test_fraction)])
    return test_keys
