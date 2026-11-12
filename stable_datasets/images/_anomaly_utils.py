"""Shared helpers for the anomaly-detection image builders.

MVTec-AD, VisA and MedIAnomaly all ship as archives that contain every split,
so each builder extracts once into the download directory and then walks the
resulting tree twice (once per split). Extraction is guarded by a sentinel file
so the second split -- and any later load that missed the processed cache --
reuses the already-extracted tree instead of decompressing again.
"""

from __future__ import annotations

import stat
import tarfile
from pathlib import Path

from filelock import FileLock
from loguru import logger as logging


_EXTRACTION_SENTINEL = ".extraction_complete"

# Binary anomaly label shared by every dataset in this family. Index 0 is the
# nominal class, matching the "good"/"normal" convention used upstream.
ANOMALY_LABEL_NAMES = ["good", "anomalous"]


def ensure_extracted(archive_path: Path, dest_dir: Path) -> Path:
    """Extract *archive_path* into *dest_dir* once and return *dest_dir*.

    Concurrent builders (the two splits of one dataset, or several SLURM jobs
    sharing a scratch cache) serialize on a FileLock; the first one extracts and
    writes the sentinel, the rest return immediately.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    sentinel = dest_dir / _EXTRACTION_SENTINEL

    if sentinel.exists():
        return dest_dir

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest_dir.with_suffix(dest_dir.suffix + ".extract.lock")

    with FileLock(str(lock_path)):
        # Re-check inside the lock: another process may have finished while we waited.
        if sentinel.exists():
            return dest_dir

        dest_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Extracting {archive_path.name} -> {dest_dir}")
        with tarfile.open(archive_path, "r:*") as tar:
            _safe_extractall(tar, dest_dir)

        _restore_owner_write(dest_dir)
        sentinel.touch()

    return dest_dir


def _restore_owner_write(dest_dir: Path) -> None:
    """Give the owner write permission across the extracted tree.

    MVTec-AD's archives carry read-only modes (``dr-xr-x---`` directories,
    ``-r-xr-----`` files). Preserved verbatim, the extracted cache cannot be
    deleted -- ``shutil.rmtree`` fails, and the CI fixture that reclaims disk
    between tests silently gives up because it passes ``ignore_errors=True``.
    """
    for path in [dest_dir, *dest_dir.rglob("*")]:
        try:
            mode = path.stat().st_mode
            # Directories also need the owner execute bit to stay traversable.
            path.chmod(mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
        except OSError:  # pragma: no cover - best effort on exotic filesystems
            continue


def _safe_extractall(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """Extract *tar* refusing any member that would escape *dest_dir*.

    Guards against path traversal via ``..`` components, absolute paths and
    symlinks pointing outside the destination.
    """
    dest_root = dest_dir.resolve()
    for member in tar.getmembers():
        target = (dest_root / member.name).resolve()
        if not target.is_relative_to(dest_root):
            raise ValueError(f"Refusing to extract {member.name!r}: escapes {dest_root}.")
        if (member.issym() or member.islnk()) and not (dest_root / member.linkname).resolve().is_relative_to(
            dest_root
        ):
            raise ValueError(f"Refusing to extract link {member.name!r}: target escapes {dest_root}.")

    try:
        # Python >= 3.10.12 / 3.11.4 / 3.12 warn without an explicit filter and
        # default to "data" from 3.14 onwards.
        tar.extractall(dest_dir, filter="data")
    except TypeError:
        tar.extractall(dest_dir)


def iter_images(
    directory: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
) -> list[Path]:
    """Return image files directly under *directory*, sorted for a stable row order.

    Row order determines cache row order, so it must not depend on filesystem
    iteration order -- otherwise the same build produces different caches.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
