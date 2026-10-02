"""Tests for the shared anomaly-dataset extraction helpers."""

import os
import shutil
import stat
import tarfile

import pytest

from stable_datasets.images._anomaly_utils import _EXTRACTION_SENTINEL, ensure_extracted, iter_images


def _build_tar(path, entries, *, mode=None):
    """Write a tar containing ``entries`` ({relative path: bytes}) under ``root/``."""
    staging = path.parent / "staging"
    for name, payload in entries.items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    if mode is not None:
        # Apply to files first, then directories bottom-up, or we lock ourselves out.
        for current, dirnames, filenames in os.walk(staging, topdown=False):
            for filename in filenames:
                os.chmod(os.path.join(current, filename), mode)
            for dirname in dirnames:
                os.chmod(os.path.join(current, dirname), mode)

    with tarfile.open(path, "w:gz") as archive:
        archive.add(staging / "root", arcname="root")

    # Restore write access so the staging tree can be cleaned up.
    for current, dirnames, _ in os.walk(staging):
        for dirname in dirnames:
            os.chmod(os.path.join(current, dirname), 0o755)
    shutil.rmtree(staging, ignore_errors=True)


def test_ensure_extracted_extracts_once(tmp_path):
    archive = tmp_path / "data.tar.gz"
    _build_tar(archive, {"root/a.png": b"a", "root/sub/b.png": b"b"})

    dest = tmp_path / "out"
    ensure_extracted(archive, dest)

    assert (dest / "root" / "a.png").read_bytes() == b"a"
    assert (dest / _EXTRACTION_SENTINEL).exists()

    # A second call is a no-op: deleting the archive must not break it.
    archive.unlink()
    assert ensure_extracted(archive, dest) == dest


def test_ensure_extracted_makes_readonly_archives_cleanable(tmp_path):
    """MVTec-AD ships dr-xr-x--- dirs and -r-xr----- files.

    Preserved verbatim those make the extracted cache undeletable, so rmtree
    fails and the CI fixture that reclaims disk between tests silently gives up.
    """
    archive = tmp_path / "readonly.tar.gz"
    _build_tar(archive, {"root/test/defect/000.png": b"x"}, mode=0o550)

    dest = tmp_path / "out"
    ensure_extracted(archive, dest)

    extracted = dest / "root" / "test" / "defect" / "000.png"
    assert extracted.read_bytes() == b"x", "Content must survive the permission fixup."
    assert stat.S_IMODE(extracted.stat().st_mode) & stat.S_IWUSR, "Owner needs write to clean the cache."
    assert stat.S_IMODE((dest / "root").stat().st_mode) & stat.S_IXUSR, "Directories must stay traversable."

    shutil.rmtree(dest)
    assert not dest.exists(), "The extracted cache must be deletable."


def test_ensure_extracted_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_bytes(b"pwned")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="escapes"):
        ensure_extracted(archive, tmp_path / "out")

    assert not (tmp_path / "escaped.txt").exists()


def test_iter_images_is_sorted_and_filtered(tmp_path):
    for name in ["b.png", "a.png", "c.PNG", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "d.png").write_bytes(b"x")

    names = [p.name for p in iter_images(tmp_path)]
    # Sorted, since row order determines cache row order; extension match is
    # case-insensitive; and only the top level is listed.
    assert names == ["a.png", "b.png", "c.PNG"]


def test_iter_images_on_missing_directory(tmp_path):
    assert iter_images(tmp_path / "absent") == []
