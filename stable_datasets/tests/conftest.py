"""Pytest configuration file.

This file is automatically loaded by pytest and ensures that the project root
is added to sys.path, allowing tests to import from the stable_datasets package
without needing sys.path.insert in each test file.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest


# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _stable_datasets_cache_dirs():
    """Cache dirs used by dataset builders (downloads + processed).

    Resolved through the library rather than hard-coded: the default is
    ``~/.stable-datasets`` (hyphen) and ``STABLE_DATASETS_CACHE_DIR`` overrides it,
    so spelling the path here by hand silently pointed the cleanup below at a
    directory that never exists.
    """
    from stable_datasets.utils import _get_cache_dir

    base = Path(os.path.expanduser(_get_cache_dir()))
    return [base / "downloads", base / "processed"]


@pytest.fixture(scope="module")
def shared_download_dir(tmp_path_factory, request):
    """A raw-download directory shared by every test in one module.

    The autouse cleanup below wipes the *default* download directory after each
    test, so a test file with several functions would otherwise re-fetch the same
    archive once per test -- MVTec-AD's smallest category archive is ~109 MB and
    MedIAnomaly's smallest subset ~42 MB, so that adds up quickly in CI.

    Anchoring downloads here keeps one copy per module. It sits outside
    ``~/.stable_datasets`` so the cleanup does not touch it, and pytest reclaims
    it with the rest of the session's temporary directories.

    Pass it as ``download_dir=`` alongside a per-test ``processed_cache_dir=``,
    so each test still builds its cache from scratch while reusing the download.
    """
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    return str(tmp_path_factory.mktemp(f"dl_{module_name}_"))


@pytest.fixture(autouse=True, scope="function")
def cleanup_stable_datasets_cache_after_test():
    """Remove stable_datasets cache dirs after each test when running in CI.

    Frees disk space between tests to avoid "No space left on device" when
    many dataset tests run sequentially on runners with limited disk.

    With scope="function", cleanup runs after every test, including each
    parametrized iteration (e.g. between datasets in test_datasets.py).
    """
    yield
    if os.environ.get("CI") != "true" and os.environ.get("GITHUB_ACTIONS") != "true":
        return
    for d in _stable_datasets_cache_dirs():
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
