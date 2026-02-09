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
    """Default cache dirs used by dataset builders (downloads + processed)."""
    base = Path(os.path.expanduser("~/.stable_datasets"))
    return [base / "downloads", base / "processed"]


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
