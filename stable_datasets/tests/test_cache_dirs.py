"""Tests that the CI cache-cleanup fixture targets the directories actually used.

The autouse fixture in ``conftest.py`` exists to reclaim disk between tests on CI
runners. It previously hard-coded ``~/.stable_datasets`` (underscore) while the
library writes to ``~/.stable-datasets`` (hyphen), so it silently deleted nothing
and runners accumulated every dataset downloaded during a session.
"""

import importlib.util
import os
from pathlib import Path

from stable_datasets.utils import CACHE_DIR_ENV_VAR, _default_dest_folder, _default_processed_cache_dir


def _load_conftest():
    """Import conftest by path: ``stable_datasets/tests`` is not a package."""
    spec = importlib.util.spec_from_file_location("_sd_conftest", Path(__file__).parent / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stable_datasets_cache_dirs = _load_conftest()._stable_datasets_cache_dirs


def test_cleanup_targets_the_real_default_cache_dirs(monkeypatch):
    monkeypatch.delenv(CACHE_DIR_ENV_VAR, raising=False)

    targets = {Path(p) for p in _stable_datasets_cache_dirs()}

    assert Path(_default_dest_folder()) in targets, "The raw download directory must be reclaimed."
    assert Path(_default_processed_cache_dir()) in targets, "The processed cache directory must be reclaimed."


def test_cleanup_follows_the_cache_dir_env_override(monkeypatch, tmp_path):
    """STABLE_DATASETS_CACHE_DIR relocates the cache, so cleanup must follow it."""
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path))

    targets = {Path(p) for p in _stable_datasets_cache_dirs()}

    assert Path(_default_dest_folder()) in targets
    assert Path(_default_processed_cache_dir()) in targets
    assert all(str(target).startswith(str(tmp_path)) for target in targets)


def test_shared_download_dir_is_outside_the_reclaimed_tree(shared_download_dir, monkeypatch):
    """The module-scoped download dir must survive the per-test cleanup.

    Otherwise every test in a module re-fetches the same archive.
    """
    monkeypatch.delenv(CACHE_DIR_ENV_VAR, raising=False)

    shared = Path(shared_download_dir).resolve()
    for target in _stable_datasets_cache_dirs():
        resolved = Path(os.path.expanduser(str(target))).resolve()
        assert not shared.is_relative_to(resolved), f"{shared} would be wiped by the cleanup of {resolved}."
