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
    """Default cache dirs used by dataset builders (downloads + processed).

    Asks the library, so the cleanup follows ``STABLE_DATASETS_CACHE_DIR`` when it is set.
    """
    from stable_datasets.utils import _default_dest_folder, _default_processed_cache_dir

    return [_default_dest_folder(), _default_processed_cache_dir()]


# Full CI runs on PRs and main set this, so a dataset host being down doesn't fail a change that
# didn't touch that dataset. The nightly run and PRs that only change datasets leave it unset, so
# an unreachable host still fails there.
SKIP_UNREACHABLE_HOSTS = os.environ.get("STABLE_DATASETS_SKIP_UNREACHABLE_HOSTS") == "1"
_unreachable_host_skips = []  # (test id, error message)


def _host_unreachable_error(error):
    """The ``HostUnreachableError`` in *error*'s cause chain, if any."""
    from stable_datasets.utils import HostUnreachableError

    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, HostUnreachableError):
            return error
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a test whose dataset host can't be reached as skipped, when SKIP_UNREACHABLE_HOSTS is on."""
    outcome = yield
    report = outcome.get_result()
    if not SKIP_UNREACHABLE_HOSTS or report.when not in ("setup", "call") or not report.failed:
        return
    error = _host_unreachable_error(call.excinfo.value) if call.excinfo else None
    if error is None:
        return
    path, lineno, _ = item.location
    report.outcome = "skipped"
    report.longrepr = (path, (lineno or 0) + 1, f"Skipped: dataset host unreachable: {error}")
    _unreachable_host_skips.append((item.nodeid, str(error)))


def pytest_terminal_summary(terminalreporter):
    """List the tests skipped for an unreachable host, also in the GitHub Actions job summary."""
    if not _unreachable_host_skips:
        return
    terminalreporter.section("skipped: dataset host unreachable")
    for nodeid, message in _unreachable_host_skips:
        terminalreporter.line(f"{nodeid}: {message}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        lines = [
            f"**{len(_unreachable_host_skips)} tests skipped because their dataset host was unreachable**",
            "",
            "| Test | Error |",
            "|---|---|",
        ]
        for nodeid, message in _unreachable_host_skips:
            message = message.replace("|", "\\|")
            lines.append(f"| `{nodeid}` | {message} |")
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")


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
