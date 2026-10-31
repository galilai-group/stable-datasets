"""Decide which tests CI runs for a pull request or a push.

Usage:
    python .github/scripts/select_tests.py --base <sha> [--head <sha>]
    python .github/scripts/select_tests.py --full

Picks one of three modes and, on GitHub Actions, writes ``mode``, ``tests`` and ``shards``
to ``$GITHUB_OUTPUT``:

- ``full``: shared code changed, so run the whole suite split across several jobs.
- ``targeted``: only datasets (or tests, or benchmarks) changed, so run just their tests in one job.
- ``none``: nothing that tests can check changed (docs, examples, ...).
"""

import argparse
import json
import os
import sys
from pathlib import PurePosixPath

from dataset_files import (
    ROOT,
    TESTS_DIR,
    changed_files,
    dataset_module,
    modules_importing,
    tests_for_module,
    tests_importing_package,
)


FULL_SHARDS = 8

# Changing these can affect every test.
FULL_RUN_FILES = {
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    ".github/workflows/testing.yaml",
    ".github/scripts/select_tests.py",
    ".github/scripts/dataset_files.py",
}
# Changing these can't affect any test.
NO_TEST_PREFIXES = ("docs/", "examples/", "profiling/", ".github/")
NO_TEST_FILES = {
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    "LICENSE",
    "generate_teaser.py",
}


def select(changes):
    """Return ``(mode, tests, notes)`` for a list of changed files."""
    tests: set[str] = set()
    full_reasons: list[str] = []
    notes: list[str] = []
    touched_datasets = False

    for change in changes:
        path, deleted = change.path, change.status == "D"
        name = PurePosixPath(path).name

        if path in FULL_RUN_FILES:
            full_reasons.append(f"`{path}` changed")
        elif deleted and path.startswith("stable_datasets/"):
            full_reasons.append(f"`{path}` was deleted")
        elif path.startswith(TESTS_DIR + "/"):
            folder = PurePosixPath(path).parent
            if name.startswith("test_") and name.endswith(".py"):
                tests.add(path)
            elif name == "__init__.py":
                continue
            elif folder.as_posix() == TESTS_DIR:
                full_reasons.append(f"test helper `{path}` changed")
            else:
                # A conftest.py or helper in a subfolder only affects the tests in that subfolder.
                tests.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob("test_*.py"))
        elif (folder_module := dataset_module(path)) is not None:
            folder, module = folder_module
            touched_datasets = True
            if module == "__init__":
                continue
            covered = tests_for_module(folder, module)
            if module.startswith("_"):
                users = modules_importing(folder, module)
                for user in users:
                    covered += tests_for_module(folder, user)
                notes.append(f"`{path}` is a helper used by: {', '.join(users) or 'nothing'}")
            if not covered:
                notes.append(f"no tests found for `{path}`")
            tests.update(covered)
        elif path.startswith("stable_datasets/"):
            full_reasons.append(f"shared code `{path}` changed")
        elif path.startswith("benchmarks/"):
            tests.update(tests_importing_package("benchmarks"))
        elif path.startswith(NO_TEST_PREFIXES) or path in NO_TEST_FILES or ("/" not in path and path.endswith(".md")):
            continue
        else:
            full_reasons.append(f"`{path}` changed (not a known dataset, test or docs path)")

    if full_reasons:
        return "full", [], full_reasons + notes
    tests = sorted(t for t in tests if (ROOT / t).is_file())
    if tests or touched_datasets:
        return "targeted", tests, notes
    return "none", [], notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="commit to diff against")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true", help="skip the diff and run everything")
    args = parser.parse_args()
    if not args.full and not args.base:
        parser.error("pass --base or --full")

    if args.full:
        mode, tests, notes = "full", [], ["full run requested"]
    else:
        mode, tests, notes = select(changed_files(args.base, args.head))
    shards = list(range(1, FULL_SHARDS + 1)) if mode == "full" else [1]

    lines = [f"**Test selection: `{mode}`**", ""]
    lines += [f"- {n}" for n in notes]
    if mode == "targeted":
        lines += ["", "Test files:" if tests else "No test files; only checking that the package imports."]
        lines += [f"- `{t}`" for t in tests]
    report = "\n".join(lines)
    print(report)

    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(report + "\n")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"mode={mode}\n")
            f.write(f"tests={' '.join(tests)}\n")
            f.write(f"shards={json.dumps(shards)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
