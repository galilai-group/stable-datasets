"""Check that every dataset builder a PR adds comes with tests and a docs page.

Usage:
    python .github/scripts/check_new_datasets.py --base <sha> [--head <sha>]

For each new module under ``stable_datasets/{images,timeseries,video,tabular}/``:

- a test file under ``stable_datasets/tests/`` imports it (or a class from it),
- each builder class in it is mentioned on a page under ``docs/source/datasets/``,
- that page is listed in ``docs/source/datasets/index.rst``, and
- it shows a teaser figure (``.. image::`` or ``.. figure::``) from ``docs/source/datasets/teasers/``.
"""

import argparse
import os
import re
import sys

from dataset_files import (
    DOCS_DATASETS_DIR,
    ROOT,
    builder_classes,
    changed_files,
    dataset_module,
    tests_for_module,
)


def docs_pages() -> dict[str, str]:
    """Page stem -> text for every dataset docs page except the index."""
    pages = {}
    for page in sorted((ROOT / DOCS_DATASETS_DIR).glob("*")):
        if page.suffix in (".rst", ".md") and page.stem != "index":
            pages[page.stem] = page.read_text(encoding="utf-8", errors="replace")
    return pages


def index_entries() -> set[str]:
    """Pages listed in a toctree of ``docs/source/datasets/index.rst``."""
    index = ROOT / DOCS_DATASETS_DIR / "index.rst"
    if not index.is_file():
        return set()
    return {m.group(1) for m in re.finditer(r"^\s+([A-Za-z0-9_\-]+)\s*$", index.read_text(encoding="utf-8"), re.M)}


_FIGURE_RE = re.compile(r"^\s*\.\. (?:image|figure)::\s*(\S+)", re.M)


def shows_teaser(text: str) -> bool:
    """Whether a docs page shows a figure that exists under ``docs/source/datasets/teasers/``."""
    teasers = (ROOT / DOCS_DATASETS_DIR / "teasers").resolve()
    for target in _FIGURE_RE.findall(text):
        # Sphinx resolves a leading "/" from the docs source folder, anything else from the page's folder.
        base = ROOT / "docs/source" if target.startswith("/") else ROOT / DOCS_DATASETS_DIR
        path = (base / target.lstrip("/")).resolve()
        if teasers in path.parents and path.is_file():
            return True
    return False


def check(new_modules: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """Return ``(path, problem)`` pairs for the new modules ``(path, folder, module)``."""
    pages, listed = docs_pages(), index_entries()
    problems = []
    for path, folder, module in new_modules:
        classes = builder_classes(folder, module)
        if not classes:
            problems.append(
                (
                    path,
                    "no dataset builder class found (a class that subclasses `BaseDatasetBuilder`); "
                    "if this is a helper module, start its name with an underscore",
                )
            )
            continue
        if not tests_for_module(folder, module):
            problems.append(
                (
                    path,
                    f"no test file imports `{module}` or {', '.join(f'`{c}`' for c in classes)}; "
                    f"add one under `stable_datasets/tests/{folder}/`",
                )
            )
        for cls in classes:
            on_pages = [stem for stem, text in pages.items() if re.search(rf"\b{re.escape(cls)}\b", text)]
            if not on_pages:
                problems.append(
                    (path, f"`{cls}` isn't mentioned on any page in `{DOCS_DATASETS_DIR}/`; add a docs page for it")
                )
                continue
            pages_list = ", ".join(f"`{s}`" for s in on_pages)
            listed_pages = [stem for stem in on_pages if stem in listed]
            if not listed_pages:
                problems.append(
                    (path, f"the docs page for `{cls}` ({pages_list}) isn't listed in `{DOCS_DATASETS_DIR}/index.rst`")
                )
            elif not any(shows_teaser(pages[stem]) for stem in listed_pages):
                problems.append(
                    (
                        path,
                        f"the docs page for `{cls}` ({pages_list}) doesn't show a teaser figure "
                        f"from `{DOCS_DATASETS_DIR}/teasers/`",
                    )
                )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="commit to diff against")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    new_modules = []
    for change in changed_files(args.base, args.head):
        found = dataset_module(change.path)
        if change.status == "A" and found and not found[1].startswith("_"):
            new_modules.append((change.path, *found))

    if not new_modules:
        print("No new dataset modules in this change.")
        return 0

    problems = check(new_modules)
    lines = ["**New dataset checklist**", ""]
    for path, _, _ in new_modules:
        mine = [p for f, p in problems if f == path]
        lines.append(f"- {'❌' if mine else '✅'} `{path}`")
        lines += [f"  - {p}" for p in mine]
    report = "\n".join(lines)
    print(report)
    for path, problem in problems:
        print(f"::error file={path}::{problem.replace('`', '')}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(report + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
