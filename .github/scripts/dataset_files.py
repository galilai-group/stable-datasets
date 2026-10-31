"""Shared helpers for the CI scripts: what a change touches and which tests cover it.

Everything here reads source files with ``ast``; nothing from the PR is imported or run.
"""

import ast
import re
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRS = ("images", "timeseries", "video", "tabular")
TESTS_DIR = "stable_datasets/tests"
DOCS_DATASETS_DIR = "docs/source/datasets"

_MODULE_RE = re.compile(r"stable_datasets/(" + "|".join(DATASET_DIRS) + r")/([A-Za-z0-9_]+)\.py")


@dataclass(frozen=True)
class Change:
    status: str  # "A" added, "M" modified, "D" deleted
    path: str


def changed_files(base: str, head: str = "HEAD") -> list[Change]:
    """Files that differ between ``base`` and ``head``. Renames show up as a delete plus an add."""
    out = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base, head],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changes = []
    for line in out.splitlines():
        status, path = line.split("\t", 1)
        changes.append(Change(status[0], path))
    return changes


def dataset_module(path: str) -> tuple[str, str] | None:
    """``("images", "gtsrb")`` for ``stable_datasets/images/gtsrb.py``; None for any other path."""
    m = _MODULE_RE.fullmatch(path)
    return (m.group(1), m.group(2)) if m else None


@cache
def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


@cache
def _builder_names(folder: str) -> frozenset[str]:
    """Names of classes in a dataset folder that subclass ``BaseDatasetBuilder``, directly or not."""
    bases: dict[str, set[str]] = {}
    for py in (ROOT / "stable_datasets" / folder).glob("*.py"):
        tree = _parse(py)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(_base_name(b) for b in node.bases)
    builders = {"BaseDatasetBuilder"}
    grew = True
    while grew:
        new = {name for name, b in bases.items() if b & builders} - builders
        builders |= new
        grew = bool(new)
    return frozenset(builders - {"BaseDatasetBuilder"})


def public_names(folder: str, module: str) -> set[str]:
    """Top-level public classes and functions defined in a dataset module."""
    tree = _parse(ROOT / "stable_datasets" / folder / f"{module}.py")
    if tree is None:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    }


def builder_classes(folder: str, module: str) -> list[str]:
    """Public dataset builder classes defined in ``stable_datasets/<folder>/<module>.py``."""
    tree = _parse(ROOT / "stable_datasets" / folder / f"{module}.py")
    if tree is None:
        return []
    builders = _builder_names(folder)
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in builders and not node.name.startswith("_")
    ]


def modules_importing(folder: str, helper: str) -> list[str]:
    """Modules in a dataset folder that import the private helper module ``helper`` (e.g. ``_cifar10``)."""
    users = []
    for py in sorted((ROOT / "stable_datasets" / folder).glob("*.py")):
        tree = _parse(py)
        if tree is None or py.stem == helper:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                (node.level == 1 and node.module == helper)
                or (node.level == 1 and node.module is None and any(a.name == helper for a in node.names))
                or node.module == f"stable_datasets.{folder}.{helper}"
            ):
                users.append(py.stem)
                break
    return users


@dataclass(frozen=True)
class _TestImports:
    modules: frozenset[str]  # dotted module paths imported
    names: frozenset[tuple[str, str]]  # (package, name) from `from package import name`
    text: str


@cache
def _test_files() -> dict[str, _TestImports]:
    files = {}
    for py in sorted((ROOT / TESTS_DIR).rglob("test_*.py")):
        tree = _parse(py)
        modules, names = set(), set()
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules.add(node.module)
                    names.update((node.module, a.name) for a in node.names)
        text = py.read_text(encoding="utf-8", errors="replace")
        files[py.relative_to(ROOT).as_posix()] = _TestImports(frozenset(modules), frozenset(names), text)
    return files


def tests_for_module(folder: str, module: str) -> list[str]:
    """Test files that import a dataset module or any public name defined in it."""
    package = f"stable_datasets.{folder}"
    dotted = f"{package}.{module}"
    defined = public_names(folder, module)
    found = []
    for test, imp in _test_files().items():
        if (
            dotted in imp.modules
            or (package, module) in imp.names
            or any((package, name) in imp.names for name in defined)
            or dotted in imp.text  # e.g. monkeypatch targets like "stable_datasets.images.gtsrb.download"
            or any(re.search(rf"\b{folder}\.{re.escape(name)}\b", imp.text) for name in defined)
        ):
            found.append(test)
    return found


def tests_importing_package(package: str) -> list[str]:
    """Test files that import ``package`` or one of its submodules (e.g. ``benchmarks``)."""
    return [
        test
        for test, imp in _test_files().items()
        if any(m == package or m.startswith(package + ".") for m in imp.modules)
    ]
