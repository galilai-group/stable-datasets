"""Tests for the package metadata and for imports without optional dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject():
    tomllib = pytest.importorskip("tomllib")
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _names(requirements):
    return {req.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for req in requirements}


def test_runtime_dependencies(pyproject):
    deps = pyproject["project"]["dependencies"]
    names = _names(deps)
    assert "scipy" in names
    assert "torch" not in names
    assert "pre-commit" not in names
    assert any(dep.startswith("pyarrow>=") for dep in deps)
    assert any(dep.startswith("numpy>=") for dep in deps)


def test_optional_dependencies(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert "pylance" in _names(extras["lance"])
    assert "torch" in _names(extras["torch"])
    assert {"opencv-python-headless", "torchcodec"} <= _names(extras["video"])
    assert extras["all"] == ["stable-datasets[lance,torch,video]"]
    assert "stable-datasets[all]" in extras["dev"]


def test_tool_config(pyproject):
    assert "pythonpath" not in pyproject["tool"]["pytest"]["ini_options"]
    assert pyproject["tool"]["ruff"]["lint"]["isort"]["known-first-party"] == ["stable_datasets"]


def test_import_and_iterate_without_torch():
    code = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"blocked {name}")

sys.meta_path.insert(0, _Block())

import pyarrow as pa
import stable_datasets.utils
from stable_datasets import DatasetInfo, Features, StableDataset, Value

assert "pd" not in vars(stable_datasets.utils)
features = Features({"x": Value("int32")})
ds = StableDataset(features=features, info=DatasetInfo(features=features), table=pa.table({"x": [1, 2, 3]}))
assert [row["x"] for row in ds.as_iterable()] == [1, 2, 3]
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
