"""Capture portable hardware/software provenance without changing system settings."""

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


PACKAGES = ("numpy", "scipy", "torch", "pyarrow", "mne", "edfio", "pyedflib", "triton", "pytest")


def _command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_environment() -> dict:
    """Return CPU model, OS, Python, installed package versions and GPU driver data.

    Missing utilities and packages are reported rather than preventing CPU-only
    checks. GPU names in nvidia-smi are physical devices, not CUDA index mappings.
    """
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    if cpu_model is None and platform.system() == "Darwin":
        cpu_model = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "packages": versions,
        "nvidia_smi_csv": _command(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
    }


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
