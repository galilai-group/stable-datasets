"""Experimental numeric signal caches independent of source readers.

One cache contains one continuous, equal-rate recording in (time, channels)
layout. Both representations yield volts: float32 directly, or native int16
via per-channel affine calibration. No resampling or channel inference occurs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignalMetadata:
    """Common signal contract; provenance retains source-specific information."""

    channel_names: tuple[str, ...]
    sample_rate_hz: float
    provenance: dict[str, Any]

    def validate(self) -> None:
        """Require a known positive sample clock and an explicit channel order."""
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Channel names must be nonempty and unique")
        if not all(isinstance(name, str) and name for name in self.channel_names):
            raise ValueError("Every channel requires a name")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be finite and positive")


def checksum(path: Path) -> str:
    """Return a streaming SHA256 digest without loading a cache into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_representations(
    physical: NDArray[np.float32],
    digital: NDArray[np.int16],
    gain: NDArray[np.float32],
    offset: NDArray[np.float32],
    metadata: SignalMetadata,
) -> None:
    """Check every sample before persistence, using bounded temporary arrays.

    Float32 samples and calibrated native integers must match in volts within
    abs(error) <= 1e-10 + 2e-6 * abs(reference). Source readers remain responsible
    for calibration metadata semantics and continuity of the recording.
    """
    metadata.validate()
    if physical.ndim != 2 or physical.shape != digital.shape or not physical.shape[0]:
        raise ValueError("Representations must have the same nonempty (time, channels) shape")
    if physical.dtype != np.float32 or digital.dtype != np.int16:
        raise ValueError("Expected float32 physical values and native int16 samples")
    channels = physical.shape[1]
    if channels != len(metadata.channel_names) or gain.shape != (channels,) or offset.shape != (channels,):
        raise ValueError("Channel metadata and calibration must match the sample layout")
    if gain.dtype != np.float32 or offset.dtype != np.float32:
        raise ValueError("Calibration must be float32")
    if not np.isfinite(gain).all() or not np.isfinite(offset).all() or np.any(gain == 0):
        raise ValueError("Calibration must be finite with nonzero gains")
    for start in range(0, len(physical), 65536):
        reference = physical[start : start + 65536]
        calibrated = digital[start : start + 65536].astype(np.float32) * gain + offset
        if not np.isfinite(reference).all() or not np.isfinite(calibrated).all():
            raise ValueError("Nonfinite signal values are unsupported")
        np.testing.assert_allclose(calibrated, reference, rtol=2e-6, atol=1e-10)


def write_signal_cache(
    directory: Path,
    physical: NDArray[np.float32],
    digital: NDArray[np.int16],
    gain: NDArray[np.float32],
    offset: NDArray[np.float32],
    metadata: SignalMetadata,
) -> None:
    """Write validated NPY arrays to a new directory, publishing the manifest last.

    Existing directories are never overwritten. A failed write may leave an
    incomplete directory; readers reject it because it lacks a valid manifest.
    Labels/annotations belong to provenance and are not silently reinterpreted.
    """
    validate_representations(physical, digital, gain, offset, metadata)
    manifest = {
        "schema_version": 1,
        "memory_order": "C",
        "layout": "time_channels",
        "physical_unit": "V",
        "shape": list(physical.shape),
        "metadata": asdict(metadata),
        "gain": gain.tolist(),
        "offset": offset.tolist(),
    }
    # Check provenance is serializable before creating any files.
    json.dumps(manifest, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=False)
    for name, samples in (("float32", physical), ("int16", digital)):
        path = directory / f"{name}.npy"
        with path.open("xb") as stream:
            # A transposed MNE array can be Fortran-contiguous. Normalize both
            # representations to the same physical layout for a fair comparison.
            np.save(stream, np.ascontiguousarray(samples), allow_pickle=False)
        manifest[f"{name}_sha256"] = checksum(path)
    with (directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


@dataclass(frozen=True)
class SignalCache:
    """Read-only memory maps sharing channel order, units and window semantics."""

    physical: NDArray[np.float32]
    digital: NDArray[np.int16]
    gain: NDArray[np.float32]
    offset: NDArray[np.float32]
    metadata: SignalMetadata
    manifest: dict[str, Any]

    @classmethod
    def open(cls, directory: Path) -> SignalCache:
        """Verify checksums and structural metadata once, then memory-map arrays.

        Checksum verification warms the OS page cache. Subsequent benchmarks
        must therefore report warm-cache behavior, not cold disk performance.
        """
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["schema_version"], manifest["layout"], manifest["physical_unit"]) != (1, "time_channels", "V"):
            raise ValueError("Unsupported signal cache schema, layout or units")
        if manifest.get("memory_order") != "C":
            raise ValueError("Signal caches require C-contiguous time-major storage")
        arrays = []
        for name, dtype in (("float32", np.float32), ("int16", np.int16)):
            path = directory / f"{name}.npy"
            if checksum(path) != manifest[f"{name}_sha256"]:
                raise ValueError(f"Cache checksum mismatch: {name}")
            samples = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                samples.dtype != dtype
                or list(samples.shape) != manifest["shape"]
                or samples.ndim != 2
                or not samples.flags.c_contiguous
            ):
                raise ValueError(f"Invalid cache shape or dtype: {name}")
            arrays.append(samples)
        metadata = SignalMetadata(
            tuple(manifest["metadata"]["channel_names"]),
            manifest["metadata"]["sample_rate_hz"],
            manifest["metadata"]["provenance"],
        )
        gain = np.asarray(manifest["gain"], dtype=np.float32)
        offset = np.asarray(manifest["offset"], dtype=np.float32)
        # Structural and numerical validation also detects edited calibration.
        validate_representations(arrays[0], arrays[1], gain, offset, metadata)
        return cls(arrays[0], arrays[1], gain, offset, metadata, manifest)

    def gather(self, representation: str, starts: NDArray[np.int64], output: NDArray) -> None:
        """Copy ordered windows directly into a reusable (batch, length, channels) buffer.

        Output dtype must match storage; repeats and arbitrary request order are
        preserved. Checks happen before any writes. No resampling is performed.
        """
        if representation not in ("float32", "int16"):
            raise ValueError("Unknown signal representation")
        samples = self.physical if representation == "float32" else self.digital
        if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("Window starts must be an integer vector")
        if output.ndim != 3 or output.shape[0] != len(starts) or output.shape[2] != samples.shape[1]:
            raise ValueError("Output must have shape (requests, length, channels)")
        length = output.shape[1]
        if length < 1 or output.dtype != samples.dtype or not output.flags.writeable:
            raise ValueError("Output must be writable with matching dtype and positive window length")
        if np.any(starts < 0) or np.any(starts > len(samples) - length):
            raise IndexError("Window extends outside the recording")
        for destination, start in zip(output, starts):
            np.copyto(destination, samples[int(start) : int(start) + length])
