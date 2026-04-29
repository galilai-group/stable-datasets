"""Formatters that convert Arrow-native values to user-facing types.

Middle layer of the three-layer split
(:class:`StorageBackend` -> :class:`Formatter` -> :class:`StableDataset`).
Formatters consume Arrow values and emit PIL images, numpy arrays, torch
tensors, or raw Python, never touching files or storage themselves.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Features, FeatureType


def _zip_cols_to_rows(cols: dict, n: int) -> list[dict]:
    """Build list of row dicts from column-oriented dict."""
    keys = list(cols.keys())
    return [{k: cols[k][i] for k in keys} for i in range(n)]


def _extract_columns(table) -> dict:
    """Extract Python column lists one column at a time."""
    return {name: table.column(name).to_pylist() for name in table.column_names}


class Formatter:
    """Base formatter. Subclasses convert Arrow-native values to user-facing types."""

    format_type = "default"

    def __init__(self, features: Features, decode_images: bool = True, cache_dir: Path | None = None):
        self.features = features
        self.decode_images = decode_images
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def format_row(self, row: dict) -> dict:
        """Format a single row dict (from backend.get_row)."""
        return {name: self._format_value(name, value) for name, value in row.items()}

    def format_batch(self, table) -> list[dict]:
        """Format a batch (from backend.take). Returns list of row dicts.

        Column-first: extract columns once, decode each column in bulk,
        then zip into per-row dicts at the end.
        """
        cols = self._format_columns(_extract_columns(table))
        return _zip_cols_to_rows(cols, table.num_rows)

    def _format_value(self, name: str, value):
        feat = self.features.get(name)
        if isinstance(feat, FeatureType):
            return feat.format(
                value,
                format_type=self.format_type,
                decode_images=self.decode_images,
                cache_dir=self.cache_dir,
            )
        return value

    def _format_columns(self, cols: dict) -> dict:
        return {name: [self._format_value(name, value) for value in values] for name, values in cols.items()}


class PythonFormatter(Formatter):
    """Default format: Image -> PIL, Array3D -> numpy, scalars -> Python native."""

    format_type = "default"


class RawFormatter(Formatter):
    """Raw format: all values as-is from Arrow (bytes for images, bytes for Array3D)."""

    format_type = "raw"

    def __init__(self, features: Features, decode_images: bool = False, cache_dir: Path | None = None):
        super().__init__(features, decode_images=False, cache_dir=cache_dir)

    def format_row(self, row: dict) -> dict:
        return row

    def format_batch(self, table) -> list[dict]:
        return _zip_cols_to_rows(_extract_columns(table), table.num_rows)


class TorchFormatter(Formatter):
    """Torch format: Image -> CHW float32 tensor, Array3D -> float32 tensor, scalars -> tensors."""

    format_type = "torch"


class NumpyFormatter(Formatter):
    """Numpy format: Image -> HWC numpy array, rest as-is."""

    format_type = "numpy"


def get_formatter(
    format_type: str | None,
    features: Features,
    decode_images: bool = True,
    cache_dir: Path | None = None,
) -> Formatter:
    """Factory for formatter instances."""
    if format_type is None:
        return PythonFormatter(features, decode_images=decode_images, cache_dir=cache_dir)
    if format_type == "raw":
        return RawFormatter(features, decode_images=False, cache_dir=cache_dir)
    if format_type == "torch":
        return TorchFormatter(features, decode_images=decode_images, cache_dir=cache_dir)
    if format_type == "numpy":
        return NumpyFormatter(features, decode_images=decode_images, cache_dir=cache_dir)
    raise ValueError(f"Unknown format type: {format_type!r}")
