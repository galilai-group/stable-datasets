"""StableDataset: PyArrow-backed dataset with optional TorchDict conversion.

Drop-in replacement for ``datasets.Dataset`` / ``datasets.DatasetDict`` with
the same ``__len__``, ``__getitem__``, ``.features``, ``.train_test_split()``
API that downstream code (benchmarks) relies on.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
from PIL import Image as PILImage

from .schema import (
    Array3D,
    ClassLabel,
    DatasetInfo,
    Features,
    Image,
    Sequence,
    Value,
    Video,
)


class StableDataset:
    """A single-split dataset backed by a PyArrow Table."""

    def __init__(self, table: pa.Table, features: Features, info: DatasetInfo):
        self._table = table
        self._features = features
        self._info = info

    # -- public properties ---------------------------------------------------

    @property
    def features(self) -> Features:
        return self._features

    @property
    def info(self) -> DatasetInfo:
        return self._info

    # -- sequence protocol ---------------------------------------------------

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            return self._decode_row(idx)
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            table = self._table.take(list(indices))
            return StableDataset(table, self._features, self._info)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    # -- splitting -----------------------------------------------------------

    def train_test_split(self, test_size: float = 0.1, seed: int = 42) -> dict[str, "StableDataset"]:
        """Random split. Returns ``{"train": StableDataset, "test": StableDataset}``."""
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)
        split_idx = int(n * (1 - test_size))
        train_indices = indices[:split_idx].tolist()
        test_indices = indices[split_idx:].tolist()
        return {
            "train": StableDataset(self._table.take(train_indices), self._features, self._info),
            "test": StableDataset(self._table.take(test_indices), self._features, self._info),
        }

    # -- optional TorchDict conversion ---------------------------------------

    def to_tensordict(self, columns: list[str] | None = None):
        """Convert numeric columns to a ``tensordict.TensorDict``.

        Image and Video columns are skipped (they stay lazy-decoded).
        Requires ``tensordict`` to be installed.
        """
        from tensordict import TensorDict
        import torch

        td = {}
        for col_name, feat in self._features.items():
            if isinstance(feat, (Image, Video, Array3D)):
                continue
            if columns and col_name not in columns:
                continue
            col = self._table.column(col_name)
            if isinstance(feat, Sequence):
                td[col_name] = torch.tensor(col.to_pylist())
            else:
                td[col_name] = torch.from_numpy(col.to_numpy(zero_copy_only=False))
        return TensorDict(td, batch_size=[len(self)])

    # -- internal decoding ---------------------------------------------------

    def _decode_row(self, idx: int) -> dict:
        """Decode a single row, converting Arrow values back to Python objects."""
        result = {}
        for col_name in self._table.column_names:
            feat = self._features.get(col_name)
            raw = self._table.column(col_name)[idx]

            if isinstance(feat, Image):
                img_bytes = raw.as_py()
                if img_bytes is not None:
                    result[col_name] = PILImage.open(io.BytesIO(img_bytes))
                else:
                    result[col_name] = None
            elif isinstance(feat, Array3D):
                arr_bytes = raw.as_py()
                if arr_bytes is not None:
                    result[col_name] = np.frombuffer(arr_bytes, dtype=feat.dtype).reshape(feat.shape)
                else:
                    result[col_name] = None
            elif isinstance(feat, Video):
                result[col_name] = raw.as_py()
            elif isinstance(feat, Sequence):
                result[col_name] = raw.as_py()
            elif isinstance(feat, ClassLabel):
                result[col_name] = raw.as_py()
            else:
                result[col_name] = raw.as_py()
        return result


class StableDatasetDict(dict):
    """Dict of ``split_name -> StableDataset``, replacing ``datasets.DatasetDict``."""

    pass
