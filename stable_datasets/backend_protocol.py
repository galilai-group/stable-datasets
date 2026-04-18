"""Read-side storage backend protocol.

Defines :class:`StorageBackend`, the interface :class:`StableDataset`
depends on for row access, iteration, and materialization. Concrete
backends (e.g. :class:`ArrowBackend`) conform structurally.

Arrow types (:class:`pa.Table`, :class:`pa.RecordBatch`,
:class:`pa.Schema`) are the boundary types. Members not declared on the
protocol are backend-private.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa


@runtime_checkable
class StorageBackend(Protocol):
    """Read-side storage interface consumed by ``StableDataset``."""

    # -- Shape ----------------------------------------------------------------

    @property
    def num_rows(self) -> int: ...

    @property
    def num_shards(self) -> int: ...

    @property
    def is_file_backed(self) -> bool: ...

    @property
    def schema(self) -> pa.Schema: ...

    # -- Materialization ------------------------------------------------------

    @property
    def table(self) -> pa.Table:
        """Full materialization as a single ``pa.Table``.

        Expensive for multi-shard datasets. Hot paths should prefer
        ``get_row``, ``take``, ``slice``, or ``iter_batches``.
        """
        ...

    # -- Random access --------------------------------------------------------

    def get_row(self, idx: int) -> dict: ...

    def take(self, indices: np.ndarray | list[int]) -> pa.Table: ...

    def slice(self, start: int, length: int) -> pa.Table: ...

    # -- Sequential iteration -------------------------------------------------

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[pa.RecordBatch]: ...

    # -- Pickle contract ------------------------------------------------------
    # Backends must survive DataLoader worker forks. State should reference
    # files by path (not hold open handles/mmaps) so workers can re-open
    # after fork.

    def __getstate__(self) -> dict: ...

    def __setstate__(self, state: dict) -> None: ...
