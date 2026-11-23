"""Read-side storage backend protocol.

Defines :class:`StorageBackend`, the interface :class:`StableDataset`
depends on for row access, iteration, and materialization. Concrete
backends (e.g. :class:`LanceBackend`, :class:`TableBackend`) conform structurally.

The boundary types are :class:`Columns` (column name -> the rows' Python
values), numpy arrays and Python dicts. Arrow, which Lance needs, stays
inside the backends (see ``_arrow.ArrowReads``). Members not declared on
the protocol are backend-private.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from .columns import Columns


@runtime_checkable
class StorageBackend(Protocol):
    """Read-side storage interface consumed by ``StableDataset``."""

    # -- Shape ----------------------------------------------------------------

    @property
    def num_rows(self) -> int: ...

    @property
    def num_shards(self) -> int: ...

    @property
    def shard_row_counts(self) -> list[int]: ...

    @property
    def is_file_backed(self) -> bool: ...

    @property
    def column_names(self) -> list[str]: ...

    # -- Materialization ------------------------------------------------------

    def column(self, name: str, indices: np.ndarray | None = None) -> np.ndarray | list:
        """One column's stored values at ``indices`` (all rows if None)."""
        ...

    def materialize(self, indices: np.ndarray | None = None):
        """The rows at ``indices`` (all rows if None) in memory, as a backend
        with column operations (``with_column``, ``drop_columns``,
        ``rename_columns``). Expensive for large datasets."""
        ...

    # -- Random access --------------------------------------------------------

    def get_row(self, idx: int) -> dict: ...

    def take(self, indices: np.ndarray | list[int]) -> Columns: ...

    def slice(self, start: int, length: int) -> Columns: ...

    # -- Sequential iteration -------------------------------------------------

    def iter_batches(
        self,
        shard_indices: list[int] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[Columns]:
        """Yield batches of rows.

        With ``shuffle=True``, shuffle the shard order. A ``seed`` of
        ``None`` gives a different order on each call.
        """
        ...

    # -- Pickle contract ------------------------------------------------------
    # Backends must survive DataLoader worker forks. State should reference
    # files by path (not hold open handles/mmaps) so workers can re-open
    # after fork.

    def __getstate__(self) -> dict: ...

    def __setstate__(self, state: dict) -> None: ...
