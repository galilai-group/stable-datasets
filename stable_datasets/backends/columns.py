"""What a backend returns for a batch of rows: a dict of column name to the rows' values, with its row count."""

from __future__ import annotations


class Columns(dict):
    """Column name -> list of the rows' Python values (bytes for binary, lists for list types), in row order.

    A plain ``dict`` with ``num_rows``, so it also gives the row count of a batch with no columns."""

    def __init__(self, data=(), num_rows: int | None = None):
        super().__init__(data)
        if num_rows is None:
            num_rows = len(next(iter(self.values()))) if self else 0
        self.num_rows = int(num_rows)

    def rows(self) -> list[dict]:
        """The batch as one dict per row."""
        keys = list(self)
        return [{k: self[k][i] for k in keys} for i in range(self.num_rows)]
