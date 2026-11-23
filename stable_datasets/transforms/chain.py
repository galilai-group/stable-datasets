"""A chain of ops: draw every op's parameters for a batch, find the input region each sample needs, apply."""

from __future__ import annotations

import numpy as np

from .ops import Op


class Chain:
    """Ops applied in order.

    ``draw`` gives one parameter dict per op for a batch of samples of the given spatial shapes; ``region`` gives
    the box of each sample's input the chain reads; ``apply`` runs the reference ops on one sample, which may be
    only that region of its input (``origin``: where the region starts).
    """

    def __init__(self, *ops: Op):
        self.ops = tuple(ops)

    def __repr__(self) -> str:
        return f"Chain({', '.join(map(repr, self.ops))})"

    def __len__(self) -> int:
        return len(self.ops)

    def signature(self) -> tuple:
        """What a fast path matches on: each op's type and settings, in order."""
        return tuple(op.signature() for op in self.ops)

    def draw(self, rng: np.random.Generator, shapes) -> list[dict]:
        shapes = np.atleast_2d(np.asarray(shapes, np.int64))
        params = []
        for op in self.ops:
            p = op.draw(rng, shapes)
            p["_in"] = shapes
            params.append(p)
            shapes = np.asarray(op.out_shape(p, shapes), np.int64)
        return params

    def out_shape(self, params: list[dict]) -> np.ndarray:
        if not self.ops:
            raise ValueError("an empty chain has no output shape of its own")
        return np.asarray(self.ops[-1].out_shape(params[-1], params[-1]["_in"]), np.int64)

    def _reader(self) -> int | None:
        """Index of the op whose region the chain reads: the first cropping op with only pointwise ops before it."""
        for k, op in enumerate(self.ops):
            if op.crops:
                return k
            if not op.pointwise:
                return None
        return None

    def region(self, params: list[dict]) -> np.ndarray:
        """(n, 2 * ndim) box of each sample's input that the chain reads: the first crop's region, else the whole
        input."""
        k = self._reader()
        shapes = params[0]["_in"] if params else None
        if k is None:
            return np.concatenate([np.zeros_like(shapes), shapes], 1)
        return self.ops[k].region(params[k], shapes)

    def apply(self, x, params: list[dict], i: int, origin=None):
        """Sample ``i`` through every op. ``origin``: where ``x`` starts in its input, when ``x`` is only the
        chain's region of it."""
        k = self._reader()
        if origin is not None and k is None and np.any(np.asarray(origin) != 0):
            raise ValueError("this chain reads the whole input, so it cannot start from a region")
        for j, (op, p) in enumerate(zip(self.ops, params)):
            x = op.apply(x, p, i, origin=origin if j == k else None)
        return x

    def apply_batch(self, xs, params: list[dict], origins=None):
        """Every sample through the chain, stacked."""
        import torch

        return torch.stack(
            [self.apply(x, params, i, None if origins is None else origins[i]) for i, x in enumerate(xs)]
        )
