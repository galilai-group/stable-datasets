"""The plan loader: batches of decoded, transformed samples with their labels, from worker processes.

    loader = PlanLoader(planner, LocalSource(records.files), batch_size=64, workers=8, seed=0)
    for batch in loader:            # {"image": (64, C, ...) tensor, "label": (64,), "record": (64,), "batch": int}
        ...

Batch b is planned from a generator seeded by (seed, b), so its content does not depend on the number of workers
or on which worker builds it, and batches arrive in order. Workers are spawned (Lance is not fork-safe).
"""

from __future__ import annotations

import numpy as np

from .plan import Planner, fetch_and_decode


def build_batch(planner: Planner, source, batch_size: int, seed: int, b: int, key: str = "image") -> dict:
    """Batch ``b``: its plan executed by the reference path."""
    import torch

    plan = planner.plan(np.random.default_rng([seed, b]), batch_size)
    xs = fetch_and_decode(plan, planner, source)
    out = {key: torch.stack(xs), "record": torch.from_numpy(plan.record.copy()), "batch": b}
    if plan.label is not None:
        out["label"] = torch.as_tensor(np.asarray(plan.label))
    return out


class PlanLoader:
    """Batches from a Planner and a ByteSource. ``batches``: how many (None: without end); ``start``: the first
    batch's number, to resume. ``workers`` processes build batches ahead (``prefetch`` each); 0 builds them in this
    process. ``pin_memory`` for copies to a GPU."""

    def __init__(self, planner: Planner, source, batch_size: int, *, seed: int = 0, batches: int | None = None,
                 start: int = 0, workers: int = 0, prefetch: int = 2, pin_memory: bool = False, key: str = "image"):  # fmt: skip
        self.planner, self.source = planner, source
        self.batch_size, self.seed, self.start, self.key = int(batch_size), int(seed), int(start), key
        self.batches = None if batches is None else int(batches)
        self.workers, self.prefetch, self.pin_memory = int(workers), int(prefetch), bool(pin_memory)

    def __len__(self) -> int:
        if self.batches is None:
            raise TypeError("an endless PlanLoader has no length")
        return self.batches

    def __iter__(self):
        import torch

        stop = None if self.batches is None else self.start + self.batches
        from ._batches import PlanBatches

        data = PlanBatches(self.planner, self.source, self.batch_size, self.seed, self.start, stop, self.key)
        kw = {}
        if self.workers > 0:
            kw = {"multiprocessing_context": "spawn", "prefetch_factor": self.prefetch, "persistent_workers": False}
        loader = torch.utils.data.DataLoader(
            data, batch_size=None, num_workers=self.workers, pin_memory=self.pin_memory, **kw
        )
        yield from loader
