"""The torch IterableDataset behind PlanLoader (a module of its own: it imports torch, and spawned workers must
import the class by name)."""

import torch

from .engine import build_batch


class PlanBatches(torch.utils.data.IterableDataset):
    """Batches start, start + 1, ... (to stop), dealt round-robin to the workers, so they arrive in order."""

    def __init__(self, planner, source, batch_size, seed, start, stop, key):
        self.planner, self.source, self.batch_size = planner, source, batch_size
        self.seed, self.start, self.stop, self.key = seed, start, stop, key

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        b = self.start + wid
        while self.stop is None or b < self.stop:
            yield build_batch(self.planner, self.source, self.batch_size, self.seed, b, self.key)
            b += nw
