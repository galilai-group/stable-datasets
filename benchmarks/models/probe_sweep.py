"""Online linear probe that sweeps probe hyperparameters in a single run.

Why: a linear probe's accuracy depends on the probe's own learning rate and
weight decay, and the best setting is not the same for every method — SSL
objectives leave embeddings at wildly different scales. Reporting one probe LR
for all methods measures "how well does this method suit our arbitrary probe LR"
as much as linear separability, which is exactly the confound this benchmark
exists to remove.

So we train K probe heads at once on the same frozen embedding and report the
best. Because the embedding is detached, the heads are independent: their losses
add, and one optimizer step updates all K as if each had been trained alone.

Per-head learning rate and weight decay come from a gradient hook rather than
separate optimizers: the hook scales each head's gradient by ``lr_scale`` and
adds ``weight_decay * param`` to it, which is what a per-head (lr, wd) would do,
at the cost of one extra elementwise op per head.

Shapes: the probe returns ``(N, K, C)``; the loss sums cross-entropy over K; the
metrics slice or reduce over K. This is the whole contract with
``spt.callbacks.OnlineProbe``, which calls ``probe(detach(x))``, then
``loss(preds, y)``, then ``metric.update(preds, y)``.

Note this makes the headline number a best-of-K over a validation-selected
hyperparameter, which is the standard linear-eval protocol (DINOv2 et al. sweep
probe LR and report the best) but IS a mild optimistic bias. It is applied
identically to every method, so between-method comparisons stay fair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


def register_lr_scale_hook(module: nn.Module, lr_scale: float, weight_decay: float = 0.0):
    """Scale grads by ``lr_scale`` and add decoupled weight decay, per-parameter.

    Local copy rather than spt's, so the exact arithmetic behind every probe
    number is in this repo. Hooks fire during backward, before the optimizer, so
    a shared optimizer ends up applying a different effective (lr, wd) per head.
    """
    for p in module.parameters():
        if not p.requires_grad:
            continue

        def hook(grad, p=p):
            if weight_decay:
                grad = grad + weight_decay * p.data
            return grad * lr_scale

        p.register_hook(hook)
    return module


class SweepLinearProbe(nn.Module):
    """K independent probe heads over one frozen embedding.

    ``head_factory(embed_dim, num_classes)`` supplies the head, so the sweep is
    orthogonal to WHICH head a method uses — each model's ``build_probe`` still
    owns that choice (see benchmarks/models/__init__.py).
    """

    def __init__(self, head_factory, embed_dim: int, num_classes: int, grid: list[dict]):
        super().__init__()
        if not grid:
            raise ValueError("probe sweep grid is empty")
        self.grid = list(grid)
        self.heads = nn.ModuleList()
        for cfg in self.grid:
            head = head_factory(embed_dim, num_classes)
            register_lr_scale_hook(head, float(cfg["lr_scale"]), float(cfg["weight_decay"]))
            self.heads.append(head)

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    def tags(self) -> list[str]:
        return [_tag(c) for c in self.grid]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, D) -> (N, K, C)
        return torch.stack([h(x) for h in self.heads], dim=1)


def _fmt(v: float) -> str:
    return f"{v:g}".replace(".", "p").replace("-", "m")


def _tag(cfg: dict) -> str:
    return f"lr{_fmt(cfg['lr_scale'])}_wd{_fmt(cfg['weight_decay'])}"


def sweep_ce_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of cross-entropy over heads.

    Sum, not mean: the heads are independent, so summing gives each its own
    full-strength gradient. Averaging would silently divide every head's
    effective LR by K and make the swept lr_scale values mean something else.
    """
    n, k, c = preds.shape
    return F.cross_entropy(preds.permute(1, 0, 2).reshape(k * n, c), target.repeat(k))


class SweepAccuracy(torchmetrics.Metric):
    """top-k accuracy over ``(N, K, C)`` predictions.

    ``head=None`` reduces across heads with ``reduce`` ("best" or "mean");
    ``head=i`` reports head i alone. State is a per-head correct count, so every
    variant is computed from the same running totals.
    """

    higher_is_better = True
    full_state_update = False

    def __init__(self, num_heads: int, num_classes: int, top_k: int = 1,
                 head: int | None = None, reduce: str = "best"):
        super().__init__()
        if reduce not in ("best", "mean"):
            raise ValueError(f"reduce must be 'best' or 'mean', got {reduce!r}")
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.top_k = min(top_k, num_classes)
        self.head = head
        self.reduce = reduce
        self.add_state("correct", default=torch.zeros(num_heads), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim != 3:
            raise ValueError(f"SweepAccuracy expects (N, K, C), got {tuple(preds.shape)}")
        n = preds.shape[0]
        idx = preds.topk(self.top_k, dim=-1).indices          # (N, K, top_k)
        hit = (idx == target.view(-1, 1, 1)).any(dim=-1)      # (N, K)
        self.correct = self.correct + hit.sum(dim=0).to(self.correct.dtype)
        self.total = self.total + n

    def compute(self) -> torch.Tensor:
        acc = self.correct / self.total.clamp(min=1)
        if self.head is not None:
            return acc[self.head]
        return acc.max() if self.reduce == "best" else acc.mean()


def build_sweep_metrics(num_heads: int, num_classes: int, tags: list[str],
                        per_head: bool = True) -> dict:
    """Metric dict for a swept probe.

    ``top1``/``top5`` stay the headline keys and now mean BEST-over-sweep, so
    every downstream consumer of ``eval/linear_probe_top1_epoch`` keeps working
    without knowing a sweep happened. Per-head keys are added for diagnostics
    (they show whether the best probe sits at the edge of the grid, which is the
    signal that the grid needs widening).
    """
    metrics = {
        "top1": SweepAccuracy(num_heads, num_classes, top_k=1, reduce="best"),
        "top5": SweepAccuracy(num_heads, num_classes, top_k=min(5, num_classes), reduce="best"),
        "top1_mean": SweepAccuracy(num_heads, num_classes, top_k=1, reduce="mean"),
    }
    if per_head:
        for i, tag in enumerate(tags):
            metrics[f"{tag}_top1"] = SweepAccuracy(num_heads, num_classes, top_k=1, head=i)
    return metrics


def expand_grid(lr_scales, weight_decays) -> list[dict]:
    """Cartesian product, in a stable order so head i means the same thing across runs."""
    return [
        {"lr_scale": float(lr), "weight_decay": float(wd)}
        for lr in lr_scales
        for wd in weight_decays
    ]
