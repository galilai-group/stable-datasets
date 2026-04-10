"""LeJEPA: multi-view invariance + SIGReg regularization (4 views).

Includes the EppsPulley + SlicingUnivariateTest loss components.
"""

from __future__ import annotations

import stable_pretraining as spt
import torch
from stable_pretraining.data import transforms
from torch import distributed as dist
from torch import nn

from stable_datasets.benchmarks.models import (
    build_optim_config,
    collate_multiview,
    create_backbone,
    create_projector,
    get_embedding_dim,
    ssl_augmentation,
    val_transform,
)


NUM_VIEWS = 4


def create_transforms(ds_config):
    """Returns (train_transform, val_transform, collate_fn)."""
    h, w = ds_config.image_size
    view = ssl_augmentation(ds_config, (h, w), crop_scale=(0.08, 1.0))
    train = transforms.MultiViewTransform([view] * NUM_VIEWS)
    return train, val_transform(ds_config), collate_multiview


# LeJEPA Loss (EppsPulley + SlicingUnivariateTest)


def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def _all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        from torch.distributed._functional_collectives import all_reduce as functional_all_reduce

        return functional_all_reduce(x, op.lower(), dist.group.WORLD)
    return x


class EppsPulley(nn.Module):
    """Fast Epps-Pulley two-sample test statistic via characteristic function integration.

    Args:
        t_max: Maximum integration point.
        n_points: Number of integration points (must be odd).
    """

    def __init__(self, t_max: float = 3, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    @property
    def world_size(self):
        if _is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1

    def forward(self, x):
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        cos_mean = cos_vals.mean(-3)
        sin_mean = sin_vals.mean(-3)

        cos_mean = _all_reduce(cos_mean)
        sin_mean = _all_reduce(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


class SlicingUnivariateTest(nn.Module):
    """Multivariate normality test via random 1D projections.

    Args:
        univariate_test: A univariate test module (e.g., EppsPulley).
        num_slices: Number of random 1D projections.
        reduction: Aggregation method: 'mean', 'sum', or None.
    """

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.univariate_test = univariate_test
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = _all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            g = self._get_generator(x.device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, device=x.device, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        return stats


# Forward


def forward(self, batch, stage):
    """LeJEPA forward: multi-view invariance + SIGReg regularization."""
    out = {}
    views = batch.get("views")

    if views is not None:
        V, N = len(views), views[0]["image"].size(0)
        all_images = torch.cat([v["image"] for v in views], dim=0)
        all_emb = self.backbone(all_images)
        out["embedding"] = all_emb[:N]

        if "label" in views[0]:
            out["label"] = views[0]["label"]

        if self.training:
            all_proj = self.projector(all_emb)
            proj_stacked = all_proj.reshape(V, N, -1)
            view_mean = proj_stacked.mean(0)
            inv_loss = (view_mean - proj_stacked).square().mean()

            if isinstance(self.sigreg_loss, SlicingUnivariateTest) and isinstance(
                self.sigreg_loss.univariate_test, EppsPulley
            ):
                sigreg_loss = self.sigreg_loss(proj_stacked)
            else:
                sigreg_loss = self.sigreg_loss(proj_stacked.reshape(-1, proj_stacked.size(-1)))

            lamb = getattr(self, "lamb", 0.02)
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            out["loss"] = lejepa_loss

            self.log(f"{stage}/sigreg", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/inv", inv_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/loss", lejepa_loss, on_step=True, on_epoch=True, sync_dist=True)
    else:
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


# Builder


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim)
    univariate_test = EppsPulley(t_max=cfg.model.loss.t_max, n_points=cfg.model.loss.n_points)
    sigreg_loss = SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=cfg.model.loss.num_slices,
        reduction="mean",
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        sigreg_loss=sigreg_loss,
        lamb=cfg.model.loss.lamb,
        forward=forward,
        optim=build_optim_config(cfg.model, cfg.backbone),
    )
    return module, embed_dim
