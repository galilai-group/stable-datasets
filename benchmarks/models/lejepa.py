from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.nn import all_reduce
from transformers.utils import ModelOutput

from stable_pretraining.backbone import MLP


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1

        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        if self._is_ddp:
            all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley test for multivariate normality.

    Projects embeddings onto random 1-D directions with a seed derived from
    a synchronized step counter and averages per-slice EP statistics.
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            step = self.global_step.clone()
            if self._is_ddp:
                torch.distributed.broadcast(step, src=0)

            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = x @ A
        return self.ep(proj).mean()


@dataclass
class LeJEPAOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    inv_loss: torch.Tensor = None
    sigreg_loss: torch.Tensor = None


class LeJEPA(nn.Module):
    """Canonical LeJEPA: multi-view invariance + sliced Epps-Pulley SIGReg.

    Loss: ``inv_loss + lamb * sigreg_loss`` where the invariance centers are
    derived from **global-view projections only** (local views are regressed
    onto those centers). SIGReg acts on the flat projection batch.

    :param backbone: Feature extractor returning ``(B, embed_dim)``. Must
        tolerate varying spatial sizes (for multi-crop) — use
        ``spt.backbone.vit_hf`` with dynamic/interpolated pos-encoding or a
        CNN.
    :param embed_dim: Backbone output dimension.
    :param projector: Optional projection head. Default:
        ``Linear(embed_dim→512) + MLP([2048, 2048, 512])`` with BN+ReLU.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

        if projector is None:
            projector = nn.Sequential(
                nn.Linear(embed_dim, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, 512],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )
        self.projector = projector

        self.sigreg = SlicedEppsPulley(
            num_slices=n_slices, t_max=t_max, n_points=n_points
        )
        self.lamb = lamb

    @staticmethod
    def _compute_loss(
        all_projected: torch.Tensor,
        n_global: int,
        sigreg: SlicedEppsPulley,
        lamb: float,
    ):
        centers = all_projected[:n_global].mean(0)  # [N, K]
        inv_loss = (centers.unsqueeze(0) - all_projected).square().mean()
        sigreg_loss = sigreg(all_projected.reshape(-1, all_projected.size(-1)))
        loss = inv_loss + lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
        if self.training:
            assert global_views is not None and local_views is not None, (
                "global_views and local_views must be provided in training mode"
            )
            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))

            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            bs = global_views[0].shape[0]
            n_views = len(global_views) + len(local_views)
            all_projected = all_projected.view(n_views, bs, -1)

            loss, inv_loss, sigreg_loss = self._compute_loss(
                all_projected, len(global_views), self.sigreg, self.lamb
            )
            embedding = g_features.detach()
            return LeJEPAOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return LeJEPAOutput(
                loss=zero, embedding=embedding, inv_loss=zero, sigreg_loss=zero
            )
