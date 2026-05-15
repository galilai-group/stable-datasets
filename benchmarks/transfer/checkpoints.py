"""Off-the-shelf SSL backbones pretrained on ImageNet-1k.

Atomic download + load helpers for the three baselines we want to probe
against the in-domain SSL runs:

  * Barlow Twins ResNet-50  — Facebook AI 1000-epoch IN1K release
  * MAE ViT-Base/16          — He et al., IN1K pretrained encoder
  * DINO ViT-Small/16        — Caron et al., IN1K pretrained encoder

Each loader returns an ``nn.Module`` whose forward yields a pooled feature
vector of size :attr:`BackboneSpec.embed_dim` (no classifier head). The
weights are downloaded once into ``CHECKPOINT_DIR`` and re-used on
subsequent calls.

Use :func:`load_backbone` for a uniform entry point keyed by the same
identifiers used in :data:`BACKBONES`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import timm
import torch
from torch import nn
from torchvision.models.resnet import resnet50 as _resnet50


log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(
    os.environ.get(
        "STABLE_DATASETS_TRANSFER_CKPT_DIR",
        Path(__file__).resolve().parent / "checkpoints",
    )
)

# ImageNet normalization stats — every off-the-shelf backbone here was
# trained with these, so transfer val/probe transforms must match.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# Atomic download


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` if it does not already exist.

    Writes to a sibling ``.part`` file first so that an interrupted
    download cannot leave a half-written checkpoint that later loads
    silently as garbage.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info(f"Downloading {url} -> {dest}")
    urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _load_state_dict(path: Path, key: str | None = None) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and key is not None and key in obj:
        return obj[key]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


# Backbone loaders


def barlow_twins_resnet50() -> nn.Module:
    """Barlow Twins ResNet-50 trunk, IN1K, 1000 epochs (FAIR release).

    The published checkpoint is the encoder only — no projector, no fc —
    so ``strict=False`` discards the missing ``fc.*`` keys and we replace
    the head with ``Identity`` to expose pooled 2048-d features.
    """
    url = "https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth"
    path = _download(url, CHECKPOINT_DIR / "barlow_twins_resnet50.pth")
    state = _load_state_dict(path)
    state = _strip_prefix(state, "module.")

    model = _resnet50(weights=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("fc.", "projector."))]
    if unexpected:
        log.warning(f"barlow_twins_resnet50: unexpected keys {unexpected[:5]}...")
    if any(not k.startswith("fc.") for k in missing):
        raise RuntimeError(f"barlow_twins_resnet50: missing trunk keys {missing}")
    model.fc = nn.Identity()
    return model.eval()


def mae_vit_base() -> nn.Module:
    """MAE ViT-Base/16 encoder, IN1K pretrained (no fine-tune).

    The release ckpt contains both encoder and decoder; ``strict=False``
    drops the decoder. ``num_classes=0`` makes timm return the pooled CLS
    token (768-d) directly.
    """
    url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
    path = _download(url, CHECKPOINT_DIR / "mae_vit_base.pth")
    state = _load_state_dict(path, key="model")

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # MAE decoder + mask token are expected leftovers; the encoder weights
    # must be fully present though.
    decoder_keys = tuple(k for k in unexpected if k.startswith(("decoder", "mask_token")))
    other_unexpected = [k for k in unexpected if k not in decoder_keys]
    if other_unexpected:
        log.warning(f"mae_vit_base: unexpected non-decoder keys {other_unexpected[:5]}...")
    # timm's ViT has a head we deleted via num_classes=0 — head.* missing is fine.
    missing = [k for k in missing if not k.startswith("head")]
    if missing:
        raise RuntimeError(f"mae_vit_base: missing encoder keys {missing[:5]}")
    return model.eval()


def dino_vit_small() -> nn.Module:
    """DINO ViT-Small/16 encoder, IN1K pretrained.

    The release ckpt is the trunk only (no projector). Loads cleanly into
    a timm ``vit_small_patch16_224`` with ``num_classes=0`` for 384-d CLS
    features.
    """
    url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    path = _download(url, CHECKPOINT_DIR / "dino_vit_small.pth")
    state = _load_state_dict(path)

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        log.warning(f"dino_vit_small: unexpected keys {unexpected[:5]}...")
    missing = [k for k in missing if not k.startswith("head")]
    if missing:
        raise RuntimeError(f"dino_vit_small: missing encoder keys {missing[:5]}")
    return model.eval()


# Registry


@dataclass(frozen=True)
class BackboneSpec:
    """Static metadata for an off-the-shelf backbone."""

    name: str
    display_name: str
    loader: Callable[[], nn.Module]
    embed_dim: int
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


BACKBONES: dict[str, BackboneSpec] = {
    "barlow_twins_resnet50_in1k": BackboneSpec(
        name="barlow_twins_resnet50_in1k",
        display_name="Barlow Twins (RN50, IN1k)",
        loader=barlow_twins_resnet50,
        embed_dim=2048,
    ),
    "mae_vit_base_in1k": BackboneSpec(
        name="mae_vit_base_in1k",
        display_name="MAE (ViT-B/16, IN1k)",
        loader=mae_vit_base,
        embed_dim=768,
    ),
    "dino_vit_small_in1k": BackboneSpec(
        name="dino_vit_small_in1k",
        display_name="DINO (ViT-S/16, IN1k)",
        loader=dino_vit_small,
        embed_dim=384,
    ),
}


def load_backbone(name: str) -> tuple[nn.Module, BackboneSpec]:
    """Load a registered off-the-shelf backbone.

    Returns ``(module, spec)`` where ``module`` is in eval mode. The caller
    is responsible for freezing parameters before passing to a probe.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown transfer backbone {name!r}. Available: {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    model = spec.loader()
    return model, spec


if __name__ == "__main__":
    # Quick smoke test: downloads (if missing) and loads each backbone,
    # then runs a single forward to confirm the embed_dim claim.
    logging.basicConfig(level=logging.INFO)
    x = torch.randn(2, 3, 224, 224)
    for name in BACKBONES:
        model, spec = load_backbone(name)
        with torch.no_grad():
            feat = model(x)
        assert feat.shape == (2, spec.embed_dim), (name, feat.shape, spec.embed_dim)
        print(f"{name}: ok, features {tuple(feat.shape)}")
