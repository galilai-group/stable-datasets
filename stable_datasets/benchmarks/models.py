"""ViT model factory for MAE experiments.

This module provides Vision Transformer implementations that support
arbitrary patch sizes and image sizes for use with stable-pretraining's
MaskedEncoder. This is necessary because timm doesn't have pretrained
models for small image sizes like CIFAR (32x32) or FashionMNIST (28x28).
"""

from __future__ import annotations

from typing import Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp, trunc_normal_


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings for rectangular grids.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4)
        grid_size: Grid height/width as int (square) or (height, width) tuple
        cls_token: If True, prepend a zero embedding for CLS token

    Returns:
        Positional embeddings of shape (H*W, embed_dim) or
        (H*W + 1, embed_dim) if cls_token=True
    """
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)

    dim = embed_dim // 4
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000**omega)

    out_h = grid[:, 0:1] @ omega.unsqueeze(0)
    out_w = grid[:, 1:2] @ omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


class PatchEmbed(nn.Module):
    """Image to patch embedding with support for arbitrary sizes."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use scaled dot product attention (Flash Attention when available)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for arbitrary patch and image sizes.

    Compatible with stable-pretraining's MaskedEncoder interface.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        self.has_class_token = True
        self.num_reg_tokens = 0
        self.no_embed_class = False

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embeddings (2D sincos for rectangular support)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        grid_size = self.patch_embed.grid_size
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classification head (will be replaced with Identity by MaskedEncoder)
        self.head = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # CLS token initialization
        trunc_normal_(self.cls_token, std=0.02)
        # pos_embed is already initialized with 2D sincos (fixed, not learned)

        # Initialize linear layers
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])


# =============================================================================
# Model Size Configurations
# =============================================================================

VIT_CONFIGS = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}


def create_vit(
    size: str = "base",
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    **kwargs,
) -> VisionTransformer:
    """Create a Vision Transformer with arbitrary patch and image sizes.

    Args:
        size: Model size - "tiny", "small", "base", or "large"
        img_size: Input image size
        patch_size: Size of image patches
        in_chans: Number of input channels
        **kwargs: Additional kwargs passed to VisionTransformer

    Returns:
        A VisionTransformer configured for the given parameters
    """
    if size not in VIT_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(VIT_CONFIGS.keys())}")

    config = VIT_CONFIGS[size].copy()
    config.update(kwargs)

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        **config,
    )


def create_vit_base(
    patch_size: int = 16,
    img_size: int = 224,
    pretrained: bool = False,
    in_chans: int = 3,
) -> nn.Module:
    """Create a ViT-Base model.

    Uses timm for standard configurations when pretrained weights are available,
    otherwise falls back to custom implementation for arbitrary sizes.

    Args:
        patch_size: Size of image patches
        img_size: Input image size
        pretrained: Whether to load pretrained weights (only for standard sizes)
        in_chans: Number of input channels

    Returns:
        A ViT-Base model configured for the given patch size and image size
    """
    # Standard timm configurations that have pretrained weights
    timm_configs = {
        (16, 224): "vit_base_patch16_224",
        (16, 384): "vit_base_patch16_384",
        (32, 224): "vit_base_patch32_224",
        (32, 384): "vit_base_patch32_384",
        (14, 224): "vit_base_patch14_224",
    }

    key = (patch_size, img_size)
    if key in timm_configs and (pretrained or in_chans == 3):
        # Use timm for standard configurations
        return timm.create_model(
            timm_configs[key],
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )

    # Custom implementation for non-standard configurations
    return create_vit(
        size="base",
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )
