"""stable_datasets.framestore.views against LeVJEPA v2's ViewBuilder ops (torchvision roi_align, flip, jitter, normalize) given the same
view parameters: whole views, and kept patches with each contrast pivot.

    pytest tests/framestore/test_views.py      (needs a GPU)
"""

# ruff: noqa: E402  (the imports follow the importorskip checks)

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")
tv_ops = pytest.importorskip("torchvision.ops")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

from stable_datasets.framestore.views import MEAN, STD, draw_views, view_frames, view_patches


B, T, H, W = 3, 4, 224, 224
MEAN_, STD_ = (torch.tensor(v, device="cuda").view(3, 1, 1) for v in (MEAN, STD))


def clip(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randint(0, 256, (B, T, H // 8, W // 8, 3), device="cuda", generator=g, dtype=torch.uint8)
    # smooth content plus noise: resampling errors show on both
    x = torch.nn.functional.interpolate(base.flatten(0, 1).permute(0, 3, 1, 2).float(), size=(H, W), mode="bilinear")
    x = x + torch.randint(-20, 21, x.shape, device="cuda", generator=g)
    return x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).reshape(B, T, H, W, 3).contiguous()


def crops(c, p):
    """ViewBuilder.crop with p's boxes and flips: (rows, T, 3, s, s) in [0, 1]."""
    frames = c.flatten(0, 1).permute(0, 3, 1, 2).float().div(255)
    rows = p.box.shape[0]
    b = torch.arange(rows, device="cuda") % B
    fidx = (b[:, None] * T + torch.arange(T, device="cuda")[None]).reshape(-1)
    rois = torch.cat([fidx[:, None].float(), p.box.repeat_interleave(T, dim=0)], dim=1)
    out = tv_ops.roi_align(frames, rois, p.size, 1.0, p.sampling_ratio, aligned=True)
    out = out.view(rows, T, 3, p.size, p.size)
    return torch.where(p.flip[:, None, None, None, None], out.flip(-1), out)


def lum(x):
    return 0.299 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :] + 0.114 * x[..., 2:3, :, :]


def jitter(x, j, mu):
    """ViewBuilder.jitter with factors j (rows, 4) and contrast pivot mu (broadcast to x)."""
    v = (slice(None),) + (None,) * (x.dim() - 1)
    x = (x * j[:, 0][v]).clamp(0, 1)
    x = ((x - mu) * j[:, 1][v] + mu).clamp(0, 1)
    lu = lum(x)
    x = ((x - lu) * j[:, 2][v] + lu).clamp(0, 1)
    return torch.where(j[:, 3][v] != 0, lum(x).expand_as(x), x)


def patchify(x):
    """(rows, T, 3, s, s) -> (rows, T * G * G, 768) in PatchEmbed's (c p1 p2) order."""
    rows, T_, C, s, _ = x.shape
    G = s // 16
    return x.view(rows, T_, C, G, 16, G, 16).permute(0, 1, 3, 5, 2, 4, 6).reshape(rows, T_ * G * G, C * 256)


def kept(rows, n, k, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(rows, n, device="cuda", generator=g).topk(k, dim=-1).indices.sort(dim=-1).values


@pytest.mark.parametrize("size,scale", [(224, (0.8, 1.0)), (96, (0.02, 0.4)), (96, (0.3, 1.0))])
def test_whole_views_match_viewbuilder(size, scale):
    c = clip()
    torch.manual_seed(0)
    p = draw_views(B, H, W, views=2, size=size, scale=scale, jitter=(0.4, 0.4, 0.2), jitter_p=1.0, gray_p=0.3)
    ref = crops(c, p)
    bri = (ref * p.jitter[:, 0, None, None, None, None]).clamp(0, 1)
    ref = (jitter(ref, p.jitter, bri.mean(dim=(-3, -2, -1), keepdim=True)) - MEAN_) / STD_
    got = view_frames(c, p)
    assert (got - ref).abs().max() < 2e-4


def test_whole_views_without_jitter():
    c = clip(3)
    p = draw_views(B, H, W, views=1, size=224, scale=(0.3, 1.0))
    assert (view_frames(c, p) - (crops(c, p) - MEAN_) / STD_).abs().max() < 2e-4


def test_kept_patches_without_jitter_are_the_whole_views_patches():
    c = clip(1)
    p = draw_views(B, H, W, views=1, size=224, scale=(0.8, 1.0))
    idx = kept(B, T * 196, T * 10)
    ref = torch.take_along_dim(patchify((crops(c, p) - MEAN_) / STD_), idx[..., None], dim=1)
    assert (view_patches(c, p, idx) - ref).abs().max() < 2e-4


@pytest.mark.parametrize("pivot", ["frame", "view"])
def test_kept_patches_pivot_on_the_kept_patches(pivot):
    c = clip(2)
    p = draw_views(B, H, W, views=4, size=96, scale=(0.02, 0.4), jitter=(0.4, 0.4, 0.2), jitter_p=1.0, gray_p=0.3)
    rows = p.box.shape[0]
    idx = kept(rows, T * 36, T * 2)
    x = torch.take_along_dim(patchify(crops(c, p)), idx[..., None], dim=1)  # (rows, S, 768) in [0, 1]
    bri = (x * p.jitter[:, 0, None, None]).clamp(0, 1)
    t = idx // 36
    if pivot == "view":
        mu = bri.mean(dim=(1, 2))[:, None, None]
    else:
        per = torch.zeros(rows, T, device="cuda").scatter_add_(1, t, bri.sum(-1))
        cnt = torch.zeros(rows, T, device="cuda").scatter_add_(1, t, torch.full_like(t, 768, dtype=torch.float))
        mu = torch.take_along_dim(per / cnt.clamp(min=1), t, dim=1)[..., None]
    # jitter on (rows, S, 3, 256) so lum mixes channels as ViewBuilder does
    y = jitter(x.view(rows, -1, 3, 256).transpose(1, 2), p.jitter, mu.view(rows, 1, -1, 1)).transpose(1, 2)
    ref = ((y.reshape(rows, -1, 3, 256) - MEAN_.view(1, 1, 3, 1)) / STD_.view(1, 1, 3, 1)).reshape(rows, -1, 768)
    assert (view_patches(c, p, idx, pivot=pivot) - ref).abs().max() < 2e-4


def test_capturable():
    c = clip(4)
    p = draw_views(B, H, W, views=4, size=96, scale=(0.02, 0.4), jitter=(0.4, 0.4, 0.2))
    out = view_frames(c, p)  # compile outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        view_frames(c, p, out=out)
    g.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
