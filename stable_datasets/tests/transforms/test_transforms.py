import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from stable_datasets.transforms import (  # noqa: E402
    Chain,
    ColorJitter,
    Flip,
    Normalize,
    RandomCrop,
    RandomResizedCrop,
    Resize,
)


def images(n=6, seed=0, shapes=((120, 160), (97, 211), (64, 64), (300, 90), (224, 224), (31, 47))):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 256, (3, h, w), generator=g, dtype=torch.uint8) for h, w in shapes[:n]]


def shapes_of(xs):
    return np.array([x.shape[-2:] for x in xs])


def crop_region(x, box):
    nd = len(box) // 2
    return x[(..., *[slice(int(a), int(b)) for a, b in zip(box[:nd], box[nd:])])]


CHAINS = [
    Chain(RandomResizedCrop(96, scale=(0.08, 1.0))),
    Chain(RandomResizedCrop(64, scale=(0.02, 0.3)), Flip(), ColorJitter(p=1.0, gray_p=0.5), Normalize()),
    Chain(Normalize(mean=(10, 20, 30), std=(2, 3, 4)), RandomResizedCrop(48, scale=(0.5, 1.0))),
    Chain(RandomCrop((40, 50)), Flip(p=1.0)),
    Chain(RandomCrop(70)),
]


@pytest.mark.parametrize("chain", CHAINS, ids=lambda c: "-".join(op[0] for op in c.signature()))
def test_the_region_alone_gives_the_same_output(chain):
    xs = images()
    if isinstance(chain.ops[0], Normalize):
        xs = [x.float() for x in xs]
    params = chain.draw(np.random.default_rng(3), shapes_of(xs))
    boxes = chain.region(params)
    assert (boxes[:, :2] >= 0).all() and (boxes[:, 2:] <= shapes_of(xs)).all()
    for i, x in enumerate(xs):
        full = chain.apply(x, params, i)
        part = chain.apply(crop_region(x, boxes[i]), params, i, origin=boxes[i, :2])
        assert full.shape == part.shape
        assert torch.equal(full, part), i


def test_region_is_small_for_small_crops():
    xs = images()
    chain = Chain(RandomResizedCrop(32, scale=(0.01, 0.02)))
    boxes = chain.region(chain.draw(np.random.default_rng(0), shapes_of(xs)))
    area = np.prod(boxes[:, 2:] - boxes[:, :2], 1)
    assert (area < 0.2 * np.prod(shapes_of(xs), 1)).all()


def test_random_resized_crop_boxes():
    shapes = np.tile([[480, 640]], (2000, 1))
    op = RandomResizedCrop(224, scale=(0.2, 0.6), ratio=(0.5, 2.0))
    b = op.draw(np.random.default_rng(1), shapes)["box"]
    h, w = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    assert (b[:, :2] >= 0).all() and (b[:, 2] <= 480 + 1e-9).all() and (b[:, 3] <= 640 + 1e-9).all()
    frac = h * w / (480 * 640)
    assert frac.min() >= 0.2 - 1e-9 and frac.max() <= 0.6 + 1e-9
    assert (w / h).min() >= 0.5 - 1e-9 and (w / h).max() <= 2.0 + 1e-9


def test_random_resized_crop_matches_roi_align_on_the_whole_frame():
    from torchvision.ops import roi_align

    x = images(1)[0]
    op = RandomResizedCrop(56, scale=(0.3, 1.0))
    chain = Chain(op)
    params = chain.draw(np.random.default_rng(0), shapes_of([x]))
    y0, x0, y1, x1 = params[0]["box"][0]
    want = roi_align(x[None].float() / 255, torch.tensor([[0, x0, y0, x1, y1]], dtype=torch.float32), 56, 1.0,
                     op.sampling_ratio(x.shape[-2:]), aligned=True)[0]  # fmt: skip
    assert torch.equal(chain.apply(x, params, 0), want)


def test_clips_share_parameters_across_frames():
    g = torch.Generator().manual_seed(0)
    clip = torch.randint(0, 256, (4, 3, 90, 120), generator=g, dtype=torch.uint8)
    chain = Chain(RandomResizedCrop(32), Flip(p=1.0), ColorJitter(p=1.0, gray_p=0.0))
    params = chain.draw(np.random.default_rng(0), [[90, 120]])
    out = chain.apply(clip, params, 0)
    assert out.shape == (4, 3, 32, 32)
    one = Chain(*chain.ops).apply(clip[2], params, 0)
    assert torch.allclose(out[2], one)


def test_random_crop_nd_and_padding():
    vol = torch.arange(2 * 10 * 12 * 14).reshape(2, 10, 12, 14)
    chain = Chain(RandomCrop((4, 5, 6)))
    params = chain.draw(np.random.default_rng(0), [[10, 12, 14]])
    out = chain.apply(vol, params, 0)
    s = params[0]["start"][0]
    assert torch.equal(out, vol[:, s[0] : s[0] + 4, s[1] : s[1] + 5, s[2] : s[2] + 6])
    small = Chain(RandomCrop((8, 8)))
    p = small.draw(np.random.default_rng(0), [[5, 20]])
    out = small.apply(torch.ones(1, 5, 20), p, 0)
    assert out.shape == (1, 8, 8) and out[:, 5:].sum() == 0 and out[:, :5].min() == 1
    assert small.out_shape(p).tolist() == [[8, 8]]


def test_jitter_and_gray():
    x = images(1)[0].float() / 255
    off = Chain(ColorJitter(p=0.0, gray_p=0.0))
    assert torch.equal(off.apply(x, off.draw(np.random.default_rng(0), [[120, 160]]), 0), x)
    gray = Chain(ColorJitter(p=0.0, gray_p=1.0))
    y = gray.apply(x, gray.draw(np.random.default_rng(0), [[120, 160]]), 0)
    assert torch.equal(y[0], y[1]) and torch.equal(y[1], y[2])


def test_whole_input_chains():
    xs = images(2)
    chain = Chain(Flip(p=1.0), RandomResizedCrop(32))
    params = chain.draw(np.random.default_rng(0), shapes_of(xs))
    assert (chain.region(params)[:, 2:] == shapes_of(xs)).all()
    with pytest.raises(ValueError, match="whole input"):
        chain.apply(xs[0][:, 1:], params, 0, origin=(1, 0))
    resized = Chain(Resize((20, 30)), Normalize())
    p = resized.draw(np.random.default_rng(0), shapes_of(xs))
    assert resized.apply(xs[1], p, 1).shape == (3, 20, 30)


def test_draws_are_reproducible_and_signatures_name_settings():
    chain = CHAINS[1]
    a = chain.draw(np.random.default_rng(7), [[100, 100]] * 3)
    b = chain.draw(np.random.default_rng(7), [[100, 100]] * 3)
    for pa, pb in zip(a, b):
        for k in pa:
            assert np.array_equal(pa[k], pb[k])
    assert chain.signature() == CHAINS[1].signature()
    assert Chain(RandomResizedCrop(96)).signature() != Chain(RandomResizedCrop(64)).signature()


@pytest.mark.gpu
def test_the_fused_view_kernel_takes_the_same_parameters():
    """framestore.views fuses RandomResizedCrop, Flip, ColorJitter and Normalize: same parameters, same output."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("triton")
    import math

    from stable_datasets.framestore.views import Views, view_frames

    B, T, H, W = 3, 4, 128, 160
    g = torch.Generator(device="cuda").manual_seed(0)
    clip = torch.randint(0, 256, (B, T, H, W, 3), device="cuda", generator=g, dtype=torch.uint8)
    rrc = RandomResizedCrop(64, scale=(0.3, 1.0))
    chain = Chain(rrc, Flip(), ColorJitter(p=1.0, gray_p=0.3), Normalize())
    params = chain.draw(np.random.default_rng(0), [[H, W]] * B)
    box = torch.tensor(params[0]["box"][:, [1, 0, 3, 2]], dtype=torch.float32, device="cuda")
    max_side = math.sqrt(rrc.scale[1] * H * W * rrc.ratio[1])
    views = Views(box, torch.tensor(params[1]["flip"], device="cuda"),
                  torch.tensor(params[2]["jitter"], dtype=torch.float32, device="cuda"), rrc.size,
                  rrc.sampling_ratio((H, W)), B, min(max_side, max(H, W)))  # fmt: skip
    fused = view_frames(clip, views)
    ref = torch.stack([chain.apply(clip[b].permute(0, 3, 1, 2), params, b) for b in range(B)])
    assert (fused - ref).abs().max() < 2e-4
