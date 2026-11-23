import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from stable_datasets.codecs import RestartJpeg  # noqa: E402
from stable_datasets.loading import PlanLoader, Planner, SamplingIndex, build_batch  # noqa: E402
from stable_datasets.sources import LocalSource  # noqa: E402
from stable_datasets.tests.loading.test_plan import jpeg_store  # noqa: E402, F401
from stable_datasets.transforms import Chain, Flip, Normalize, RandomResizedCrop  # noqa: E402


def _planner(records):
    index = SamplingIndex(np.arange(len(records)), label=np.arange(len(records)) * 10)
    return Planner(RestartJpeg(), records, Chain(RandomResizedCrop(32, scale=(0.1, 0.5)), Flip(), Normalize()),
                   index=index, gap=64)  # fmt: skip


def test_batches_carry_labels_and_follow_their_plan(jpeg_store):  # noqa: F811
    records, full = jpeg_store
    planner = _planner(records)
    batches = list(PlanLoader(planner, LocalSource(records.files), 8, seed=3, batches=3))
    assert [b["batch"] for b in batches] == [0, 1, 2]
    for b in batches:
        assert b["image"].shape == (8, 3, 32, 32) and b["image"].dtype == torch.float32
        assert torch.equal(b["label"], b["record"] * 10)
    plan = planner.plan(np.random.default_rng([3, 1]), 8)
    want = torch.stack([planner.chain.apply(full[r], plan.params, i) for i, r in enumerate(plan.record)])
    assert torch.equal(batches[1]["image"], want)


def test_batches_do_not_depend_on_workers(jpeg_store):  # noqa: F811
    records, _ = jpeg_store
    planner = _planner(records)
    source = LocalSource(records.files)
    inline = list(PlanLoader(planner, source, 4, seed=1, batches=5))
    spawned = list(PlanLoader(planner, source, 4, seed=1, batches=5, workers=2))
    assert [b["batch"] for b in spawned] == list(range(5))
    for a, b in zip(inline, spawned):
        assert torch.equal(a["image"], b["image"]) and torch.equal(a["record"], b["record"])


def test_resume_from_a_batch(jpeg_store):  # noqa: F811
    records, _ = jpeg_store
    planner = _planner(records)
    source = LocalSource(records.files)
    later = next(iter(PlanLoader(planner, source, 4, seed=1, start=3, batches=1)))
    assert later["batch"] == 3
    assert torch.equal(later["image"], build_batch(planner, source, 4, 1, 3)["image"])
    with pytest.raises(TypeError):
        len(PlanLoader(planner, source, 4))
