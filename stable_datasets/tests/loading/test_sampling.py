import numpy as np
import pytest

from stable_datasets.loading import ClassBalanced, SamplingIndex, Uniform, Weighted, clip_windows


def test_weighted_follows_weights():
    index = SamplingIndex(np.arange(4), weight=np.array([0.0, 1.0, 3.0, 0.0]))
    rows = Weighted().draw(np.random.default_rng(0), index, 40_000)
    counts = np.bincount(rows, minlength=4) / len(rows)
    assert counts[0] == 0 and counts[3] == 0 and abs(counts[2] - 0.75) < 0.01


def test_class_balanced_draws_each_label_equally():
    label = np.array([0] * 90 + [1] * 9 + [2])
    rows = ClassBalanced().draw(np.random.default_rng(0), SamplingIndex(np.arange(100), label=label), 30_000)
    freq = np.bincount(label[rows]) / len(rows)
    assert np.allclose(freq, 1 / 3, atol=0.015)


def test_uniform_is_seeded():
    index = SamplingIndex.whole(10)
    a = Uniform().draw(np.random.default_rng(5), index, 50)
    assert np.array_equal(a, Uniform().draw(np.random.default_rng(5), index, 50)) and a.max() < 10


def test_clip_windows():
    index = clip_windows([10, 3, 7], frames=4, stride=2)  # a window spans 7 frames
    assert np.bincount(index.record, minlength=3).tolist() == [4, 0, 1]
    assert index.box.tolist() == [[0, 7], [1, 8], [2, 9], [3, 10], [0, 7]]


def test_bad_indexes_and_samplers():
    with pytest.raises(ValueError, match="rows"):
        SamplingIndex(np.arange(3), label=np.arange(2))
    with pytest.raises(ValueError, match=">= 0"):
        SamplingIndex(np.arange(2), weight=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="weight column"):
        Weighted().draw(np.random.default_rng(0), SamplingIndex.whole(3), 1)


def test_lance_round_trip(tmp_path):
    pytest.importorskip("lance")
    index = SamplingIndex(np.array([0, 2, 2]), box=np.array([[0, 0, 4, 4], [1, 1, 5, 5], [2, 3, 6, 7]]),
                          label=np.array([1, 0, 1]), weight=np.array([0.5, 1.0, 2.0]))  # fmt: skip
    index.write_lance(str(tmp_path / "s.lance"))
    back = SamplingIndex.read_lance(str(tmp_path / "s.lance"))
    for k in ("record", "box", "label", "weight"):
        assert np.array_equal(getattr(back, k), getattr(index, k))
