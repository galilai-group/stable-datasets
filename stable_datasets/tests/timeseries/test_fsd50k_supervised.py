"""Offline tests for the FSD50K supervised baseline example (``examples/fsd50k_supervised.py``)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch


_EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(_EXAMPLES))  # the example imports the shared front end from fsd_simclr.py
_SPEC = importlib.util.spec_from_file_location("fsd50k_supervised", _EXAMPLES / "fsd50k_supervised.py")
example = importlib.util.module_from_spec(_SPEC)
sys.modules["fsd50k_supervised"] = example
_SPEC.loader.exec_module(example)


def write_features(root: Path, split: str, clips, labels, n_classes=4):
    """Store ``clips`` (list of ``[T, 96]`` arrays) with multi-label ``labels`` in the example's layout."""
    root.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in clips])])
    meta = [
        {"filename": f"{split}{i}.wav", "labels": list(map(int, ls)), "frames": len(c), "offset": int(o)}
        for i, (c, ls, o) in enumerate(zip(clips, labels, offsets))
    ]
    np.save(root / f"{split}.npy", np.concatenate(clips).astype(np.float32))
    names = [f"class{i}" for i in range(n_classes)]
    (root / f"{split}.json").write_text(json.dumps({"label_names": names, "clips": meta}))
    return example.Features(root, split)


def toy_split(rng, root, split, n_clips, n_classes=4):
    """Clips whose labels set a per-class offset in a few mel bands, so the task is learnable."""
    clips, labels = [], []
    for _ in range(n_clips):
        label = sorted(rng.choice(n_classes, size=rng.integers(1, 3), replace=False).tolist())
        clip = rng.normal(-4, 0.3, (int(rng.integers(101, 260)), 96))
        for c in label:
            clip[:, 10 * c : 10 * c + 8] += 2.0
        clips.append(clip)
        labels.append(label)
    return write_features(root, split, clips, labels, n_classes)


# -- Features ----------------------------------------------------------------


@pytest.mark.parametrize("seconds", [0.3, 1.0, 1.2, 7.5])
def test_expected_frames_match_front_end(seconds):
    wave = np.random.default_rng(0).standard_normal(int(seconds * 22050))
    assert len(example.simclr.LogMel()(wave)) == example.expected_frames(len(wave))


def test_build_features_writes_multi_hot_splits(tmp_path):
    rng = np.random.default_rng(1)
    rows = {
        "train": [(0.5, [0, 2]), (2.0, [1])],
        "validation": [(1.2, [3])],
        "test": [(1.0, [0, 1, 3])],
    }

    class FakeBuilder:
        def __init__(self, split, config_name):
            assert config_name == "22khz"
            items = rows[split]
            series = [rng.standard_normal((int(s * 22050), 1)).astype(np.float32).tolist() for s, _ in items]
            self.table = pa.table(
                {
                    "series": pa.array(series, type=pa.list_(pa.list_(pa.float32()))),
                    "sample_rate": [22050] * len(items),
                    "labels": [labels for _, labels in items],
                    "filename": [f"{split}{i}.wav" for i in range(len(items))],
                }
            )
            label = type("Label", (), {"names": ["a", "b", "c", "d"]})
            self.features = {"labels": type("Labels", (), {"feature": label})}

    counts = example.build_features(tmp_path, builder=FakeBuilder)
    assert counts == {"train": 2, "validation": 1, "test": 1}
    train = example.Features(tmp_path, "train")
    assert train.targets.tolist() == [[1, 0, 1, 0], [0, 1, 0, 0]]
    assert train.lengths.tolist() == [example.simclr.PATCH_LEN, 1 + 2 * 22050 // 220]
    assert train.offsets.tolist() == [0, train.lengths[0]]
    assert train.frames.shape == (train.lengths.sum(), 96)
    assert np.isfinite(train.frames).all() and train.frames[:].std() > 0
    assert example.Features(tmp_path, "test").targets.tolist() == [[1, 1, 0, 1]]
    assert example.Features(tmp_path, "validation").label_names == ["a", "b", "c", "d"]


def test_build_features_rejects_wrong_sample_rate(tmp_path):
    class FakeBuilder:
        def __init__(self, split, config_name):
            self.table = pa.table(
                {
                    "series": pa.array([[[0.1], [0.2]]], type=pa.list_(pa.list_(pa.float32()))),
                    "sample_rate": [44100],
                    "labels": [[0]],
                    "filename": ["a.wav"],
                }
            )
            label = type("Label", (), {"names": ["a"]})
            self.features = {"labels": type("Labels", (), {"feature": label})}

    with pytest.raises(ValueError, match="22050"):
        example.build_features(tmp_path, builder=FakeBuilder)


def test_training_patches_cover_every_clip_with_half_overlap(tmp_path):
    features = write_features(tmp_path, "train", [np.zeros((101, 96)), np.ones((250, 96))], [[0], [1, 2]])
    starts, clips = example.patch_index(features)
    assert starts.tolist() == [0, 101, 151, 201]  # second clip starts at frame 101; hop of 50 frames
    assert clips.tolist() == [0, 1, 1, 1]
    dataset = example.Patches(features, mean=0.5, std=2.0)
    patch, target = dataset[2]
    assert patch.shape == (1, 101, 96)
    torch.testing.assert_close(patch, torch.full((1, 101, 96), 0.25))
    assert target.tolist() == [0, 1, 1, 0]


# -- Model and metrics -------------------------------------------------------


def test_vgg_like_matches_paper_size_and_shapes():
    model = example.VGGLike(200)
    assert sum(p.numel() for p in model.parameters()) == 265_992  # paper Table VI: 0.27M
    assert model(torch.randn(3, 1, 101, 96)).shape == (3, 200)


def test_clip_scores_average_patch_probabilities(tmp_path):
    features = write_features(tmp_path, "test", [np.zeros((101, 96)), np.ones((151, 96))], [[0], [1]], n_classes=2)

    class MeanModel(torch.nn.Module):
        def forward(self, x):
            value = x.mean(dim=(1, 2, 3))
            return torch.stack([value, -value], dim=1)

    scores = example.clip_scores(MeanModel(), features, mean=0.0, std=1.0, device="cpu", batch_size=2)
    sigmoid = 1 / (1 + np.exp(-1.0))
    np.testing.assert_allclose(scores, [[0.5, 0.5], [sigmoid, 1 - sigmoid]], rtol=1e-6)


def test_metrics_match_sklearn_and_d_prime_definition():
    from scipy.stats import norm
    from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score

    rng = np.random.default_rng(2)
    targets = np.zeros((40, 5))
    for row in targets:
        row[rng.choice(4, 2, replace=False)] = 1  # two labels per clip: lwlrap equals sklearn's LRAP
    scores = rng.random((40, 5)) + targets * 0.5
    result = example.metrics(targets, scores)
    assert result["classes"] == 4  # class 4 has no positives and is skipped
    assert result["mAP"] == pytest.approx(average_precision_score(targets[:, :4], scores[:, :4], average="macro"))
    auc = roc_auc_score(targets[:, :4], scores[:, :4], average=None)
    assert result["d_prime"] == pytest.approx(np.mean(np.sqrt(2) * norm.ppf(auc)))
    assert result["lwlrap"] == pytest.approx(label_ranking_average_precision_score(targets, scores))


def test_lwlrap_weights_every_label_equally():
    targets = np.array([[1, 0, 0], [1, 1, 0]])
    scores = np.array([[0.1, 0.9, 0.5], [0.9, 0.8, 0.1]])
    # Clip 1: its label ranks 3rd (precision 1/3). Clip 2: both labels rank 1st and 2nd (precision 1, 1).
    assert example.lwlrap(targets, scores) == pytest.approx((1 / 3 + 1 + 1) / 3)


def test_class_averaged_lwlrap_weighs_every_class_equally():
    targets = np.array([[1, 0, 0], [1, 1, 0]])
    scores = np.array([[0.1, 0.9, 0.5], [0.9, 0.8, 0.1]])
    # Class 0 gets precisions 1/3 and 1, class 1 gets 1; class 2 has no labels and is left out.
    assert example.lwlrap(targets, scores, average="class") == pytest.approx(((1 / 3 + 1) / 2 + 1) / 2)
    with pytest.raises(ValueError):
        example.lwlrap(targets, scores, average="clip")


def test_plateau_halves_lr_after_five_epochs_and_stops_after_ten():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    plateau = example.Plateau(optimizer)
    improved = [plateau.step(score) for score in [0.1, 0.2] + [0.2] * 5]
    assert improved == [True, True] + [False] * 5
    assert optimizer.param_groups[0]["lr"] == 0.5 and not plateau.should_stop
    for score in [0.15, 0.2, 0.1, 0.1]:  # ties are not improvements ("no tolerance")
        plateau.step(score)
    assert optimizer.param_groups[0]["lr"] == 0.5 and not plateau.should_stop
    plateau.step(0.1)  # 10th epoch without improvement, 5th since the last decay
    assert optimizer.param_groups[0]["lr"] == 0.25 and plateau.should_stop
    assert plateau.step(0.3) and plateau.since_best == 0


# -- Training ----------------------------------------------------------------


def test_training_selects_on_validation_and_never_uses_test_labels(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / "features"
    toy_split(rng, root, "train", 24)
    toy_split(rng, root, "validation", 8)
    test = toy_split(rng, root, "test", 8)
    (tmp_path / "a").mkdir()
    first = example.train(root, tmp_path / "a", seed=0, device=torch.device("cpu"), epochs=3, batch_size=16, workers=0)

    meta = json.loads((root / "test.json").read_text())
    shuffled = [meta["clips"][i]["labels"] for i in rng.permutation(len(meta["clips"]))]
    for clip, labels in zip(meta["clips"], shuffled):
        clip["labels"] = labels
    (root / "test.json").write_text(json.dumps(meta))
    (tmp_path / "b").mkdir()
    second = example.train(
        root, tmp_path / "b", seed=0, device=torch.device("cpu"), epochs=3, batch_size=16, workers=0
    )

    history = json.loads((tmp_path / "a" / "history.json").read_text())
    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["validation"] == second["validation"]
    assert first["selected_epoch"] == max(history, key=lambda row: row["validation"]["mAP"])["epoch"]
    np.testing.assert_array_equal(
        np.load(tmp_path / "a" / "predictions.npz")["scores"], np.load(tmp_path / "b" / "predictions.npz")["scores"]
    )
    assert first["test"]["mAP"] > 0.6 and first["parameters"] == 215_620  # 4 outputs instead of 200
    assert (tmp_path / "a" / "model.pt").exists() and test.targets.shape == (8, 4)
    saved = json.loads((tmp_path / "a" / "metrics.json").read_text())
    assert saved["selected_epoch"] == first["selected_epoch"] and set(saved["test"]) == {
        "mAP",
        "d_prime",
        "lwlrap",
        "lwlrap_class",
        "classes",
    }


def test_cli_writes_provenance(tmp_path):
    rng = np.random.default_rng(4)
    root = tmp_path / "features"
    for split, n in (("train", 6), ("validation", 4), ("test", 4)):
        toy_split(rng, root, split, n)
    example.main(["train", "--features", str(root), "--out", str(tmp_path / "run"), "--epochs", "1", "--workers", "0"])
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["args"]["epochs"] == 1
    assert (
        config["script_sha256"]
        == __import__("hashlib").sha256((_EXAMPLES / "fsd50k_supervised.py").read_bytes()).hexdigest()
    )
