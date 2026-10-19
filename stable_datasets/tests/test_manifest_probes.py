"""Offline scientific invariants for label budgets and frozen transfer probes."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from benchmarks.protocols import ImageRecord, fingerprint_samples, isic2016_protocol, scene8_protocol
from benchmarks.transfer import primary_scene8_isic2016 as primary_runner
from benchmarks.transfer.protocol_probe import (
    FrozenProbe,
    classification_metrics,
    fit_select_refit,
    label_budget,
    normalize_features,
)
from benchmarks.transfer.scene8_isic2016 import evaluate_matrix, extract_features, feature_cache
from benchmarks.transfer.summarize_scene8_isic2016 import grouped_bootstrap_indices, verified_predictions


def _source():
    records = [
        ImageRecord(f"image_{label}_{i:03}", label, f"pixels_{label}_{i}") for label in range(3) for i in range(50)
    ]
    records += [ImageRecord("duplicate", 1, records[0].pixel_sha256)]
    return records


def test_nested_budget_counts_validation_and_preserves_conflicting_groups():
    records = _source()
    parent = asdict(scene8_protocol(records))
    budgets = [label_budget(parent, records, fraction, 1042) for fraction in (0.1, 0.25, 1)]
    for role in ("fit", "validation", "trainval"):
        assert set(budgets[0]["roles"][role]) <= set(budgets[1]["roles"][role]) <= set(budgets[2]["roles"][role])
        assert budgets[-1]["roles"][role] == parent["roles"][role]
    for budget in budgets:
        roles = budget["roles"]
        assert budget["actual_labeled_images"] == len(set(roles["fit"]) | set(roles["validation"]))
        assert budget["actual_labeled_images"] > len(roles["fit"])
        for role in ("fit", "validation"):
            assert ("duplicate" in roles[role]) == (records[0].image_id in roles[role])
        assert roles["test"] == parent["roles"]["test"]
    assert budgets[0] == label_budget(parent, list(reversed(records)), 0.1, 1042)
    assert budgets[0]["roles"] != label_budget(parent, records, 0.1, 1043)["roles"]


def test_budget_hash_order_matches_the_declared_canonical_json():
    records = _source()[:-1]
    parent = asdict(scene8_protocol(records))
    budget = label_budget(parent, records, 0.1, 1042)
    candidates = [r for r in records if r.image_id in parent["roles"]["validation"] and r.label == 0]
    ranked = sorted(
        candidates,
        key=lambda r: hashlib.sha256(
            json.dumps(["label-budget-v1", 1042, "validation", r.pixel_sha256], separators=(",", ":")).encode()
        ).hexdigest(),
    )
    expected = {r.image_id for r in ranked[: int(np.ceil(0.1 * len(candidates)))]}
    assert set(budget["roles"]["validation"]) & {r.image_id for r in candidates} == expected


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_budget_rejects_invalid_fractions(fraction):
    records = _source()
    with pytest.raises(ValueError, match="fraction"):
        label_budget(asdict(scene8_protocol(records)), records, fraction, 1042)


def test_binary_metrics_have_hand_computable_ap_auroc_and_fixed_threshold():
    labels = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = classification_metrics(labels, np.column_stack([1 - score, score]))
    assert metrics["average_precision"] == 1
    assert metrics["auroc"] == 1
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["sensitivity"] == 0.5 and metrics["specificity"] == 1
    tied = classification_metrics(labels, np.full((4, 2), 0.5))
    assert tied["confusion_matrix"] == [[0, 2], [0, 2]]
    assert tied["average_precision"] == tied["auroc"] == 0.5
    # Ranked labels [0,1,1,0]: AP=(1/2 + 2/3)/2, distinct from trapezoidal PR-AUC.
    nontrivial = classification_metrics(
        labels, np.column_stack([1 - np.array([0.9, 0.8, 0.1, 0.7]), [0.9, 0.8, 0.1, 0.7]])
    )
    assert nontrivial["average_precision"] == pytest.approx(7 / 12)


def test_multiclass_metric_names_and_smallest_class_tie():
    metrics = classification_metrics([0, 1, 2], np.full((3, 3), 1 / 3))
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == 1 / 3
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert "average_precision" not in metrics


def test_knn_ties_use_source_id_and_match_weighted_votes():
    features = normalize_features([[1, 0], [1, 0], [0, 1]])
    probe = FrozenProbe("knn", 1).fit(features, [1, 0, 1], ["b", "a", "c"])
    assert probe.predict_proba([[1, 0]]).tolist() == [[1, 0]]
    probe = FrozenProbe("knn", 3).fit(features, [1, 0, 1], ["b", "a", "c"])
    expected_positive = (1 + np.exp(-1 / 0.07)) / (2 + np.exp(-1 / 0.07))
    assert probe.predict_proba([[1, 0]])[0, 1] == pytest.approx(expected_positive)


@pytest.mark.parametrize("features", [[[0, 0]], [[np.nan, 1]], [1, 2]])
def test_invalid_features_fail(features):
    with pytest.raises(ValueError):
        normalize_features(features)


def test_selection_uses_only_supplied_budget_rows_and_refits_their_union():
    # Labels outside these two roles are not even accepted by the API.
    features = {"fit": normalize_features([[1, 0], [0, 1]]), "validation": normalize_features([[1, 0], [0, 1]])}
    labels = {"fit": np.array([0, 1]), "validation": np.array([0, 1])}
    ids = {"fit": ["fit_a", "fit_b"], "validation": ["val_a", "val_b"]}
    probe, selection = fit_select_refit("linear", features, labels, ids, "average_precision")
    assert selection["selected_parameter"] == 0.1  # deterministic tie: strongest regularization
    assert probe.fit_ids == ["fit_a", "fit_b", "val_a", "val_b"]
    assert len(selection["candidates"]) == 3
    frozen, frozen_selection = fit_select_refit("linear", features, labels, ids, "average_precision", 10)
    assert frozen.parameter == 10 and len(frozen_selection["candidates"]) == 1
    assert frozen_selection["frozen_from_primary"]


def test_extraction_keeps_tail_batch_and_verifies_source_pixels():
    rows = [
        {"image_id": str(i), "label": i % 2, "image": Image.new("RGB", (3, 3), (20 * i, 40, 60))} for i in range(5)
    ]
    records = fingerprint_samples(rows)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sizes = []

        def forward(self, images):
            self.sizes.append(len(images))
            return images.mean(dim=(2, 3))[:, :2]

    model = Encoder()

    def transform(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()

    result = extract_features(model, rows, records, 2, transform=transform, dimension=2)
    assert model.sizes == [2, 2, 1]
    assert result["image_ids"].tolist() == [str(i) for i in range(5)]
    assert result["labels"].tolist() == [0, 1, 0, 1, 0]
    rows[-1]["image"] = Image.new("RGB", (3, 3), "white")
    with pytest.raises(ValueError, match="differs from audit"):
        extract_features(model, rows, records, 2, transform=transform, dimension=2)


def test_feature_cache_binds_identity_payload_and_sample_alignment(tmp_path):
    records = [ImageRecord("a", 0, "pixels_a"), ImageRecord("b", 1, "pixels_b")]
    arrays = {"image_ids": np.array(["a", "b"]), "labels": np.array([0, 1]), "features": np.ones((2, 384), np.float32)}
    _, first = feature_cache(tmp_path, {"weights": "first"}, records, lambda: arrays)
    reopened, second = feature_cache(tmp_path, {"weights": "first"}, records)
    assert first == second and np.array_equal(reopened["features"], arrays["features"])
    with pytest.raises(FileNotFoundError):
        feature_cache(tmp_path, {"weights": "changed"}, records)
    with pytest.raises(ValueError, match="alignment"):
        feature_cache(tmp_path, {"weights": "first"}, list(reversed(records)))
    payload = tmp_path / first["key"] / "features.npz"
    payload.write_bytes(payload.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        feature_cache(tmp_path, {"weights": "first"}, records)


def test_invalid_computation_does_not_commit_a_completed_cache(tmp_path):
    records = [ImageRecord("a", 0, "pixels")]
    arrays = {"image_ids": np.array(["a"]), "labels": np.array([0]), "features": np.zeros((1, 384))}
    with pytest.raises(ValueError, match="zero"):
        feature_cache(tmp_path, {"test": True}, records, lambda: arrays)
    assert not list(tmp_path.rglob("metadata.json"))


def test_complete_matrix_preserves_budget_and_frozen_companion_selection(tmp_path, monkeypatch):
    scene = _source()
    train = [
        ImageRecord(f"train_{label}_{i}", label, f"train_pixels_{label}_{i}") for label in range(2) for i in range(50)
    ]
    test = [
        ImageRecord(f"test_{label}_{i}", label, f"test_pixels_{label}_{i}") for label in range(2) for i in range(10)
    ]
    test[0] = ImageRecord(test[0].image_id, 0, train[0].pixel_sha256)
    records = {"scene8": scene, "isic2016": train + test}
    manifests = {}
    for seed in (42, 43, 44):
        for manifest in (
            scene8_protocol(scene, seed),
            isic2016_protocol(train, test, seed),
            isic2016_protocol(train, test, seed, exclude_test_duplicates=True),
        ):
            manifests[(manifest.protocol, seed)] = asdict(manifest)
    arrays = {}
    for name, rows in records.items():
        vectors = np.eye(384)[[r.label for r in rows]] + 0.001
        arrays[name] = {
            "image_ids": np.array([r.image_id for r in rows]),
            "labels": np.array([r.label for r in rows]),
            "features": vectors,
        }
    matrix = evaluate_matrix(records, manifests, arrays, {name: {"test_fixture": True} for name in records}, tmp_path)
    assert matrix["complete"] and len(matrix["results"]) == 60
    assert len(verified_predictions(matrix, tmp_path)) == 60
    monkeypatch.setattr(primary_runner, "read_protocols", lambda _: (records, manifests))
    by_ids = {tuple(r.image_id for r in rows): name for name, rows in records.items()}
    monkeypatch.setattr(
        primary_runner, "feature_identity", lambda rows, *_: {"dataset": by_ids[tuple(r.image_id for r in rows)]}
    )
    monkeypatch.setattr(
        primary_runner, "feature_cache", lambda _, identity, __: (arrays[identity["dataset"]], {"fixture": True})
    )
    primary_results = primary_runner.run_primary(tmp_path, tmp_path, tmp_path / "primary")
    assert len(primary_results) == 6 and {r["method"] for r in primary_results} == {"linear"}
    for result in primary_results:
        original = next(
            r
            for r in matrix["results"]
            if r["protocol"] == result["protocol"]
            and r["seed"] == result["seed"]
            and r["method"] == "linear"
            and r["budget"]["nominal_fraction"] == 1
        )
        assert result["metrics"] == original["metrics"]
    assert "knn" not in (tmp_path / "primary" / "REPORT.md").read_text()
    for result in matrix["results"]:
        budget = result["budget"]
        assert budget["actual_labeled_images"] == len(budget["roles"]["trainval"])
        if result["protocol"] == "isic2016-official-pixel-groups-v2":
            assert result["selection"]["frozen_from_primary"]
            assert len(result["selection"]["candidates"]) == 1
        predictions = next(
            tmp_path.glob(
                f"{result['protocol']}-seed{result['seed']}-fraction{budget['nominal_fraction']:g}-{result['method']}/predictions.npz"
            )
        )
        with np.load(predictions, allow_pickle=False) as saved:
            assert classification_metrics(saved["labels"], saved["probabilities"]) == result["metrics"]


def test_group_bootstrap_preserves_duplicate_clusters_and_class_support():
    labels = np.array([0, 1, 0, 0, 1, 1])
    pixels = ["conflict", "conflict", "negative_a", "negative_b", "positive_a", "positive_b"]
    draws = list(grouped_bootstrap_indices(labels, pixels, repeats=20, seed=42))
    for indices in draws:
        assert np.sum(indices == 0) == np.sum(indices == 1)
        assert np.bincount(labels[indices]).tolist() == [3, 3]
    assert len({tuple(indices) for indices in draws}) > 1
    assert all(
        np.array_equal(a, b) for a, b in zip(draws, grouped_bootstrap_indices(labels, pixels, repeats=20, seed=42))
    )
