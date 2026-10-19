"""Label-budget selection and frozen-feature probes with explicit sample roles."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def content_hash(value):
    """Hash a JSON-serializable configuration independently of dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def label_budget(manifest, records, fraction, subset_seed):
    """Select nested pixel-group subsets, accounting for validation labels too."""
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1].")
    by_id = {r.image_id: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Source IDs must be unique.")
    roles = {role: set(manifest["roles"][role]) for role in ("fit", "validation", "test", "trainval")}
    if roles["fit"] & roles["validation"] or roles["trainval"] & roles["test"]:
        raise ValueError("Experimental roles overlap.")
    if roles["fit"] | roles["validation"] != roles["trainval"]:
        raise ValueError("Refit pool must equal fitting plus validation.")
    selected, counts, group_counts = {}, {}, {}
    required = {by_id[i].label for i in roles["trainval"]}
    for role in ("fit", "validation"):
        groups = defaultdict(list)
        for image_id in sorted(roles[role]):
            groups[by_id[image_id].pixel_sha256].append(by_id[image_id])
        strata = defaultdict(list)
        for pixel_hash, group in groups.items():
            key = content_hash(["label-budget-v1", subset_seed, role, pixel_hash])
            strata[min(r.label for r in group)].append((key, pixel_hash))
        selected[role] = []
        chosen_groups = 0
        for items in strata.values():
            chosen = sorted(items)[: math.ceil(fraction * len(items))]
            chosen_groups += len(chosen)
            for _, pixel_hash in chosen:
                selected[role].extend(r.image_id for r in groups[pixel_hash])
        selected[role].sort()
        counts[role] = dict(sorted(Counter(by_id[i].label for i in selected[role]).items()))
        if set(counts[role]) != required:
            raise ValueError(f"A class is missing from the selected {role} role.")
        group_counts[role] = chosen_groups
    selected["trainval"] = sorted(selected["fit"] + selected["validation"])
    selected["test"] = sorted(roles["test"])
    return {
        "protocol": "label-budget-v1",
        "parent_manifest_sha256": content_hash(manifest),
        "subset_seed": subset_seed,
        "nominal_fraction": fraction,
        "actual_labeled_images": len(selected["trainval"]),
        "actual_fraction": len(selected["trainval"]) / len(roles["trainval"]),
        "class_counts": counts,
        "group_counts": group_counts,
        "roles": selected,
    }


def normalize_features(features):
    """Per-row normalization uses no statistics learned from other images."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Features must be a matrix.")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if features.ndim != 2 or not np.isfinite(features).all() or np.any(norms == 0):
        raise ValueError("Features must be a finite matrix without zero vectors.")
    return features / norms


def classification_metrics(labels, probabilities):
    """Compute task metrics in fixed class order; binary threshold is 0.5."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a matrix.")
    classes = np.arange(probabilities.shape[1])
    if (
        probabilities.ndim != 2
        or len(labels) != len(probabilities)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)
        or set(labels) != set(classes)
    ):
        raise ValueError("Invalid probabilities or incomplete evaluation classes.")
    predictions = (probabilities[:, 1] >= 0.5).astype(int) if len(classes) == 2 else probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=classes)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class_recall": recall_score(labels, predictions, labels=classes, average=None, zero_division=0).tolist(),
        "class_support": np.bincount(labels, minlength=len(classes)).tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    if len(classes) == 2:
        result.update(
            average_precision=float(average_precision_score(labels, probabilities[:, 1])),
            auroc=float(roc_auc_score(labels, probabilities[:, 1])),
            sensitivity=float(matrix[1, 1] / matrix[1].sum()),
            specificity=float(matrix[0, 0] / matrix[0].sum()),
            threshold=0.5,
        )
    return result


class FrozenProbe:
    """Small classifier on normalized features; no access to source datasets."""

    def __init__(self, method, parameter=None):
        if method not in ("linear", "knn", "prior"):
            raise ValueError(f"Unknown probe: {method}")
        self.method, self.parameter = method, parameter

    def fit(self, features, labels, image_ids):
        """Fit on only the explicitly supplied rows, ordered by stable image ID."""
        order = np.argsort(np.asarray(image_ids))
        self.features = np.asarray(features)[order]
        self.labels = np.asarray(labels, dtype=int)[order]
        self.fit_ids = np.asarray(image_ids)[order].tolist()
        self.classes = np.unique(self.labels)
        if not np.array_equal(self.classes, np.arange(len(self.classes))) or len(self.classes) < 2:
            raise ValueError("Fitting requires every task class in contiguous class-ID order.")
        if self.method == "linear":
            self.model = LogisticRegression(C=float(self.parameter), solver="lbfgs", tol=1e-6, max_iter=5000)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.model.fit(self.features, self.labels)
        elif self.method == "knn" and not 1 <= int(self.parameter) <= len(self.labels):
            raise ValueError("k must be between one and the fitting sample count.")
        return self

    def predict_proba(self, features):
        """Weighted cosine neighbors break similarity ties by sorted source ID."""
        features = np.asarray(features)
        if self.method == "linear":
            return self.model.predict_proba(features)
        if self.method == "prior":
            prior = np.bincount(self.labels, minlength=len(self.classes)) / len(self.labels)
            return np.tile(prior, (len(features), 1))
        similarity = features @ self.features.T
        neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, : int(self.parameter)]
        values = np.take_along_axis(similarity, neighbors, axis=1)
        weights = np.exp((values - values.max(axis=1, keepdims=True)) / 0.07)
        probabilities = np.zeros((len(features), len(self.classes)))
        for label in self.classes:
            probabilities[:, label] = (weights * (self.labels[neighbors] == label)).sum(axis=1)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_select_refit(method, features_by_role, labels_by_role, ids_by_role, primary_metric, frozen_parameter=None):
    """Select with fit/validation only, then refit on their permitted union.

    Test features and labels are intentionally absent from this interface.
    """
    grid = {"linear": [0.1, 1.0, 10.0], "knn": [5, 10, 20], "prior": [None]}[method]
    if method == "knn":
        grid = [k for k in grid if k <= len(ids_by_role["fit"])]
    if frozen_parameter is not None:
        grid = [frozen_parameter]
    candidates, best_parameter, best_score = [], None, -np.inf
    for parameter in grid:
        probe = FrozenProbe(method, parameter).fit(features_by_role["fit"], labels_by_role["fit"], ids_by_role["fit"])
        metrics = classification_metrics(
            labels_by_role["validation"], probe.predict_proba(features_by_role["validation"])
        )
        score = metrics[primary_metric]
        candidates.append({"parameter": parameter, "validation_score": score})
        if score > best_score + 1e-12:
            best_parameter, best_score = parameter, score
    if not candidates:
        raise ValueError("No eligible hyperparameter candidate.")
    # No parent-pool refit is possible: only the allowed two roles are supplied.
    ids = ids_by_role["fit"] + ids_by_role["validation"]
    if len(ids) != len(set(ids)):
        raise ValueError("Fitting and validation IDs overlap.")
    final = FrozenProbe(method, best_parameter).fit(
        np.concatenate([features_by_role[role] for role in ("fit", "validation")]),
        np.concatenate([labels_by_role[role] for role in ("fit", "validation")]),
        ids,
    )
    return final, {
        "selected_parameter": best_parameter,
        "selection_metric": primary_metric,
        "candidates": candidates,
        "frozen_from_primary": frozen_parameter is not None,
    }
