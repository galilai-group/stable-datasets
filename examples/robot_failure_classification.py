"""Fixed random-forest validation on each separate robot learning problem.

Run: python examples/robot_failure_classification.py --output-dir results/robot
No official train/test or session IDs exist. Three-fold out-of-fold predictions
are a small-sample pipeline check, not evidence of generalization to new robots.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from stable_datasets.timeseries import RobotExecutionFailures


def arrays_and_groups(dataset):
    """Keep exact duplicate signal windows together, regardless of their labels."""
    features, labels, groups = [], [], []
    for row in dataset:
        x = np.asarray(row["series"], dtype=np.float32)
        if x.shape != (15, 6) or not np.isfinite(x).all():
            raise ValueError("Expected finite 15-by-6 windows")
        features.append(np.concatenate([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)]))
        labels.append(row["label"])
        groups.append(hashlib.sha256(x.tobytes()).hexdigest())
    return np.asarray(features), np.asarray(labels), np.asarray(groups)


def folds(y, groups, seed):
    """Construct fixed folds and explicitly check duplicate isolation and coverage."""
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Duplicate signals cross evaluation folds")
        if set(y[train]) != set(y):
            raise ValueError("A training fold is missing a class")
    if sorted(np.concatenate([test for _, test in result]).tolist()) != list(range(len(y))):
        raise ValueError("Evaluation does not cover each sample exactly once")
    return result


def evaluate(config, download_dir, processed_cache_dir, seed):
    """Evaluate fixed models without fitting or selecting anything on test folds."""
    dataset = RobotExecutionFailures(
        config_name=config, split="all", download_dir=download_dir, processed_cache_dir=processed_cache_dir
    )
    x, y, groups = arrays_and_groups(dataset)
    partition = folds(y, groups, seed)
    names = RobotExecutionFailures.CLASS_NAMES[config]
    results = {}
    for model_name in ("majority", "random_forest"):
        predictions = np.full(len(y), -1, dtype=np.int32)
        for train, test in partition:
            model = (
                DummyClassifier(strategy="most_frequent")
                if model_name == "majority"
                else RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4)
            )
            model.fit(x[train], y[train])
            predictions[test] = model.predict(x[test])
        results[model_name] = {
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=range(len(names)), average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predictions, labels=range(len(names))).tolist(),
            "out_of_fold_predictions": predictions.tolist(),
        }
    return {
        "dataset": "RobotExecutionFailures",
        "config": config,
        "count": len(y),
        "unique_signal_groups": len(set(groups)),
        "classes": list(names),
        "seed": seed,
        "protocol": "3-fold StratifiedGroupKFold; exact duplicate windows grouped; tasks evaluated separately",
        "features": "per-channel mean, standard deviation, minimum, maximum",
        "model": "RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed); other defaults",
        "folds": [{"train_sample_ids": train.tolist(), "test_sample_ids": test.tolist()} for train, test in partition],
        "results": results,
    }


def main():
    """Validate all five tasks using the same predefined model settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--processed-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config in RobotExecutionFailures.CLASS_NAMES:
        report = evaluate(config, args.download_dir, args.processed_cache_dir, args.seed)
        (args.output_dir / f"RobotExecutionFailures_{config}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(config, report["count"], report["results"]["random_forest"]["accuracy"], flush=True)


if __name__ == "__main__":
    main()
