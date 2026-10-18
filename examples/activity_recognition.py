"""Subject-disjoint activity recognition with fixed random-forest settings.

Run: python examples/activity_recognition.py --output-dir results/activity
The test split is used only for final evaluation, never fitting or tuning.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from stable_datasets.timeseries import DailyAndSportsActivities


def window_features(series):
    """Summarize each channel along time, without learning from evaluation data."""
    x = np.asarray(series, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite time-by-channel matrix")
    return np.concatenate([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)])


def split_arrays(dataset, shape):
    """Validate windows and preserve subject IDs for a leakage check."""
    features, labels, subjects = [], [], set()
    for row in dataset:
        x = np.asarray(row["series"], dtype=np.float32)
        if x.shape != shape:
            raise ValueError(f"Unexpected window shape: {x.shape}; expected {shape}")
        features.append(window_features(x))
        labels.append(row["label"])
        subjects.add(int(row["subject_id"]))
    return np.asarray(features), np.asarray(labels), subjects


def evaluate(builder, download_dir, processed_cache_dir, seed):
    """Train a fixed classifier and report an untouched subject-disjoint test."""
    dataset = builder(download_dir=download_dir, processed_cache_dir=processed_cache_dir)
    shape = (builder.SEQUENCE_LENGTH, len(builder.CHANNEL_NAMES))
    x_train, y_train, train_subjects = split_arrays(dataset["train"], shape)
    x_test, y_test, test_subjects = split_arrays(dataset["test"], shape)
    if train_subjects & test_subjects:
        raise ValueError("Subject leakage between training and evaluation")
    results = {}
    for name, model in {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4),
    }.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=range(len(builder.ACTIVITY_NAMES))
            ).tolist(),
        }
    return {
        "dataset": builder.__name__,
        "train_count": len(y_train),
        "test_count": len(y_test),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "classes": list(builder.ACTIVITY_NAMES),
        "window_shape": shape,
        "seed": seed,
        "features": "per-channel mean, standard deviation, minimum, maximum",
        "model": "RandomForestClassifier(n_estimators=200, n_jobs=4); other defaults",
        "results": results,
    }


def main():
    """Run the activity dataset and save reproducible evaluation reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--processed-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for builder in (DailyAndSportsActivities,):
        report = evaluate(builder, args.download_dir, args.processed_cache_dir, args.seed)
        (args.output_dir / f"{builder.__name__}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(builder.__name__, report["results"]["random_forest"], flush=True)


if __name__ == "__main__":
    main()
