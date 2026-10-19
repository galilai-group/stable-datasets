"""Verify saved predictions and create the English transfer experiment report."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from benchmarks.transfer.primary_scene8_isic2016 import write_main_report
from benchmarks.transfer.protocol_probe import classification_metrics
from benchmarks.transfer.scene8_isic2016 import ASSET_LOCK, file_hash, read_protocols, write_json


CLASSES = {
    "scene8": ["coast", "forest", "highway", "insidecity", "mountain", "opencountry", "street", "tallbuilding"],
    "isic2016": ["benign", "malignant"],
}


def grouped_bootstrap_indices(labels, pixels, repeats=2000, seed=20261018):
    """Resample exact-pixel groups within fixed label-composition strata."""
    groups, strata = defaultdict(list), defaultdict(list)
    for index, pixel in enumerate(pixels):
        groups[pixel].append(index)
    labels = np.asarray(labels)
    for pixel in sorted(groups):
        indices = groups[pixel]
        strata[tuple(sorted(labels[indices].tolist()))].append(indices)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        yield np.concatenate(
            [
                np.asarray(groups_in_stratum[i])
                for groups_in_stratum in strata.values()
                for i in rng.integers(0, len(groups_in_stratum), size=len(groups_in_stratum))
            ]
        )


def _run_name(result):
    return f"{result['protocol']}-seed{result['seed']}-fraction{result['budget']['nominal_fraction']:g}-{result['method']}"


def verified_predictions(matrix, experiment_dir):
    """Recompute every metric and check persisted payloads before summarizing."""
    if not matrix["complete"] or len(matrix["results"]) != 60:
        raise ValueError("Only a complete declared matrix can be summarized.")
    predictions = {}
    for result in matrix["results"]:
        name = _run_name(result)
        if name in predictions:
            raise ValueError("Repeated experiment condition.")
        path = Path(experiment_dir) / name / "predictions.npz"
        if file_hash(path) != result["prediction_sha256"]:
            raise ValueError(f"Prediction file checksum differs: {name}")
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in ("image_ids", "labels", "probabilities")}
        if values["image_ids"].tolist() != result["budget"]["roles"]["test"]:
            raise ValueError(f"Prediction ID coverage differs: {name}")
        if classification_metrics(values["labels"], values["probabilities"]) != result["metrics"]:
            raise ValueError(f"Stored metrics cannot be recomputed: {name}")
        predictions[name] = values
    return predictions


def write_tables(results, predictions, output_dir):
    """Store per-run scores, aggregate statistics, and all test predictions."""
    output_dir = Path(output_dir)
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "average_precision",
        "auroc",
        "sensitivity",
        "specificity",
    )
    columns = [
        "run",
        "dataset",
        "protocol",
        "seed",
        "method",
        "nominal_fraction",
        "fit_images",
        "validation_images",
        "total_labeled_images",
        "actual_fraction",
        "selected_parameter",
        *metric_keys,
    ]
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            budget = result["budget"]
            writer.writerow(
                {
                    "run": _run_name(result),
                    **{key: result[key] for key in ("dataset", "protocol", "seed", "method")},
                    **{key: budget[key] for key in ("nominal_fraction", "actual_fraction")},
                    "fit_images": len(budget["roles"]["fit"]),
                    "validation_images": len(budget["roles"]["validation"]),
                    "total_labeled_images": budget["actual_labeled_images"],
                    "selected_parameter": result["selection"]["selected_parameter"],
                    **{key: result["metrics"].get(key, "") for key in metric_keys},
                }
            )
    groups = defaultdict(list)
    for result in results:
        groups[(result["dataset"], result["protocol"], result["budget"]["nominal_fraction"], result["method"])].append(
            result
        )
    aggregates = []
    for (dataset, protocol, fraction, method), runs in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "protocol": protocol,
            "nominal_fraction": fraction,
            "method": method,
            "seeds": len(runs),
            "labeled_images_min": min(r["budget"]["actual_labeled_images"] for r in runs),
            "labeled_images_max": max(r["budget"]["actual_labeled_images"] for r in runs),
        }
        for metric in metric_keys:
            values = [r["metrics"][metric] for r in runs if metric in r["metrics"]]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
        aggregates.append(row)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    # All scores/labels/IDs remain reviewable without distributing any images.
    with gzip.open(output_dir / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "image_id", "label", "num_classes", *[f"probability_{i}" for i in range(8)]])
        for name, values in predictions.items():
            for image_id, label, probability in zip(values["image_ids"], values["labels"], values["probabilities"]):
                writer.writerow(
                    [name, image_id, label, len(probability), *probability.tolist(), *[""] * (8 - len(probability))]
                )
    with (output_dir / "errors_seed42.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "method", "image_id", "source_class", "predicted_class", "predicted_class_probability"]
        )
        for result in results:
            if (
                result["seed"] != 42
                or result["budget"]["nominal_fraction"] != 1
                or result["method"] == "prior"
                or "official" in result["protocol"]
            ):
                continue
            values = predictions[_run_name(result)]
            probability = values["probabilities"]
            predicted = (
                (probability[:, 1] >= 0.5).astype(int) if probability.shape[1] == 2 else probability.argmax(axis=1)
            )
            errors = np.flatnonzero(predicted != values["labels"])
            errors = sorted(errors, key=lambda i: (-probability[i, predicted[i]], values["image_ids"][i]))
            for i in errors:
                classes = CLASSES[result["dataset"]]
                writer.writerow(
                    [
                        result["dataset"],
                        result["method"],
                        values["image_ids"][i],
                        classes[values["labels"][i]],
                        classes[predicted[i]],
                        float(probability[i, predicted[i]]),
                    ]
                )
    return aggregates


def uncertainty(results, predictions, records):
    """Conditional group-bootstrap intervals with shared draws for differences."""
    report = {
        "replicates": 2000,
        "seed": 20261018,
        "interpretation": "Conditional percentile intervals; exclude patient, partition, training, and pretraining uncertainty.",
        "results": {},
    }
    for dataset in CLASSES:
        runs = [
            r
            for r in results
            if r["dataset"] == dataset
            and r["seed"] == 42
            and r["budget"]["nominal_fraction"] == 1
            and r["method"] != "prior"
        ]
        by_key = {(r["protocol"], r["method"]): predictions[_run_name(r)] for r in runs}
        first = next(iter(by_key.values()))
        for values in by_key.values():
            if not np.array_equal(first["image_ids"], values["image_ids"]) or not np.array_equal(
                first["labels"], values["labels"]
            ):
                raise ValueError("Paired bootstrap predictions must have identical test order and targets.")
        pixel_map = {r.image_id: r.pixel_sha256 for r in records[dataset]}
        pixels = [pixel_map[i] for i in first["image_ids"]]
        samples = {key: [] for key in by_key}
        for indices in grouped_bootstrap_indices(first["labels"], pixels):
            for key, values in by_key.items():
                labels, probabilities = values["labels"][indices], values["probabilities"][indices]
                score = (
                    average_precision_score(labels, probabilities[:, 1])
                    if dataset == "isic2016"
                    else balanced_accuracy_score(labels, probabilities.argmax(axis=1))
                )
                samples[key].append(float(score))
        for (protocol, method), values in samples.items():
            report["results"][f"{protocol}/{method}"] = {"percentile_95": np.percentile(values, [2.5, 97.5]).tolist()}
        primary = "scene8-pixel-groups-v1" if dataset == "scene8" else "isic2016-train-filtered-pixel-groups-v2"
        difference = np.asarray(samples[(primary, "knn")]) - samples[(primary, "linear")]
        report["results"][f"{dataset}/knn_minus_linear"] = {
            "percentile_95": np.percentile(difference, [2.5, 97.5]).tolist()
        }
        if dataset == "isic2016":
            for method in ("linear", "knn"):
                difference = (
                    np.asarray(samples[("isic2016-official-pixel-groups-v2", method)]) - samples[(primary, method)]
                )
                report["results"][f"isic2016/official_minus_filtered/{method}"] = {
                    "percentile_95": np.percentile(difference, [2.5, 97.5]).tolist()
                }
    return report


def figures(results, predictions, output_dir):
    """Create standalone publication-style plots with explicit metric names."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/project1/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    for axis, dataset in zip(axes, CLASSES):
        primary = "balanced_accuracy" if dataset == "scene8" else "average_precision"
        for method, color in (("prior", "#7c8796"), ("linear", "#1667b7"), ("knn", "#ce6a17")):
            xs, ys, deviations = [], [], []
            for fraction in (0.1, 0.25, 1):
                runs = [
                    r
                    for r in results
                    if r["dataset"] == dataset
                    and r["method"] == method
                    and r["budget"]["nominal_fraction"] == fraction
                    and "official" not in r["protocol"]
                ]
                xs.append(np.mean([r["budget"]["actual_labeled_images"] for r in runs]))
                scores = [r["metrics"][primary] for r in runs]
                ys.append(np.mean(scores))
                deviations.append(np.std(scores, ddof=1))
            axis.errorbar(xs, ys, yerr=deviations, color=color, marker="o", capsize=3, label=method)
        axis.set(
            title="Scene8" if dataset == "scene8" else "ISIC2016 (filtered)",
            xlabel="Labeled images, including validation",
            ylabel="Balanced accuracy" if dataset == "scene8" else "Average precision",
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Frozen DINO ViT-S/16: mean and sample SD across three partition/subset seeds", fontsize=11)
    fig.savefig(Path(output_dir) / "label_efficiency.png", dpi=200)
    plt.close(fig)
    for dataset in CLASSES:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        for axis, method in zip(axes, ("linear", "knn")):
            result = next(
                r
                for r in results
                if r["dataset"] == dataset
                and r["method"] == method
                and r["seed"] == 42
                and r["budget"]["nominal_fraction"] == 1
                and "official" not in r["protocol"]
            )
            matrix = np.asarray(result["metrics"]["confusion_matrix"])
            axis.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for row, column in np.ndindex(matrix.shape):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[row, column] / matrix[row].sum() > 0.5 else "black",
                )
            axis.set(
                title=method,
                xlabel="Predicted class",
                ylabel="Source class",
                xticks=range(len(matrix)),
                yticks=range(len(matrix)),
            )
            axis.set_xticklabels(CLASSES[dataset], rotation=55, ha="right", fontsize=8)
            axis.set_yticklabels(CLASSES[dataset], fontsize=8)
        fig.suptitle(f"{dataset}: seed 42, full budget; counts with row-normalized shading")
        fig.savefig(Path(output_dir) / f"{dataset}_confusion.png", dpi=200)
        plt.close(fig)


def report_markdown(aggregates, matrix, intervals, output_dir):
    """Generate a compact report without inventing explanations for outcomes."""
    rows = [
        "# Supplementary Verification: Scene8 and ISIC2016",
        "",
        "## Contribution and verified execution",
        "",
        "Two native builders preserve original JPEG bytes, labels, stable IDs, and source memberships. "
        "A complete audit decoded 3,967 images and found exact duplicates, including a Scene8 label conflict "
        "and three ISIC train/test duplicate pairs. The primary ISIC protocol excludes the three training "
        "duplicates; its official companion shares validation and restores them only for the declared full-budget comparison.",
        "",
        "The locked matrix contains 42 learned-model conditions and 18 prior baselines. Every saved prediction "
        "payload was checksum-verified and every reported metric recomputed. Raw images and pretrained weights "
        "are not distributed with these results.",
        "",
        "## Evaluation",
        "",
        "The original ImageNet-1K DINO ViT-S/16 remains frozen. Both probes use the same 384-dimensional "
        "final-block CLS features and deterministic 224-pixel center crops. The linear probe selects C from "
        "[0.1, 1, 10]; weighted cosine kNN selects k from [5, 10, 20] at temperature 0.07. Selection uses "
        "balanced accuracy for Scene8 and AP for ISIC. AP uses scikit-learn's definition, not trapezoidal PR-AUC.",
        "",
        "Nominal 10%, 25%, and 100% subsets preserve exact-pixel groups and are nested within each role. "
        "Validation labels count toward each budget. Hyperparameters are selected on the permitted fitting/validation "
        "subsets, and final classifiers refit on only their union. ISIC uses a fixed 0.5 decision threshold. "
        "The official companion reuses the filtered full-budget hyperparameter without retuning.",
        "",
        "## Results",
        "",
        "Values are mean +/- sample SD across seeds 42, 43, and 44. Seed variation concerns partitions/subsets; "
        "the pretrained encoder is identical. These SDs are not confidence intervals.",
        "",
        "| Dataset/protocol | Nominal fraction | Method | Labeled images | Primary metric | Top-1 accuracy |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        metric = "balanced_accuracy" if row["dataset"] == "scene8" else "average_precision"
        protocol = (
            "official companion"
            if "official" in row["protocol"]
            else ("filtered" if row["dataset"] == "isic2016" else "grouped")
        )
        rows.append(
            f"| {row['dataset']}/{protocol} | {row['nominal_fraction']:.0%} | {row['method']} | "
            f"{row['labeled_images_min']}-{row['labeled_images_max']} | {row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f} | "
            f"{row['accuracy_mean']:.4f} +/- {row['accuracy_sd']:.4f} |"
        )
    rows += [
        "",
        "![Label efficiency](label_efficiency.png)",
        "",
        "![Scene8 confusion matrices](scene8_confusion.png)",
        "",
        "![ISIC confusion matrices](isic2016_confusion.png)",
        "",
        "## Findings and error analysis",
        "",
        "The linear classifier's mean primary score increases with the label budget on both tasks in this "
        "declared experiment. kNN does not improve monotonically on ISIC, so more labeled examples do not "
        "guarantee improvement for every fixed-feature classifier/selection protocol. These observations do "
        "not isolate a causal effect of domain or establish a general algorithm ranking.",
        "",
        "The three full-budget ISIC linear runs refit on the same 897-image pool after selecting the same C. "
        "Their deterministic predictions coincide, explaining the zero seed SD. This is not evidence from "
        "three independently pretrained encoders or three independent test sets. kNN can differ because the "
        "inner partitions select different neighbor counts.",
        "",
        "`errors_seed42.csv` contains every error from the full-budget primary linear/kNN runs at seed 42, "
        "ranked by predicted-class probability within each dataset/method. This explicit rule avoids selecting "
        "only visually convenient failures. Image IDs link back to the original data without redistributing photographs.",
        "",
        "## Uncertainty and limitations",
        "",
        "The companion `uncertainty.json` reports seed-42 full-budget conditional percentile intervals from "
        f"{intervals['replicates']:,} stratified exact-pixel-group bootstrap draws. Paired differences share draws. "
        "These intervals exclude patient, partition, training, and pretraining uncertainty. Scene8 test sets "
        "overlap across seeds; ISIC reuses its 379-image official test set. Neither supports treating seed runs "
        "as independent samples from a population of trained encoders.",
        "",
        "The smallest ISIC validation subset is small, so hyperparameter selection is noisy. Pixel duplicates "
        "do not establish patient independence or rule out near-duplicates and ImageNet overlap. Filtering "
        "uses test pixels in a disclosed transductive audit, not diagnoses or scores. The Scene8 conflicting "
        "labels are retained. Center cropping can discard peripheral scene or lesion information. These are "
        "transfer-validation results, not clinical validation or target-dataset DINO pretraining.",
        "",
        "## Reproduction and evidence",
        "",
        "Run the source checkout commands in `benchmarks/transfer/SCENE8_ISIC2016_README.md`. "
        "`runs.csv` contains every condition and parameter; `summary.csv` contains aggregates; "
        "`predictions.csv.gz` contains every test ID, target, and class probability. "
        "`evidence.json` binds these files to feature caches, code, weights, manifests, and software versions. "
        "`matrix.json.gz` additionally retains every selected subset and validation candidate score.",
        "",
        "A known-method optimization smoke test is recorded separately and must not be interpreted as "
        "converged representation learning. Course acceptance of frozen transfer versus complete target-data "
        "SSL pretraining still depends on the instructor's interpretation.",
        "",
    ]
    isic = next(
        r
        for r in matrix["results"]
        if r["dataset"] == "isic2016"
        and r["method"] == "linear"
        and r["seed"] == 42
        and r["budget"]["nominal_fraction"] == 1
        and "official" not in r["protocol"]
    )
    cm = isic["metrics"]["confusion_matrix"]
    detail = (
        f"At the fixed 0.5 threshold, the seed-42 ISIC linear classifier correctly identifies {cm[1][1]} of "
        f"{sum(cm[1])} malignant examples and misses {cm[1][0]}; it produces {cm[0][1]} false positives. "
        "Its relatively high overall accuracy therefore must not be interpreted as high malignant sensitivity. "
        "AP measures ranking across thresholds, while this confusion matrix describes the predeclared single threshold."
    )
    rows[rows.index("## Uncertainty and limitations") : rows.index("## Uncertainty and limitations")] = [detail, ""]
    (Path(output_dir) / "SUPPLEMENT.md").write_text("\n".join(rows), encoding="utf-8")
    write_main_report(matrix["results"], output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(".cache/project1/experiments"))
    parser.add_argument("--protocol-dir", type=Path, default=Path(".cache/project1/verified-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/source/datasets/experiments/scene8_isic2016"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.experiment_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    predictions = verified_predictions(matrix, args.experiment_dir)
    records, _ = read_protocols(args.protocol_dir)
    # A deterministic full-pool refit with identical selected C must coincide.
    linear_runs = [
        r
        for r in matrix["results"]
        if r["protocol"] == "isic2016-train-filtered-pixel-groups-v2"
        and r["method"] == "linear"
        and r["budget"]["nominal_fraction"] == 1
    ]
    if len({r["selection"]["selected_parameter"] for r in linear_runs}) == 1:
        if not all(
            np.array_equal(
                predictions[_run_name(linear_runs[0])]["probabilities"], predictions[_run_name(r)]["probabilities"]
            )
            for r in linear_runs[1:]
        ):
            raise ValueError("Identical deterministic ISIC refits unexpectedly differ.")
    for result in matrix["results"]:
        values = predictions[_run_name(result)]
        by_id = {r.image_id: r.label for r in records[result["dataset"]]}
        if values["labels"].tolist() != [by_id[i] for i in values["image_ids"]]:
            raise ValueError("Saved prediction targets differ from audited source labels.")
    aggregates = write_tables(matrix["results"], predictions, args.output_dir)
    (args.output_dir / "matrix.json.gz").write_bytes(gzip.compress(matrix_path.read_bytes(), mtime=0))
    intervals = uncertainty(matrix["results"], predictions, records)
    write_json(args.output_dir / "uncertainty.json", intervals)
    figures(matrix["results"], predictions, args.output_dir)
    report_markdown(aggregates, matrix, intervals, args.output_dir)
    evidence = {
        "matrix_sha256": file_hash(matrix_path),
        "encoder_assets": json.loads(ASSET_LOCK.read_text()),
        "executed_source_sha256": {
            name.replace("\\", "/"): digest for name, digest in matrix["source_sha256"].items()
        },
        "summary_source_sha256": file_hash(__file__),
        "primary_runner_sha256": file_hash(Path(__file__).with_name("primary_scene8_isic2016.py")),
        "presentation_scope": "Two builders, correctness tests, and full-label DINO + linear validation; extended results are supplementary.",
        "environment": matrix["environment"],
        "verified_conditions": len(matrix["results"]),
        "all_saved_metrics_recomputed": True,
        "feature_caches": {r["dataset"]: r["feature_cache"] for r in matrix["results"]},
        "manifest_hashes": sorted({r["manifest_sha256"] for r in matrix["results"]}),
        "artifacts": {p.name: file_hash(p) for p in args.output_dir.iterdir() if p.name != "evidence.json"},
    }
    write_json(args.output_dir / "evidence.json", evidence)
    print(json.dumps({"verified_conditions": len(matrix["results"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
