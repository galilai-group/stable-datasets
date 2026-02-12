"""Collect benchmark results from W&B into summary tables.

Pulls linear probe top-1/top-5 and KNN top-1/top-5 accuracy for all
benchmark runs, then pivots into a {dataset x (model, backbone)} table.
Shows the epoch at which peak performance was achieved.

Usage:
    python -m stable_datasets.benchmarks.self_supervised.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks

    # Save to CSV
    python -m stable_datasets.benchmarks.self_supervised.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks --output results.csv

    # Export pivot table as LaTeX
    python -m stable_datasets.benchmarks.self_supervised.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks --latex table.tex
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from stable_pretraining.utils.log_reader import WandbLogReader
from tqdm import tqdm

log = logging.getLogger(__name__)

METRICS = [
    "eval/linear_probe_top1_epoch",
    "eval/linear_probe_top5_epoch",
    "eval/knn_probe_top1",
    "eval/knn_probe_top5",
]

CONFIG_COLS = ["model", "backbone", "dataset"]


def collect(entity: str, project: str) -> pd.DataFrame:
    """Pull per-run best metrics from all runs in a W&B project.

    For each run and metric, finds the peak value across all epochs
    and records which epoch it occurred at.

    Returns a DataFrame with columns: model, backbone, dataset,
    each metric value (peak), each metric's best epoch, name, id.
    """
    reader = WandbLogReader()

    # Get summary for metadata (name, id, config fields)
    summary_df = reader.read_project(entity, project, return_summary=True)
    if summary_df.empty:
        log.warning("No runs found.")
        return summary_df

    # Download per-run history to find peak values and their epochs
    history_keys = METRICS + ["epoch"]
    rows = []
    for _, run_row in tqdm(
        summary_df.iterrows(), total=len(summary_df), desc="Downloading run histories"
    ):
        run_id = run_row["id"]
        row = {"id": run_id, "name": run_row.get("name", "")}
        for col in CONFIG_COLS:
            if col in run_row:
                row[col] = run_row[col]

        try:
            history_df, _ = reader.read(
                entity, project, run_id, keys=history_keys, _tqdm_disable=True
            )
        except Exception as e:
            log.warning(f"Failed to read history for run {run_id}: {e}")
            rows.append(row)
            continue

        for metric in METRICS:
            if metric not in history_df.columns:
                continue
            valid = history_df[metric].dropna()
            if valid.empty:
                continue
            best_idx = valid.idxmax()
            row[metric] = valid.loc[best_idx]
            # Get epoch number from the history row where peak occurred
            if "epoch" in history_df.columns:
                row[f"{metric}_best_epoch"] = int(history_df.loc[best_idx, "epoch"])
            else:
                row[f"{metric}_best_epoch"] = int(best_idx)

        rows.append(row)

    return pd.DataFrame(rows)


def pivot_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot results into {dataset x (model, backbone)} numeric table.

    When multiple runs exist for the same combo, keeps the one
    with the highest metric value.

    Args:
        df: DataFrame from collect().
        metric: Which metric column to use as values.
    """
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available: {list(df.columns)}"
        )
    df = df.copy()
    df["method"] = df["model"] + " / " + df["backbone"]
    # Deduplicate: keep the best run per (dataset, method) group
    df = df.sort_values(metric, ascending=False).drop_duplicates(
        subset=["dataset", "method"], keep="first"
    )
    return df.pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="max"
    )


def pivot_table_with_epochs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot results with epoch annotations: "0.7618 (ep. 95)".

    When multiple runs exist for the same combo, keeps the one
    with the highest metric value.

    Args:
        df: DataFrame from collect().
        metric: Which metric column to use as values.
    """
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available: {list(df.columns)}"
        )
    df = df.copy()
    df["method"] = df["model"] + " / " + df["backbone"]
    # Deduplicate: keep the best run per (dataset, method) group
    df = df.sort_values(metric, ascending=False).drop_duplicates(
        subset=["dataset", "method"], keep="first"
    )

    epoch_col = f"{metric}_best_epoch"
    has_epoch = epoch_col in df.columns

    if has_epoch:
        df["_display"] = df.apply(
            lambda r: f"{r[metric]:.4f} (ep. {int(r[epoch_col])})"
            if pd.notna(r.get(epoch_col)) and pd.notna(r[metric])
            else (f"{r[metric]:.4f}" if pd.notna(r[metric]) else ""),
            axis=1,
        )
        return df.pivot_table(
            index="dataset", columns="method", values="_display", aggfunc="first"
        )

    # Fallback: no epoch info, just show numeric values
    return df.pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="max"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark results from W&B"
    )
    parser.add_argument("--entity", default="stable-ssl", help="W&B entity")
    parser.add_argument("--project", default="stable-datasets-benchmarks", help="W&B project")
    parser.add_argument("--output", default='ssl_baselines_results.csv', help="Output CSV path")
    parser.add_argument("--latex", default='ssl_baselines_results_table.tex', help="Output LaTeX table path (.tex)")
    parser.add_argument(
        "--metric",
        default=None,
        choices=METRICS,
        help="Metric to pivot on for the summary table (default: first available)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = collect(args.entity, args.project)
    if df.empty:
        return

    print("\n=== All Results ===")
    print(df.to_string(index=False))

    # Pick the requested metric, or fall back to the first available one
    metric = args.metric
    if metric is None or metric not in df.columns:
        available = [m for m in METRICS if m in df.columns]
        if not available:
            log.warning("No known metric columns found for pivot table.")
            metric = None
        else:
            metric = available[0]
            if args.metric is not None:
                log.warning(f"Metric '{args.metric}' not in results, falling back to '{metric}'")

    table = None
    display_table = None
    if metric and all(c in df.columns for c in ("model", "backbone", "dataset")):
        table = pivot_table(df, metric)
        display_table = pivot_table_with_epochs(df, metric)
        print(f"\n=== {metric} (dataset x method) ===")
        print(display_table.to_markdown())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    if args.latex:
        if table is None:
            log.warning("Cannot produce LaTeX: pivot table requires model, backbone, and dataset columns.")
        else:
            latex_str = table.to_latex(float_format="%.2f", bold_rows=True)
            with open(args.latex, "w") as f:
                f.write(latex_str)
            print(f"LaTeX table saved to {args.latex}")


if __name__ == "__main__":
    main()
