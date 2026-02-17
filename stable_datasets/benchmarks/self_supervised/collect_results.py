"""Collect offline probe results from W&B into summary tables.

Pulls offline linear-probe top-1/top-5 accuracy for runs tagged
``offline_probe``, then pivots into a {dataset x (model, backbone)} table.
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

OFFLINE_PROBE_TAG = "offline_probe"

METRICS = [
    "eval/accuracy_top1_epoch",
    "eval/accuracy_top5_epoch",
]

# Offline probe runs store the SSL model name as "ssl_model" in the W&B config.
# We remap it to "model" so the pivot tables stay consistent.
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
        if "created_at" in run_row:
            row["created_at"] = run_row["created_at"]
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
            # .loc can return a Series if the index has duplicates, so
            # extract a scalar with .iloc[0] when that happens.
            if "epoch" in history_df.columns:
                epoch_value = history_df.loc[best_idx, "epoch"]
                # Handle case where duplicate indices return a Series
                if isinstance(epoch_value, pd.Series):
                    epoch_value = epoch_value.iloc[0]
                row[f"{metric}_best_epoch"] = int(epoch_value)
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
    parser.add_argument(
        "--after",
        default=HPARAMS_V2_CUTOFF,
        help="Only include runs created after this ISO-8601 timestamp "
             "(default: hparams-v2 cutoff %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = collect(args.entity, args.project)
    if df.empty:
        return

    # Filter runs by creation time (only keep runs after the hparams change)
    if "created_at" in df.columns and args.after:
        cutoff = datetime.fromisoformat(args.after)
        created = pd.to_datetime(df["created_at"], utc=True)
        before_mask = created < cutoff
        if before_mask.any():
            log.info(
                f"Filtering out {before_mask.sum()} run(s) created before {args.after}."
            )
            df = df[~before_mask].reset_index(drop=True)

    # Also filter out any ResNet runs as a safety net
    if "backbone" in df.columns:
        resnet_mask = df["backbone"].str.contains("resnet", case=False, na=False)
        if resnet_mask.any():
            log.info(f"Filtering out {resnet_mask.sum()} ResNet run(s).")
            df = df[~resnet_mask].reset_index(drop=True)

    # Drop runs from datasets without num_classes (no eval metrics logged)
    metric_cols = [m for m in METRICS if m in df.columns]
    if metric_cols:
        df = df.dropna(subset=metric_cols, how="all").reset_index(drop=True)
    if df.empty:
        log.warning("No runs with eval metrics found (datasets may lack num_classes).")
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
