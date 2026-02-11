"""Collect benchmark results from W&B into summary tables.

Pulls linear probe top-1/top-5 and KNN top-1/top-5 accuracy for all
benchmark runs, then pivots into a {dataset x (model, backbone)} table.

Usage:
    python -m stable_datasets.benchmarks.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks

    # Save to CSV
    python -m stable_datasets.benchmarks.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks --output results.csv

    # Export pivot table as LaTeX
    python -m stable_datasets.benchmarks.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks --latex table.tex
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from stable_pretraining.utils.log_reader import WandbLogReader

log = logging.getLogger(__name__)

METRICS = [
    "eval/linear_probe_top1",
    "eval/linear_probe_top5",
    "eval/knn_probe_top1",
    "eval/knn_probe_top5",
]

CONFIG_COLS = ["model", "backbone", "dataset"]


def collect(entity: str, project: str) -> pd.DataFrame:
    """Pull summary metrics from all runs in a W&B project.

    Returns a DataFrame with columns for model, backbone, dataset,
    and all metric values from the run summary.
    """
    reader = WandbLogReader()
    summary_df = reader.read_project(entity, project, return_summary=True)

    if summary_df.empty:
        log.warning("No runs found.")
        return summary_df

    available_metrics = [m for m in METRICS if m in summary_df.columns]
    available_configs = [c for c in CONFIG_COLS if c in summary_df.columns]

    if not available_metrics:
        log.warning(
            f"No metric columns found. Available: {list(summary_df.columns)}"
        )
        return summary_df

    cols = available_configs + available_metrics + ["name", "id"]
    return summary_df[[c for c in cols if c in summary_df.columns]]


def pivot_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot results into {dataset x (model, backbone)} table.

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
    return df.pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="max"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark results from W&B"
    )
    parser.add_argument("--entity", required=True, help="W&B entity")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--latex", default=None, help="Output LaTeX table path (.tex)")
    parser.add_argument(
        "--metric",
        default="eval/linear_probe_top1",
        choices=METRICS,
        help="Metric to pivot on for the summary table",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = collect(args.entity, args.project)
    if df.empty:
        return

    print("\n=== All Results ===")
    print(df.to_string(index=False))

    table = None
    if all(c in df.columns for c in ("model", "backbone", "dataset")):
        table = pivot_table(df, args.metric)
        print(f"\n=== {args.metric} (dataset x method) ===")
        print(table.to_markdown())

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
