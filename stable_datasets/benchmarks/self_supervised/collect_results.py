"""Collect SSL and supervised results from W&B into summary tables.

Pulls linear-probe top-1 accuracy from SSL runs and optionally supervised
baseline results, then pivots into a {dataset x method} table.

Usage:
    python -m stable_datasets.benchmarks.self_supervised.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks

    # With supervised baselines and LaTeX export
    python -m stable_datasets.benchmarks.self_supervised.collect_results \\
        --entity stable-ssl --project stable-datasets-benchmarks \\
        --supervised-entity samibg --supervised-project stable-datasets \\
        --latex table.tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import wandb

log = logging.getLogger(__name__)

# Datasets excluded from result tables (e.g. no classification labels).
SKIP_DATASETS = {"facepointing"}

SSL_METRIC = "eval/linear_probe_top1_epoch"

CACHE_DIR = Path(__file__).resolve().parent / ".result_cache"


def _collect_ssl(entity: str, project: str, refresh: bool = False) -> pd.DataFrame:
    """Collect SSL linear-probe results from W&B summary (no history download needed).

    Looks for untagged, finished runs with vit_small backbone that have
    ``eval/linear_probe_top1_epoch`` in their summary.

    Caches finished runs so subsequent calls skip W&B API for them.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"ssl_{hashlib.md5(f'{entity}/{project}'.encode()).hexdigest()}.json"

    cached_runs = {}
    if not refresh and cache_file.exists():
        try:
            cached_runs = json.loads(cache_file.read_text())
            log.info(f"Loaded {len(cached_runs)} cached SSL runs")
        except Exception:
            cached_runs = {}

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=1000))
    log.info(f"Found {len(runs)} total runs in {entity}/{project}")

    rows = []
    new_finished = 0
    for run in runs:
        run_id = run.id

        # Use cached result for previously finished runs
        if run_id in cached_runs:
            rows.append(cached_runs[run_id])
            continue

        # Only include untagged runs (SSL training runs, not offline probes)
        if run.tags:
            continue

        # Only finished runs
        if run.state != "finished":
            continue

        backbone = run.config.get("backbone", "")
        if backbone != "vit_small":
            continue

        dataset = run.config.get("dataset", "")
        model = run.config.get("model", "")
        if not dataset or not model:
            continue

        if dataset.lower() in SKIP_DATASETS:
            continue

        top1 = run.summary.get(SSL_METRIC)
        if top1 is None:
            continue

        row = {
            "dataset": dataset.lower(),
            "model": model,
            "backbone": backbone,
            SSL_METRIC: float(top1),
            "id": run_id,
            "name": run.name,
        }
        rows.append(row)
        cached_runs[run_id] = row
        new_finished += 1

    if new_finished > 0:
        cache_file.write_text(json.dumps(cached_runs, indent=2))
        log.info(f"Cached {new_finished} new finished SSL runs")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info(f"Collected {len(df)} SSL runs (vit_small, finished)")
    return df


def _collect_supervised(entity: str, project: str, refresh: bool = False) -> pd.DataFrame:
    """Collect supervised baseline results from a W&B project.

    Supervised runs store ``test_accuracy`` in their W&B summary.
    Caches finished runs to avoid re-downloading.

    Returns a DataFrame with columns: dataset, test_accuracy.
    Keeps the best test_accuracy per dataset when duplicates exist.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"supervised_{hashlib.md5(f'{entity}/{project}'.encode()).hexdigest()}.json"

    cached_runs = {}
    if not refresh and cache_file.exists():
        try:
            cached_runs = json.loads(cache_file.read_text())
            log.info(f"Loaded {len(cached_runs)} cached supervised runs")
        except Exception:
            cached_runs = {}

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=200))
    log.info(f"Found {len(runs)} supervised runs in {entity}/{project}")

    rows = []
    new_finished = 0
    for run in runs:
        run_id = run.id

        if run_id in cached_runs:
            rows.append(cached_runs[run_id])
            continue

        if run.state != "finished":
            continue

        dataset = run.config.get("dataset", "")
        if not dataset:
            continue
        dataset_key = dataset.lower()

        if dataset_key in SKIP_DATASETS:
            continue

        test_acc = run.summary.get("test_accuracy")
        if test_acc is None:
            continue

        row = {
            "dataset": dataset_key,
            "test_accuracy": float(test_acc),
            "id": run_id,
            "name": run.name,
        }
        rows.append(row)
        cached_runs[run_id] = row
        new_finished += 1

    if new_finished > 0:
        cache_file.write_text(json.dumps(cached_runs, indent=2))
        log.info(f"Cached {new_finished} new finished supervised runs")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep best test_accuracy per dataset
    df = df.sort_values("test_accuracy", ascending=False).drop_duplicates(
        subset=["dataset"], keep="first"
    ).reset_index(drop=True)
    return df


def pivot_table(
    ssl_df: pd.DataFrame,
    supervised_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pivot SSL results into {dataset x method} numeric table.

    Deduplicates by keeping the best run per (dataset, method) group.
    Optionally adds a "Supervised" column.
    """
    metric = SSL_METRIC
    df = ssl_df.copy()
    df["method"] = df["model"]
    # Deduplicate: keep the best run per (dataset, method) group
    df = df.sort_values(metric, ascending=False).drop_duplicates(
        subset=["dataset", "method"], keep="first"
    )
    numeric = df.pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="max"
    )

    if supervised_df is not None and not supervised_df.empty:
        sup_series = supervised_df.set_index("dataset")["test_accuracy"]
        numeric["Supervised"] = sup_series

    # Row-wise average
    numeric["Average"] = numeric.mean(axis=1)
    # Col-wise average
    avg_row = numeric.drop(columns="Average").mean(axis=0)
    avg_row["Average"] = numeric["Average"].mean()
    numeric.loc["Average"] = avg_row
    return numeric


_DATASET_DISPLAY_NAMES: dict[str, str] = {
    "arabiccharacters": "Arabic Characters",
    "arabicdigits": "Arabic Digits",
    "awa2": "AWA2",
    "beans": "Beans",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "country211": "Country-211",
    "cub200": "CUB-200",
    "dtd": "DTD",
    "emnist": "EMNIST",
    "facepointing": "Face Pointing",
    "fashionmnist": "FashionMNIST",
    "fgvcaircraft": "FGVC Aircraft",
    "flowers102": "Flowers-102",
    "food101": "Food-101",
    "galaxy10decal": "Galaxy10 DECal",
    "hasyv2": "HASYv2",
    "imagenet": "ImageNet",
    "kmnist": "KMNIST",
    "linnaeus5": "Linnaeus 5",
    "medmnist": "MedMNIST",
    "notmnist": "NotMNIST",
    "rockpaperscissor": "Rock-Paper-Scissors",
    "stl10": "STL-10",
    "svhn": "SVHN",
    "tinyimagenet": "Tiny ImageNet",
}

_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "simclr": "SimCLR",
    "dino": "DINO",
    "mae": "MAE",
    "lejepa": "LeJEPA",
    "nnclr": "NNCLR",
    "barlow_twins": "Barlow Twins",
}


def _display_name(raw: str, mapping: dict[str, str]) -> str:
    """Return a human-readable display name, falling back to *raw*."""
    return mapping.get(raw, raw)


def _format_latex(table: pd.DataFrame) -> str:
    """Format a pivot table as a booktabs LaTeX table.

    Columns are SSL method names, optionally "Supervised", plus "Average".
    """
    pct = table * 100

    ssl_cols = [c for c in pct.columns if c not in ("Average", "Supervised")]
    has_supervised = "Supervised" in pct.columns

    model_labels = [_display_name(m, _MODEL_DISPLAY_NAMES) for m in ssl_cols]
    n_ssl = len(ssl_cols)
    n_extra = 1 if has_supervised else 0
    n_total_data = n_ssl + n_extra + 1  # +1 for Avg.

    datasets = [idx for idx in pct.index if idx != "Average"]

    def _fmt(val):
        if pd.isna(val):
            return "---"
        return f"{val:.1f}"

    lines = []
    lines.append("\\begin{tabular}{l " + " ".join(["c"] * n_total_data) + "}")
    lines.append("\\toprule")

    # Multicolumn header for SSL methods
    if has_supervised:
        lines.append(
            f" & \\multicolumn{{{n_ssl}}}{{c}}"
            f"{{\\textbf{{Method (ViT-Small)}}}} & & \\\\"
        )
    else:
        lines.append(
            f" & \\multicolumn{{{n_ssl}}}{{c}}"
            f"{{\\textbf{{Method (ViT-Small)}}}} & \\\\"
        )
    lines.append(f"\\cmidrule(lr){{2-{n_ssl + 1}}}")

    # Header row
    header = "\\textbf{Dataset}"
    for m in model_labels:
        header += f" & {m}"
    if has_supervised:
        header += " & Supervised"
    header += " & \\textbf{Avg.} \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    all_data_cols = ssl_cols + (["Supervised"] if has_supervised else [])
    for ds in datasets:
        ds_label = _display_name(ds, _DATASET_DISPLAY_NAMES)
        row = ds_label
        for col in all_data_cols:
            row += f" & {_fmt(pct.loc[ds, col])}"
        row += f" & {_fmt(pct.loc[ds, 'Average'])}"
        row += " \\\\"
        lines.append(row)

    # Average row
    lines.append("\\midrule")
    avg_row = "\\textbf{Average}"
    for col in all_data_cols:
        avg_row += f" & {_fmt(pct.loc['Average', col])}"
    avg_row += f" & {_fmt(pct.loc['Average', 'Average'])}"
    avg_row += " \\\\"
    lines.append(avg_row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Collect SSL and supervised results from W&B"
    )
    parser.add_argument("--entity", default="stable-ssl", help="W&B entity for SSL runs")
    parser.add_argument("--project", default="stable-datasets-benchmarks", help="W&B project for SSL runs")
    parser.add_argument("--output", default="offline_probe_results.csv", help="Output CSV path")
    parser.add_argument("--latex", default=None, help="Output LaTeX table path (.tex)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass caches and re-download all data from W&B",
    )
    parser.add_argument(
        "--supervised-entity",
        default=None,
        help="W&B entity for supervised baselines (e.g. samibg)",
    )
    parser.add_argument(
        "--supervised-project",
        default=None,
        help="W&B project for supervised baselines (e.g. stable-datasets)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Collect SSL results
    ssl_df = _collect_ssl(args.entity, args.project, refresh=args.refresh)
    if ssl_df.empty:
        log.warning("No SSL runs found.")
        return

    # Drop runs without the metric
    ssl_df = ssl_df.dropna(subset=[SSL_METRIC]).reset_index(drop=True)
    if ssl_df.empty:
        log.warning("No SSL runs with linear probe metrics found.")
        return

    print(f"\n=== SSL Results ({len(ssl_df)} runs) ===")
    display_cols = [c for c in ["model", "backbone", "dataset", SSL_METRIC] if c in ssl_df.columns]
    print(ssl_df[display_cols].to_string(index=False))

    # Collect supervised baselines if requested
    supervised_df = None
    if args.supervised_entity and args.supervised_project:
        supervised_df = _collect_supervised(
            args.supervised_entity, args.supervised_project, refresh=args.refresh
        )
        if supervised_df is not None and not supervised_df.empty:
            print(f"\n=== Supervised Baselines ({len(supervised_df)} datasets) ===")
            print(supervised_df[["dataset", "test_accuracy"]].to_string(index=False))

    # Build pivot table
    table = pivot_table(ssl_df, supervised_df=supervised_df)
    print(f"\n=== Linear Probe Top-1 (dataset x method) ===")
    # Display as percentages
    print((table * 100).round(1).to_markdown())

    if args.output:
        ssl_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    if args.latex:
        latex_str = _format_latex(table)
        with open(args.latex, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {args.latex}")


if __name__ == "__main__":
    main()
