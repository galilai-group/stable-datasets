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
import time
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

# Seconds to sleep between consecutive W&B API attribute accesses to avoid 429s.
_RATE_LIMIT_SECONDS = 1

log = logging.getLogger(__name__)

# Datasets excluded from result tables (e.g. no classification labels).
SKIP_DATASETS = {"facepointing", "kmnist"}

SSL_METRIC = "eval/linear_probe_top1_epoch"

# Per-model LR requirements: only accept runs with these LRs.
# This filters out runs from before LR/scheduler fixes.
_REQUIRED_LR: dict[str, float] = {
    "lejepa": 5e-4,
    "dino": 5e-4,
}

CACHE_DIR = Path(__file__).resolve().parent / ".result_cache"


def _collect_ssl(entity: str, project: str, refresh: bool = False, require_tag: str | None = None) -> pd.DataFrame:
    """Collect SSL linear-probe results from W&B summary (no history download needed).

    Looks for untagged runs with vit_small backbone that have
    ``eval/linear_probe_top1_epoch`` in their summary.  Accepts any run state
    (finished *or* failed) as long as the metric is present — some runs that
    W&B marks "failed" still produced valid linear-probe results.

    Caches runs that have the metric so subsequent calls skip W&B API for them.
    Use ``--refresh`` to clear the cache and re-download everything.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"ssl_{hashlib.md5(f'{entity}/{project}'.encode()).hexdigest()}.json"

    cached_runs: dict[str, dict] = {}
    if refresh:
        # Wipe the cache so every run is re-checked
        if cache_file.exists():
            cache_file.unlink()
            log.info("Cleared SSL cache (--refresh)")
    elif cache_file.exists():
        try:
            cached_runs = json.loads(cache_file.read_text())
            # Evict cached runs that now fail the LR filter
            evicted = [
                rid for rid, row in cached_runs.items()
                if row.get("model") in _REQUIRED_LR
                and abs(float(row.get("lr", 0)) - _REQUIRED_LR[row["model"]]) > 1e-8
            ]
            for rid in evicted:
                del cached_runs[rid]
            if evicted:
                log.info(f"Evicted {len(evicted)} cached runs failing LR filter")
                cache_file.write_text(json.dumps(cached_runs, indent=2))
            log.info(f"Loaded {len(cached_runs)} cached SSL runs")
        except Exception:
            cached_runs = {}

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=1000))
    log.info(f"Found {len(runs)} total runs in {entity}/{project}")

    rows = []
    new_cached = 0
    for i, run in enumerate(tqdm(runs, desc="Scanning SSL runs")):
        run_id = run.id

        # Use cached result if available
        if run_id in cached_runs:
            rows.append(cached_runs[run_id])
            continue

        # Rate-limit W&B API access (accessing .tags, .config, .summary hits API)
        if i > 0:
            time.sleep(_RATE_LIMIT_SECONDS)

        # Only include runs whose tags are a subset of known SSL tags
        tags = set(run.tags or [])
        _KNOWN_TAGS = {"seed", "final_runs"}
        is_seed_run = "seed" in tags
        if tags - _KNOWN_TAGS:
            log.debug(f"Skipping {run_id} (unknown tags: {tags - _KNOWN_TAGS})")
            continue

        # If --tag is specified, only include runs with that tag
        if require_tag and require_tag not in tags:
            continue

        # Skip still-running runs
        if run.state == "running":
            log.debug(f"Skipping {run_id} (still running)")
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

        # Filter by required LR (rejects old runs with wrong LR/scheduler)
        if model in _REQUIRED_LR:
            run_lr = run.config.get("lr")
            required_lr = _REQUIRED_LR[model]
            if run_lr is None or abs(float(run_lr) - required_lr) > 1e-8:
                log.debug(
                    f"Skipping {run_id} ({model}/{dataset}): "
                    f"lr={run_lr}, required={required_lr}"
                )
                continue

        top1 = run.summary.get(SSL_METRIC)
        if top1 is None:
            log.debug(f"Skipping {run_id} ({model}/{dataset}): no {SSL_METRIC} in summary")
            continue

        row = {
            "dataset": dataset.lower(),
            "model": model,
            "backbone": backbone,
            "lr": run.config.get("lr"),
            "seed": run.config.get("seed"),
            "is_seed_run": is_seed_run,
            SSL_METRIC: float(top1),
            "id": run_id,
            "name": run.name,
            "state": run.state,
        }
        rows.append(row)
        cached_runs[run_id] = row
        new_cached += 1
        log.info(f"  Found {model}/{dataset} = {float(top1):.4f} (state={run.state})")

    if new_cached > 0:
        cache_file.write_text(json.dumps(cached_runs, indent=2))
        log.info(f"Cached {new_cached} new SSL runs")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info(f"Collected {len(df)} SSL runs (vit_small)")
    return df


def _collect_supervised(entity: str, project: str, refresh: bool = False) -> pd.DataFrame:
    """Collect supervised baseline results from a W&B project.

    Supervised runs store ``test_accuracy`` in their W&B summary.
    Caches runs to avoid re-downloading.

    Returns a DataFrame with columns: dataset, test_accuracy.
    Keeps the best test_accuracy per dataset when duplicates exist.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"supervised_{hashlib.md5(f'{entity}/{project}'.encode()).hexdigest()}.json"

    cached_runs: dict[str, dict] = {}
    if refresh:
        if cache_file.exists():
            cache_file.unlink()
            log.info("Cleared supervised cache (--refresh)")
    elif cache_file.exists():
        try:
            cached_runs = json.loads(cache_file.read_text())
            log.info(f"Loaded {len(cached_runs)} cached supervised runs")
        except Exception:
            cached_runs = {}

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=200))
    log.info(f"Found {len(runs)} supervised runs in {entity}/{project}")

    rows = []
    new_cached = 0
    for i, run in enumerate(tqdm(runs, desc="Scanning supervised runs")):
        run_id = run.id

        if run_id in cached_runs:
            rows.append(cached_runs[run_id])
            continue

        if i > 0:
            time.sleep(_RATE_LIMIT_SECONDS)

        if run.state == "running":
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
        new_cached += 1

    if new_cached > 0:
        cache_file.write_text(json.dumps(cached_runs, indent=2))
        log.info(f"Cached {new_cached} new supervised runs")

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
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Pivot SSL results into {dataset x method} numeric table.

    Returns (mean_table, std_table). std_table is None if no seed runs exist.

    Primary (non-seed) runs: deduplicates by keeping the best run per
    (dataset, method) group. Seed runs are aggregated separately to compute
    mean +/- std across seeds for each (dataset, method).
    """
    metric = SSL_METRIC
    df = ssl_df.copy()
    df["method"] = df["model"]

    # Split into primary and seed runs
    is_seed = df["is_seed_run"].fillna(False).astype(bool)
    primary_df = df[~is_seed]
    seed_df = df[is_seed]

    # Primary table: best run per (dataset, method)
    primary_df = primary_df.sort_values(metric, ascending=False).drop_duplicates(
        subset=["dataset", "method"], keep="first"
    )
    numeric = primary_df.pivot_table(
        index="dataset", columns="method", values=metric, aggfunc="max"
    )

    # Seed table: mean and std across seeds per (dataset, method)
    std_table = None
    if not seed_df.empty:
        seed_mean = seed_df.pivot_table(
            index="dataset", columns="method", values=metric, aggfunc="mean"
        )
        seed_std = seed_df.pivot_table(
            index="dataset", columns="method", values=metric, aggfunc="std"
        )
        # Overlay: where we have seed data, use seed mean instead of primary
        for col in seed_mean.columns:
            if col in numeric.columns:
                mask = seed_mean[col].notna()
                numeric.loc[mask, col] = seed_mean.loc[mask, col]
            else:
                numeric[col] = seed_mean[col]
        std_table = seed_std

    if supervised_df is not None and not supervised_df.empty:
        sup_series = supervised_df.set_index("dataset")["test_accuracy"]
        numeric["Supervised"] = sup_series

    # Row-wise average
    numeric["Average"] = numeric.mean(axis=1)
    # Col-wise average
    avg_row = numeric.drop(columns="Average").mean(axis=0)
    avg_row["Average"] = numeric["Average"].mean()
    numeric.loc["Average"] = avg_row

    if std_table is not None:
        std_table["Average"] = std_table.mean(axis=1)
        std_avg_row = std_table.mean(axis=0)
        std_table.loc["Average"] = std_avg_row

    return numeric, std_table


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
    "imagenette": "Imagenette",
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


def _format_latex(table: pd.DataFrame, std_table: pd.DataFrame | None = None) -> str:
    """Format a pivot table as a booktabs LaTeX table.

    Columns are SSL method names, optionally "Supervised", plus "Average".
    When std_table is provided, cells with seed data render as ``mean {\\pm} std``.
    """
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None

    ssl_cols = [c for c in pct.columns if c not in ("Average", "Supervised")]
    has_supervised = "Supervised" in pct.columns

    model_labels = [_display_name(m, _MODEL_DISPLAY_NAMES) for m in ssl_cols]
    n_ssl = len(ssl_cols)
    n_extra = 1 if has_supervised else 0
    n_total_data = n_ssl + n_extra + 1  # +1 for Avg.

    datasets = [idx for idx in pct.index if idx != "Average"]

    def _fmt(val, std_val=None):
        if pd.isna(val):
            return "---"
        if std_val is not None and not pd.isna(std_val):
            return f"{val:.1f} {{\\scriptsize $\\pm$ {std_val:.1f}}}"
        return f"{val:.1f}"

    def _get_std(ds, col):
        if pct_std is None:
            return None
        if ds not in pct_std.index or col not in pct_std.columns:
            return None
        return pct_std.loc[ds, col]

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
            row += f" & {_fmt(pct.loc[ds, col], _get_std(ds, col))}"
        row += f" & {_fmt(pct.loc[ds, 'Average'], _get_std(ds, 'Average'))}"
        row += " \\\\"
        lines.append(row)

    # Average row
    lines.append("\\midrule")
    avg_row = "\\textbf{Average}"
    for col in all_data_cols:
        avg_row += f" & {_fmt(pct.loc['Average', col], _get_std('Average', col))}"
    avg_row += f" & {_fmt(pct.loc['Average', 'Average'], _get_std('Average', 'Average'))}"
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
    parser.add_argument(
        "--tag",
        default=None,
        help="Only include runs with this wandb tag (e.g. final_runs)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Collect SSL results
    ssl_df = _collect_ssl(args.entity, args.project, refresh=args.refresh, require_tag=args.tag)
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
    table, std_table = pivot_table(ssl_df, supervised_df=supervised_df)
    print(f"\n=== Linear Probe Top-1 (dataset x method) ===")
    # Display as percentages
    print((table * 100).round(1).to_markdown())

    if std_table is not None:
        print(f"\n=== Seed Std Dev (dataset x method) ===")
        print((std_table * 100).round(1).to_markdown())

    if args.output:
        ssl_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    if args.latex:
        latex_str = _format_latex(table, std_table=std_table)
        with open(args.latex, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {args.latex}")


if __name__ == "__main__":
    main()
