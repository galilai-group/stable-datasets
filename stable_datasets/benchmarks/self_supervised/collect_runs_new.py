"""Collect SSL and supervised results from W&B into summary tables.

Optimized version with retry-on-429 instead of fixed sleeps, server-side
filters, and negative caching for skipped runs.

Usage:
    python -m stable_datasets.benchmarks.self_supervised.collect_runs_new \
        --entity stable-ssl --project stable-datasets-benchmarks

    # With supervised baselines
    python -m stable_datasets.benchmarks.self_supervised.collect_runs_new \
        --entity stable-ssl --project stable-datasets-benchmarks \
        --supervised-entity samibg --supervised-project stable-datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import wandb
from tqdm import tqdm

log = logging.getLogger(__name__)

# Datasets excluded from result tables (e.g. no classification labels).
SKIP_DATASETS = {"facepointing", "kmnist"}

SSL_METRIC = "eval/linear_probe_top1_epoch"

# Per-model run requirements: only accept runs matching these configs.
# This filters out runs from before fixes (wrong LR, grad accum, batch size).
_REQUIRED_LR: dict[str, float] = {
    "lejepa": 5e-4,
    "dino": 5e-4,
}
_REQUIRED_EBS: dict[str, int] = {
    "lejepa": 256,
    "dino": 256,
}

# ---------------------------------------------------------------------------
# Required max_epochs per (model, dataset).
#
# Every model has a base epoch count (its ImageNet value) and each dataset
# belongs to a size/difficulty tier with an integer multiplier.  The tier
# assignment is derived from dataset size — smaller or harder datasets need
# proportionally more epochs to see enough samples.
#
#   Base epochs:
#       barlow_twins / simclr / nnclr  =  50
#       dino / lejepa                  =  62.5  (→ ceil when non-integer)
#       mae                            = 100
#
#   Tier multipliers:
#       1x  imagenet                          (largest)
#       2x  food101, tinyimagenet             (large)
#       3x  svhn, fashionmnist, emnist,       (medium, simple classes)
#           kmnist, hasyv2, stl10
#       4x  cifar10, cifar100, notmnist,      (medium, more classes or
#           medmnist, country211               harder distribution)
#       5x  arabiccharacters, arabicdigits    (small)
#       6x  flowers102, cub200, dtd,          (small, fine-grained)
#           rockpaperscissor
#
# For dino/lejepa the base is 62.5, so non-integer products are rounded up:
#   62.5 × 1 → 63,  ×3 → 188,  ×5 → 313
# ---------------------------------------------------------------------------

_BASE_EPOCHS: dict[str, float] = {
    "barlow_twins": 50,
    "simclr": 50,
    "nnclr": 50,
    "dino": 62.5,
    "lejepa": 62.5,
    "mae": 100,
}

_DATASET_TIER: dict[str, int] = {
    # Tier 1 — largest dataset
    "imagenet": 1,
    # Tier 2 — large
    "food101": 2,
    "tinyimagenet": 2,
    "imagenette": 2,
    # Tier 3 — medium, simple classes
    "svhn": 3,
    "fashionmnist": 3,
    "emnist": 3,
    "kmnist": 3,
    "hasyv2": 3,
    "stl10": 3,
    # Tier 4 — medium, harder
    "cifar10": 4,
    "cifar100": 4,
    "notmnist": 4,
    "medmnist": 4,
    "country211": 4,
    # Tier 5 — small
    "arabiccharacters": 5,
    "arabicdigits": 5,
    # Tier 6 — small, fine-grained
    "flowers102": 6,
    "cub200": 6,
    "dtd": 6,
    "rockpaperscissor": 6,
}

_REQUIRED_EPOCHS: dict[str, dict[str, int]] = {}
for _model, _base in _BASE_EPOCHS.items():
    _REQUIRED_EPOCHS[_model] = {
        ds: math.ceil(_base * mult) for ds, mult in _DATASET_TIER.items()
    }

CACHE_DIR = Path(__file__).resolve().parent / ".result_cache"

# Run states that are final and safe to cache.
_TERMINAL_STATES = {"finished", "failed", "crashed"}


def _retry_on_429(fn, max_retries=5):
    """Call fn(), retrying with exponential backoff on HTTP 429."""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"Rate limited (429), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    return fn()


def _load_cache(cache_file: Path, refresh: bool) -> dict[str, dict]:
    if refresh:
        if cache_file.exists():
            cache_file.unlink()
            log.info("Cleared cache (--refresh)")
        return {}
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            log.info(f"Loaded {len(data)} cached entries from {cache_file.name}")
            return data
        except Exception:
            return {}
    return {}


def _save_cache(cache_file: Path, data: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, indent=2))


def _collect_ssl(
    entity: str,
    project: str,
    refresh: bool = False,
    require_tag: str | None = None,
) -> pd.DataFrame:
    """Collect SSL linear-probe results from W&B.

    Uses server-side filters to only fetch vit_small runs.
    Caches both accepted and rejected (negative cache) terminal runs.
    Uses retry-on-429 instead of fixed sleeps.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    proj_hash = hashlib.md5(f"{entity}/{project}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"ssl_v2_{proj_hash}.json"

    cached_runs = _load_cache(cache_file, refresh)

    # Evict cached runs that now fail the LR filter
    evicted = [
        rid
        for rid, row in cached_runs.items()
        if not row.get("_skip")
        and row.get("model") in _REQUIRED_LR
        and abs(float(row.get("lr", 0)) - _REQUIRED_LR[row["model"]]) > 1e-8
    ]
    for rid in evicted:
        del cached_runs[rid]
    if evicted:
        log.info(f"Evicted {len(evicted)} cached runs failing LR filter")
        _save_cache(cache_file, cached_runs)

    # Evict cached runs that now fail the epoch filter
    evicted_ep = [
        rid
        for rid, row in cached_runs.items()
        if not row.get("_skip")
        and row.get("model") in _REQUIRED_EPOCHS
        and row.get("dataset", "").lower() in _REQUIRED_EPOCHS.get(row.get("model", ""), {})
        and int(row.get("max_epochs", 0)) != _REQUIRED_EPOCHS[row["model"]][row["dataset"].lower()]
    ]
    for rid in evicted_ep:
        del cached_runs[rid]
    if evicted_ep:
        log.info(f"Evicted {len(evicted_ep)} cached runs failing epoch filter")
        _save_cache(cache_file, cached_runs)

    # Server-side filter: only fetch vit_small backbone runs
    api = wandb.Api(timeout=60)
    filters = {"config.backbone": "vit_small"}
    runs = _retry_on_429(
        lambda: list(api.runs(f"{entity}/{project}", filters=filters, per_page=1000))
    )
    log.info(f"Found {len(runs)} vit_small runs in {entity}/{project}")

    _KNOWN_TAGS = {"seed", "final_runs", "rescaled"}
    rows = []
    new_cached = 0
    dirty = False

    for run in tqdm(runs, desc="Scanning SSL runs"):
        run_id = run.id

        # Use cached result if available
        if run_id in cached_runs:
            entry = cached_runs[run_id]
            if not entry.get("_skip"):
                rows.append(entry)
            continue

        # Access run attributes with retry protection
        state = _retry_on_429(lambda: run.state)

        # Skip in-progress runs — don't cache them since state may change
        if state not in _TERMINAL_STATES:
            log.debug(f"Skipping {run_id} (state={state}, not terminal)")
            continue

        tags = set(_retry_on_429(lambda: run.tags or []))
        config = _retry_on_429(lambda: run.config)
        summary = _retry_on_429(lambda: run.summary)

        # Only include runs whose tags are a subset of known SSL tags
        is_seed_run = "seed" in tags
        if tags - _KNOWN_TAGS:
            log.debug(f"Skipping {run_id} (unknown tags: {tags - _KNOWN_TAGS})")
            cached_runs[run_id] = {"_skip": True, "reason": "unknown_tags"}
            dirty = True
            continue

        dataset = config.get("dataset", "")
        model = config.get("model", "")
        if not dataset or not model:
            cached_runs[run_id] = {"_skip": True, "reason": "missing_dataset_or_model"}
            dirty = True
            continue


        if dataset.lower() in SKIP_DATASETS:
            cached_runs[run_id] = {"_skip": True, "reason": "skip_dataset"}
            dirty = True
            continue

        # Filter by required LR, batch size, and grad accum
        def _check_required(config_key, required_dict, reason, is_int=False):
            if model not in required_dict:
                return True
            val = config.get(config_key)
            req = required_dict[model]
            if is_int:
                if val is None or int(val) != req:
                    log.debug(f"Skipping {run_id} ({model}/{dataset}): {config_key}={val}, required={req}")
                    cached_runs[run_id] = {"_skip": True, "reason": reason}
                    return False
            else:
                if val is None or abs(float(val) - req) > 1e-8:
                    log.debug(f"Skipping {run_id} ({model}/{dataset}): {config_key}={val}, required={req}")
                    cached_runs[run_id] = {"_skip": True, "reason": reason}
                    return False
            return True

        skip = False
        for cfg_key, req_dict, reason, is_int in [
            ("lr", _REQUIRED_LR, "wrong_lr", False),
        ]:
            if not _check_required(cfg_key, req_dict, reason, is_int):
                dirty = True
                skip = True
                break
        if skip:
            continue

        # Check effective batch size (batch_size * accumulate_grad_batches).
        # wandb logs the micro-batch size (already divided by accum), so we
        # reconstruct the effective batch size for comparison.
        if model in _REQUIRED_EBS:
            bs = config.get("batch_size")
            accum = config.get("accumulate_grad_batches", 1)
            if bs is not None and accum is not None:
                ebs = int(bs) * int(accum)
            else:
                ebs = None
            req_ebs = _REQUIRED_EBS[model]
            if ebs is None or ebs != req_ebs:
                log.debug(
                    f"Skipping {run_id} ({model}/{dataset}): "
                    f"effective_batch_size={ebs} (bs={bs}, accum={accum}), required={req_ebs}"
                )
                cached_runs[run_id] = {"_skip": True, "reason": "wrong_ebs"}
                dirty = True
                skip = True
        if skip:
            continue

        # Filter by required max_epochs (model + dataset specific)
        model_epochs = _REQUIRED_EPOCHS.get(model, {})
        req_epochs = model_epochs.get(dataset.lower())
        if req_epochs is not None:
            run_epochs = config.get("max_epochs")
            if run_epochs is None or int(run_epochs) != req_epochs:
                log.debug(
                    f"Skipping {run_id} ({model}/{dataset}): "
                    f"max_epochs={run_epochs}, required={req_epochs}"
                )
                cached_runs[run_id] = {"_skip": True, "reason": "wrong_epochs"}
                dirty = True
                continue

        top1 = summary.get(SSL_METRIC)
        if top1 is None:
            log.debug(f"Skipping {run_id} ({model}/{dataset}): no {SSL_METRIC}")
            cached_runs[run_id] = {"_skip": True, "reason": "no_metric"}
            dirty = True
            continue

        row = {
            "dataset": dataset.lower(),
            "model": model,
            "backbone": "vit_small",
            "lr": config.get("lr"),
            "max_epochs": config.get("max_epochs"),
            "seed": config.get("seed"),
            "is_seed_run": is_seed_run,
            SSL_METRIC: float(top1),
            "id": run_id,
            "name": run.name,
            "state": state,
        }
        rows.append(row)
        cached_runs[run_id] = row
        new_cached += 1
        dirty = True
        log.info(f"  Found {model}/{dataset} = {float(top1):.4f} (state={state})")

    if dirty:
        _save_cache(cache_file, cached_runs)
        log.info(f"Updated SSL cache ({new_cached} new result runs)")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info(f"Collected {len(df)} SSL runs (vit_small)")
    return df


def _collect_supervised(
    entity: str, project: str, refresh: bool = False
) -> pd.DataFrame:
    """Collect supervised baseline results from a W&B project.

    Uses retry-on-429 and negative caching.
    Returns a DataFrame with the best test_accuracy per dataset.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    proj_hash = hashlib.md5(f"{entity}/{project}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"supervised_v2_{proj_hash}.json"

    cached_runs = _load_cache(cache_file, refresh)

    api = wandb.Api(timeout=60)
    runs = _retry_on_429(
        lambda: list(api.runs(f"{entity}/{project}", per_page=200))
    )
    log.info(f"Found {len(runs)} supervised runs in {entity}/{project}")

    rows = []
    new_cached = 0
    dirty = False

    for run in tqdm(runs, desc="Scanning supervised runs"):
        run_id = run.id

        if run_id in cached_runs:
            entry = cached_runs[run_id]
            if not entry.get("_skip"):
                rows.append(entry)
            continue

        state = _retry_on_429(lambda: run.state)

        # Skip non-terminal runs — don't cache them
        if state not in _TERMINAL_STATES:
            continue

        config = _retry_on_429(lambda: run.config)
        summary = _retry_on_429(lambda: run.summary)

        dataset = config.get("dataset", "")
        if not dataset:
            cached_runs[run_id] = {"_skip": True, "reason": "no_dataset"}
            dirty = True
            continue

        dataset_key = dataset.lower()
        if dataset_key in SKIP_DATASETS:
            cached_runs[run_id] = {"_skip": True, "reason": "skip_dataset"}
            dirty = True
            continue

        test_acc = summary.get("test_accuracy")
        if test_acc is None:
            cached_runs[run_id] = {"_skip": True, "reason": "no_test_accuracy"}
            dirty = True
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
        dirty = True

    if dirty:
        _save_cache(cache_file, cached_runs)
        log.info(f"Updated supervised cache ({new_cached} new result runs)")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep best test_accuracy per dataset
    df = (
        df.sort_values("test_accuracy", ascending=False)
        .drop_duplicates(subset=["dataset"], keep="first")
        .reset_index(drop=True)
    )
    return df


def pivot_table(
    ssl_df: pd.DataFrame,
    supervised_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Pivot SSL results into {dataset x method} numeric table.

    Returns (mean_table, std_table). std_table is None if no seed runs exist.
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
    return mapping.get(raw, raw)


def _format_latex(table: pd.DataFrame, std_table: pd.DataFrame | None = None) -> str:
    """Format a pivot table as a booktabs LaTeX table."""
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None

    ssl_cols = [c for c in pct.columns if c not in ("Average", "Supervised")]
    has_supervised = "Supervised" in pct.columns

    model_labels = [_display_name(m, _MODEL_DISPLAY_NAMES) for m in ssl_cols]
    n_ssl = len(ssl_cols)
    n_extra = 1 if has_supervised else 0
    n_total_data = n_ssl + n_extra + 1

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

    header = "\\textbf{Dataset}"
    for m in model_labels:
        header += f" & {m}"
    if has_supervised:
        header += " & Supervised"
    header += " & \\textbf{Avg.} \\\\"
    lines.append(header)
    lines.append("\\midrule")

    all_data_cols = ssl_cols + (["Supervised"] if has_supervised else [])
    for ds in datasets:
        ds_label = _display_name(ds, _DATASET_DISPLAY_NAMES)
        row = ds_label
        for col in all_data_cols:
            row += f" & {_fmt(pct.loc[ds, col], _get_std(ds, col))}"
        row += f" & {_fmt(pct.loc[ds, 'Average'], _get_std(ds, 'Average'))}"
        row += " \\\\"
        lines.append(row)

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
    parser.add_argument("--entity", default="samibg", help="W&B entity for SSL runs")
    parser.add_argument("--project", default="finalized-stable-datasets", help="W&B project for SSL runs")
    parser.add_argument("--output", default="offline_probe_results.csv", help="Output CSV path")
    parser.add_argument("--latex", default=None, help="Output LaTeX table path (.tex); auto-generated if not set")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass caches and re-download all data from W&B",
    )
    parser.add_argument(
        "--supervised-entity",
        default="samibg",
        help="W&B entity for supervised baselines",
    )
    parser.add_argument(
        "--supervised-project",
        default="finalized-stable-datasets",
        help="W&B project for supervised baselines",
    )
    parser.add_argument(
        "--tag",
        default="rescaled",
        help="Only include lejepa/dino runs with this wandb tag (e.g. rescaled, final_runs)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Collect SSL results
    ssl_df = _collect_ssl(args.entity, args.project, refresh=args.refresh, require_tag=args.tag)
    if ssl_df.empty:
        log.warning("No SSL runs found.")
        return

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
    print((table * 100).round(1).to_markdown())

    if std_table is not None:
        print(f"\n=== Seed Std Dev (dataset x method) ===")
        print((std_table * 100).round(1).to_markdown())

    if args.output:
        ssl_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    # Auto-generate LaTeX file with datetime stamp
    latex_path = args.latex or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
    latex_str = _format_latex(table, std_table=std_table)
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
