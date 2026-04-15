"""Render benchmark results from W&B into a LaTeX summary table.

Fetches runs from a W&B project, validates them against the expected
hyperparameters from conf/model/*.yaml, produces a pivot table
(dataset x method), and writes it as a LaTeX table to
``benchmarks/results/benchmark_table.tex``.

Usage:
    python -m benchmarks.render_latex
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

import wandb
from benchmarks.dataset import DATASET_CONFIGS


CONF_DIR = Path(__file__).resolve().parent / "conf" / "model"

METRIC = "eval/linear_probe_top1_epoch"

# Datasets included in the reported benchmark suite. Any run whose
# dataset (from config or parsed from run name) is not in this set is
# silently skipped. Add or remove entries here to change the suite.
INCLUDED_DATASETS: set[str] = {
    "arabiccharacters",
    "arabicdigits",
    "beans",
    "cifar10",
    "cifar100",
    "country211",
    "cub200",
    "dtd",
    "emnist",
    "fashionmnist",
    "fgvcaircraft",
    "flowers102",
    "food101",
    "imagenette",
    "medmnist",
    "notmnist",
    "rockpaperscissor",
    "stl10",
    "svhn",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "simclr": "SimCLR",
    "dino": "DINO",
    "mae": "MAE",
    "lejepa": "LeJEPA",
    "nnclr": "NNCLR",
    "barlow_twins": "Barlow Twins",
    "supervised": "Supervised",
}


def _display_name(key: str) -> str:
    """Look up display name: datasets from DATASET_CONFIGS, models from MODEL_DISPLAY_NAMES."""
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    if key in DATASET_CONFIGS:
        return DATASET_CONFIGS[key].display_name
    return key


# Read expected hyperparams from YAML configs


def _load_expected_params() -> dict[str, dict[str, dict]]:
    """Load expected (model, dataset) → {batch_size, max_epochs, lr} from YAML configs.

    Returns {model_name: {dataset_name: {param: value, ...}, ...}, ...}.
    """
    expected = {}
    for yaml_path in sorted(CONF_DIR.glob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["name"]
        params = cfg.get("params", {})
        defaults = params.get("default", {})
        lr = cfg.get("vit_optimizer", {}).get("lr")

        model_expected = {}
        for ds_name, ds_params in params.items():
            if ds_name == "default":
                continue
            entry = {**defaults, **ds_params}
            if lr is not None:
                entry["lr"] = lr
            # Effective batch size: batch_size (before accum division)
            accum = entry.get("accumulate_grad_batches", 1)
            entry["effective_batch_size"] = (
                entry.get("batch_size", 256) * accum if accum > 1 else entry.get("batch_size", 256)
            )
            model_expected[ds_name] = entry

        expected[model_name] = model_expected
    return expected


# W&B helpers


def _retry(fn, max_retries=5):
    """Call fn() with exponential backoff on HTTP 429."""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                time.sleep(2**attempt)
            else:
                raise
    return fn()


def _matches_expected(config: dict, expected: dict) -> bool:
    """Check if a run's W&B config matches expected hyperparams.

    Missing fields pass through instead of rejecting — older runs often
    have empty configs on W&B but we still want to count them.
    """
    if not config:
        return True

    if "max_epochs" in expected:
        run_epochs = config.get("max_epochs")
        if run_epochs is not None and int(run_epochs) != expected["max_epochs"]:
            return False

    if "lr" in expected:
        run_lr = config.get("lr")
        if run_lr is not None and abs(float(run_lr) - expected["lr"]) > 1e-8:
            return False

    if "effective_batch_size" in expected:
        bs = config.get("batch_size")
        if bs is not None:
            accum = config.get("accumulate_grad_batches", 1)
            ebs = int(bs) * int(accum)
            if ebs != expected["effective_batch_size"]:
                return False

    return True


_KNOWN_MODELS = {"simclr", "dino", "mae", "lejepa", "nnclr", "barlow_twins", "supervised"}


def _parse_name(name: str, backbone: str = "vit_small") -> tuple[str | None, str | None]:
    """Parse ``'{model}_{backbone}_{dataset}'`` from a W&B run name.

    Handles multi-word model names (e.g., 'barlow_twins'). Returns
    ``(model, dataset)`` or ``(None, None)`` on failure.
    """
    tag = f"_{backbone}_"
    if tag not in name:
        return None, None
    model_part, _, dataset_part = name.partition(tag)
    if model_part not in _KNOWN_MODELS:
        return None, None
    return model_part, dataset_part.lower()


# Collection


def collect_runs(
    entity: str,
    project: str,
    expected_params: dict[str, dict[str, dict]],
    backbone: str = "vit_small",
) -> pd.DataFrame:
    """Fetch runs from W&B and filter against expected hyperparameters.

    Only includes finished runs with the target metric in their summary.
    Older runs sometimes have empty W&B configs — in that case we parse
    the model and dataset from the run name.

    Per (model, dataset) we keep the highest linear-probe top-1 value.
    """
    api = wandb.Api(timeout=60)
    filters = {"config.backbone": backbone}
    runs = _retry(lambda: list(api.runs(f"{entity}/{project}", filters=filters, per_page=1000)))
    print(f"Found {len(runs)} {backbone} runs in {entity}/{project}")

    rows = []
    skipped_no_metric = 0
    skipped_bad_params = 0
    skipped_unparseable = 0
    skipped_excluded: dict[str, int] = {}
    for run in tqdm(runs, desc="Scanning runs"):
        state = _retry(lambda: run.state)
        if state != "finished":
            continue

        config = _retry(lambda: run.config)
        summary = _retry(lambda: run.summary)

        model = config.get("model") or ""
        dataset = (config.get("dataset") or "").lower()

        # Fall back to parsing the run name when config is empty
        if not model or not dataset:
            name_model, name_ds = _parse_name(run.name, backbone)
            if name_model:
                model = model or name_model
                dataset = dataset or name_ds

        if not model or not dataset:
            skipped_unparseable += 1
            continue

        if dataset not in INCLUDED_DATASETS:
            skipped_excluded[dataset] = skipped_excluded.get(dataset, 0) + 1
            continue

        model_params = expected_params.get(model, {})
        ds_params = model_params.get(dataset)
        if ds_params is not None and not _matches_expected(config, ds_params):
            skipped_bad_params += 1
            continue

        top1 = summary.get(METRIC)
        if top1 is None:
            skipped_no_metric += 1
            continue

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "seed": config.get("seed"),
                "is_seed_run": "seed" in set(_retry(lambda: run.tags or [])),
                METRIC: float(top1),
            }
        )

    print(
        f"Kept {len(rows)} runs | "
        f"skipped: unparseable={skipped_unparseable}, "
        f"bad_params={skipped_bad_params}, "
        f"no_metric={skipped_no_metric}"
    )
    if skipped_excluded:
        excluded_summary = ", ".join(
            f"{d}={n}" for d, n in sorted(skipped_excluded.items(), key=lambda x: -x[1])
        )
        print(
            f"WARNING: skipped {sum(skipped_excluded.values())} runs whose dataset is "
            f"not in INCLUDED_DATASETS: {excluded_summary}"
        )
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# Pivot table


def pivot_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Pivot into {dataset x method} table.

    For seed runs: uses mean across seeds.
    For non-seed runs: uses the best run per (dataset, method).
    Returns (table, std_table).  std_table is None if no seed runs exist.
    """
    is_seed = df["is_seed_run"].fillna(False).astype(bool)
    primary = (
        df[~is_seed].sort_values(METRIC, ascending=False).drop_duplicates(subset=["dataset", "model"], keep="first")
    )
    table = primary.pivot_table(index="dataset", columns="model", values=METRIC, aggfunc="max")

    std_table = None
    seed_df = df[is_seed]
    if not seed_df.empty:
        seed_mean = seed_df.pivot_table(index="dataset", columns="model", values=METRIC, aggfunc="mean")
        std_table = seed_df.pivot_table(index="dataset", columns="model", values=METRIC, aggfunc="std")
        # Overlay seed means where available
        for col in seed_mean.columns:
            mask = seed_mean[col].notna()
            if col in table.columns:
                table.loc[mask, col] = seed_mean.loc[mask, col]
            else:
                table[col] = seed_mean[col]

    # Averages
    table["Average"] = table.mean(axis=1)
    avg_row = table.drop(columns="Average").mean()
    avg_row["Average"] = table["Average"].mean()
    table.loc["Average"] = avg_row

    if std_table is not None:
        std_table["Average"] = std_table.mean(axis=1)
        std_table.loc["Average"] = std_table.mean()

    return table, std_table


# LaTeX formatting


def format_latex(table: pd.DataFrame, std_table: pd.DataFrame | None = None) -> str:
    """Format pivot table as a booktabs LaTeX table (percentages)."""
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None

    method_cols = [c for c in pct.columns if c != "Average"]
    datasets = [idx for idx in pct.index if idx != "Average"]

    def _fmt(val, std_val=None):
        if pd.isna(val):
            return "---"
        if std_val is not None and not pd.isna(std_val):
            return f"{val:.1f} {{\\scriptsize $\\pm$ {std_val:.1f}}}"
        return f"{val:.1f}"

    def _std(ds, col):
        if pct_std is None or ds not in pct_std.index or col not in pct_std.columns:
            return None
        return pct_std.loc[ds, col]

    n_cols = len(method_cols) + 1  # +1 for Average
    lines = [
        f"\\begin{{tabular}}{{l {'c ' * n_cols}}}",
        "\\toprule",
        "\\textbf{Dataset} & " + " & ".join(_display_name(m) for m in method_cols) + " & \\textbf{Avg.} \\\\",
        "\\midrule",
    ]

    for ds in datasets:
        cells = [_display_name(ds)]
        for col in method_cols:
            cells.append(_fmt(pct.loc[ds, col], _std(ds, col)))
        cells.append(_fmt(pct.loc[ds, "Average"], _std(ds, "Average")))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    cells = ["\\textbf{Average}"]
    for col in method_cols:
        cells.append(_fmt(pct.loc["Average", col], _std("Average", col)))
    cells.append(_fmt(pct.loc["Average", "Average"], _std("Average", "Average")))
    lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


# CLI


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_LATEX_PATH = RESULTS_DIR / "benchmark_table.tex"
DEFAULT_CSV_PATH = RESULTS_DIR / "benchmark_results.csv"


def main():
    parser = argparse.ArgumentParser(description="Render benchmark results from W&B to LaTeX")
    parser.add_argument("--entity", default="samibg")
    parser.add_argument("--project", default="finalized-stable-datasets")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_params = _load_expected_params()
    df = collect_runs(args.entity, args.project, expected_params)

    if df.empty:
        print("No runs found.")
        return

    print(f"\n=== Results ({len(df)} runs) ===")
    print(df[["model", "dataset", METRIC]].to_string(index=False))

    table, std_table = pivot_table(df)
    print("\n=== Linear Probe Top-1 (dataset x method) ===")
    print((table * 100).round(1).to_markdown())

    df.to_csv(DEFAULT_CSV_PATH, index=False)
    print(f"\nSaved CSV to {DEFAULT_CSV_PATH}")

    DEFAULT_LATEX_PATH.write_text(format_latex(table, std_table))
    print(f"LaTeX table saved to {DEFAULT_LATEX_PATH}")


if __name__ == "__main__":
    main()
