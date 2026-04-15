"""Render benchmark results from W&B into LaTeX summary tables.

Fetches runs from a W&B project, validates them against the expected
hyperparameters from conf/model/*.yaml, and writes one table per
evaluation metric (linear probe + kNN) to ``benchmarks/results/``.

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

# Evaluation metrics: (short_name, wandb_summary_key).
METRICS: dict[str, str] = {
    "probe": "eval/linear_probe_top1_epoch",
    "knn": "eval/knn_probe_top1",
}

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

    Only includes finished runs. Older runs sometimes have empty W&B
    configs — in that case we parse the model and dataset from the run
    name. All metrics in :data:`METRICS` are fetched per run; rows that
    have none of them are skipped.

    Returns one row per run with one column per metric short-name in
    :data:`METRICS`.
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

        metric_values: dict[str, float] = {}
        for short, wandb_key in METRICS.items():
            val = summary.get(wandb_key)
            if val is not None:
                metric_values[short] = float(val)
        if not metric_values:
            skipped_no_metric += 1
            continue

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "seed": config.get("seed"),
                "is_seed_run": "seed" in set(_retry(lambda: run.tags or [])),
                **metric_values,
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


def pivot_table(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Pivot into {dataset x method} table for a given metric column.

    For seed runs: uses mean across seeds.
    For non-seed runs: uses the best run per (dataset, method).
    Returns (table, std_table).  std_table is None if no seed runs exist.
    """
    if metric not in df.columns:
        return pd.DataFrame(), None
    df = df.dropna(subset=[metric])
    if df.empty:
        return pd.DataFrame(), None

    is_seed = df["is_seed_run"].fillna(False).astype(bool)
    primary = (
        df[~is_seed].sort_values(metric, ascending=False).drop_duplicates(subset=["dataset", "model"], keep="first")
    )
    table = primary.pivot_table(index="dataset", columns="model", values=metric, aggfunc="max")

    std_table = None
    seed_df = df[is_seed]
    if not seed_df.empty:
        seed_mean = seed_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
        std_table = seed_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="std")
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


def format_latex(
    table: pd.DataFrame,
    std_table: pd.DataFrame | None = None,
    *,
    secondary: pd.DataFrame | None = None,
    secondary_std: pd.DataFrame | None = None,
) -> str:
    """Format pivot table as a booktabs LaTeX table (percentages).

    If *secondary* is provided, cells render as ``primary / secondary``
    so a single table carries both metrics at once (e.g. probe/knn).
    The ``std`` tables are only used when the corresponding primary /
    secondary table is present.
    """
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None
    pct2 = secondary * 100 if secondary is not None else None
    pct2_std = secondary_std * 100 if secondary_std is not None else None

    method_cols = [c for c in pct.columns if c != "Average"]
    datasets = [idx for idx in pct.index if idx != "Average"]

    def _fmt_one(val, std_val=None):
        if pd.isna(val):
            return "---"
        if std_val is not None and not pd.isna(std_val):
            return f"{val:.1f} {{\\scriptsize $\\pm$ {std_val:.1f}}}"
        return f"{val:.1f}"

    def _std_at(tbl, ds, col):
        if tbl is None or ds not in tbl.index or col not in tbl.columns:
            return None
        return tbl.loc[ds, col]

    def _cell(ds, col):
        primary = _fmt_one(pct.loc[ds, col], _std_at(pct_std, ds, col))
        if pct2 is None:
            return primary
        if ds in pct2.index and col in pct2.columns:
            sec_val = pct2.loc[ds, col]
        else:
            sec_val = float("nan")
        secondary_s = _fmt_one(sec_val, _std_at(pct2_std, ds, col))
        return f"{primary} / {secondary_s}"

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
            cells.append(_cell(ds, col))
        cells.append(_cell(ds, "Average"))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    cells = ["\\textbf{Average}"]
    for col in method_cols:
        cells.append(_cell("Average", col))
    cells.append(_cell("Average", "Average"))
    lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


# CLI


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

METRIC_TITLES: dict[str, str] = {
    "probe": "Linear Probe Top-1",
    "knn": "kNN Top-1",
}


def main():
    parser = argparse.ArgumentParser(description="Render benchmark results from W&B to LaTeX")
    parser.add_argument("--entity", default="samibg")
    parser.add_argument("--project", default="finalized-stable-datasets")
    parser.add_argument(
        "--split-evals",
        action="store_true",
        help="Emit one LaTeX table per metric "
        "(benchmark_table_probe.tex, benchmark_table_knn.tex) instead of a "
        "single combined table with 'probe / knn' cells.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_params = _load_expected_params()
    df = collect_runs(args.entity, args.project, expected_params)

    if df.empty:
        print("No runs found.")
        return

    # Pivot each metric (may yield empty frames for missing metrics).
    pivots: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = {
        short: pivot_table(df, short) for short in METRICS
    }

    summary_cols = ["model", "dataset"] + [m for m in METRICS if m in df.columns]
    print(f"\n=== Results ({len(df)} runs) ===")
    print(df[summary_cols].to_string(index=False))

    for short, (tbl, _std) in pivots.items():
        if tbl.empty:
            continue
        print(f"\n=== {METRIC_TITLES.get(short, short)} (dataset x method) ===")
        print((tbl * 100).round(1).to_markdown())

    df.to_csv(DEFAULT_CSV_PATH, index=False)
    print(f"\nSaved CSV to {DEFAULT_CSV_PATH}")

    if args.split_evals:
        for short, (tbl, std) in pivots.items():
            if tbl.empty:
                continue
            out_path = RESULTS_DIR / f"benchmark_table_{short}.tex"
            out_path.write_text(format_latex(tbl, std))
            print(f"LaTeX table saved to {out_path}")
    else:
        # Combined: primary = probe (if present), secondary = knn.
        probe_tbl, probe_std = pivots.get("probe", (pd.DataFrame(), None))
        knn_tbl, knn_std = pivots.get("knn", (pd.DataFrame(), None))
        if probe_tbl.empty and not knn_tbl.empty:
            # Fall back to knn-only when probe is missing.
            primary, primary_std = knn_tbl, knn_std
            secondary, secondary_std = None, None
        else:
            primary, primary_std = probe_tbl, probe_std
            secondary = knn_tbl if not knn_tbl.empty else None
            secondary_std = knn_std if not knn_tbl.empty else None
        if primary.empty:
            print("No metric data to render.")
            return
        out_path = RESULTS_DIR / "benchmark_table.tex"
        out_path.write_text(
            format_latex(primary, primary_std, secondary=secondary, secondary_std=secondary_std)
        )
        print(f"LaTeX table saved to {out_path} (cells show probe / knn)")


if __name__ == "__main__":
    main()
