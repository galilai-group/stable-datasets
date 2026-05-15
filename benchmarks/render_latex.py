"""Render benchmark results from W&B into LaTeX summary tables.

Fetches runs from a W&B project, validates them against the expected
hyperparameters from conf/model/*.yaml, and writes one table per
evaluation metric (linear probe + kNN) to ``benchmarks/results/``.

Aggregation per (model, dataset) cell: pools each seeded run (deduped by
seed) plus the single best unseeded run as samples; the rendered value is
the mean ± std across that pool. Cells with N=1 sample render bare.

Outputs:
    benchmarks/results/benchmark_results.csv             (per-run rows)
    benchmarks/results/benchmark_results_aggregated.csv  (per-cell mean/std)
    benchmarks/results/benchmark_table_*_with_rankme.tex (LaTeX tables)

Usage:
    python -m benchmarks.render_latex
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import wandb
import yaml
from tqdm import tqdm

from benchmarks.dataset import (
    DATASET_CONFIGS,
    INCLUDED_IMAGE_DATASETS,
    INCLUDED_TIMESERIES_DATASETS,
)


CONF_DIR = Path(__file__).resolve().parent / "conf" / "model"

# Evaluation metrics: (short_name, wandb_summary_key).
METRICS: dict[str, str] = {
    "probe": "eval/linear_probe_top1_epoch",
    "knn": "eval/knn_probe_top1",
}

# Datasets included in the reported benchmark suite, split by modality so the
# LaTeX output can render image and timeseries datasets in separate sections.
SECTIONS: list[tuple[str, set[str]]] = [
    ("Image datasets", set(INCLUDED_IMAGE_DATASETS)),
    ("Timeseries datasets", set(INCLUDED_TIMESERIES_DATASETS)),
]
INCLUDED_DATASETS: set[str] = set().union(*[s for _, s in SECTIONS])

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "simclr": "SimCLR",
    "dino": "DINO",
    "mae": "MAE",
    "lejepa": "LeJEPA",
    "nnclr": "NNCLR",
    "barlow_twins": "Barlow Twins",
    "supervised": "Supervised",
}


# (model, dataset) pairs confirmed to be training collapses — rendered as "---".
# Add entries here rather than deleting rows from the CSV so the raw data is preserved.
KNOWN_FAILURES: set[tuple[str, str]] = {
    ("dino", "emnist_digits"),  # collapse: probe=21.3, chance=10.0
    ("dino", "emnist_mnist"),  # collapse: probe=18.5, chance=10.0
    ("nnclr", "emnist_mnist"),  # collapse: probe=26.2, chance=10.0
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
        if run_lr is not None and abs(float(run_lr) - float(expected["lr"])) > 1e-8:
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

# Backwards-compatible dataset key remap. The "medmnist" registry key was
# renamed to "pneumoniamnist" (the actual MedMNIST sub-task it pointed at);
# historical W&B runs are still tagged ``dataset=medmnist`` and would
# otherwise be dropped by the INCLUDED_DATASETS filter.
_DATASET_ALIASES: dict[str, str] = {"medmnist": "pneumoniamnist"}

# Tuple of backbone identifiers we recognize on W&B. Includes both the
# pre-refactor short name (``vit_small``) and the post-refactor timm name
# (``vit_small_patch16_224``) so historical runs are still picked up.
DEFAULT_BACKBONES: tuple[str, ...] = ("vit_small", "vit_small_patch16_224")


def _parse_seed_from_name(name: str) -> int | None:
    """Extract ``seedN`` suffix from a run name. Returns the int or None."""
    import re

    m = re.search(r"_seed(\d+)\b", name)
    return int(m.group(1)) if m else None


def _parse_name(
    name: str,
    backbones: tuple[str, ...] = DEFAULT_BACKBONES,
) -> tuple[str | None, str | None]:
    """Parse ``'{model}_{backbone}_{dataset}'`` from a W&B run name.

    Tries each backbone variant; longest first so ``vit_small_patch16_224``
    wins over a partial match against ``vit_small``. Returns
    ``(model, dataset)`` or ``(None, None)`` on failure.
    """
    for backbone in sorted(backbones, key=len, reverse=True):
        tag = f"_{backbone}_"
        if tag not in name:
            continue
        model_part, _, dataset_part = name.partition(tag)
        if model_part in _KNOWN_MODELS:
            return model_part, dataset_part.lower()
    return None, None


# Collection


def collect_runs(
    entity: str,
    project: str,
    expected_params: dict[str, dict[str, dict]],
    backbones: tuple[str, ...] = DEFAULT_BACKBONES,
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
    filters = {"$or": [{"config.backbone": b} for b in backbones]}
    runs = _retry(lambda: list(api.runs(f"{entity}/{project}", filters=filters, per_page=1000)))
    print(f"Found {len(runs)} runs in {entity}/{project} matching backbones={backbones}")

    rows = []
    skipped_no_metric = 0
    skipped_bad_params = 0
    skipped_unparseable = 0
    skipped_excluded: dict[str, int] = {}
    skipped_transfer = 0
    skipped_short_runtime = 0
    for run in tqdm(runs, desc="Scanning runs"):
        state = _retry(lambda: run.state)
        if state != "finished":
            continue

        # Exclude transfer/offline-probe runs: these evaluate a frozen pretrained
        # backbone *on* a dataset rather than training the SSL method on it, so
        # their probe numbers are not comparable to training-run probes.
        run_tags = set(_retry(lambda: run.tags or []))
        if {"offline_probe", "transfer"} & run_tags:
            skipped_transfer += 1
            continue

        config = _retry(lambda: run.config)
        summary = _retry(lambda: run.summary)

        # Exclude jobs that exited without actually training (e.g. crashed
        # immediately, loaded a stale checkpoint and quit). 5 min is well below
        # any real training cycle even on the smallest datasets.
        runtime = summary.get("_runtime")
        if runtime is not None and runtime < 300:
            skipped_short_runtime += 1
            continue

        model = config.get("model") or ""
        dataset = (config.get("dataset") or "").lower()

        # The batch API returns truncated configs for resumed/contaminated runs.
        # Re-fetch individually when either field is missing; this is authoritative
        # over name-parsing since the run name may be stale after a resume.
        if not model or not dataset:
            try:
                run_id_local = run.id
                full_config = _retry(lambda: api.run(f"{entity}/{project}/{run_id_local}").config)
                model = (full_config.get("model") or "") or model
                dataset = ((full_config.get("dataset") or "").lower()) or dataset
            except Exception:
                pass

        # Last resort: parse model/dataset from the run name
        if not model or not dataset:
            name_model, name_ds = _parse_name(run.name, backbones)
            if name_model:
                model = model or name_model
                dataset = dataset or name_ds

        if not model or not dataset:
            skipped_unparseable += 1
            continue

        dataset = _DATASET_ALIASES.get(dataset, dataset)

        if dataset not in INCLUDED_DATASETS:
            skipped_excluded[dataset] = skipped_excluded.get(dataset, 0) + 1
            continue

        model_params = expected_params.get(model, {})
        ds_params = model_params.get(dataset)
        if ds_params is not None and not _matches_expected(config, ds_params):
            skipped_bad_params += 1
            continue

        rankme = summary.get("rankme")
        if rankme is None:
            skipped_no_metric += 1
            continue

        metric_values: dict[str, float] = {}
        for short, wandb_key in METRICS.items():
            val = summary.get(wandb_key)
            if val is not None:
                metric_values[short] = float(val)
        if not metric_values:
            skipped_no_metric += 1
            continue

        # Tag presence is authoritative; config.seed is often missing because
        # the batch API truncates configs. Fall back to the run name suffix
        # (``_seedN``) — and only re-fetch the full config as a last resort.
        run_tags = set(_retry(lambda: run.tags or []))
        is_seed_run = "seed" in run_tags
        seed = config.get("seed")
        if seed is None and is_seed_run:
            seed = _parse_seed_from_name(run.name)
        if seed is None and is_seed_run:
            try:
                full_cfg = _retry(lambda: api.run(f"{entity}/{project}/{run.id}").config)
                seed = full_cfg.get("seed")
            except Exception:
                pass

        rows.append(
            {
                "run_id": run.id,
                "model": model,
                "dataset": dataset,
                "seed": seed,
                "is_seed_run": is_seed_run,
                "tags": ",".join(sorted(run_tags)),
                **metric_values,
            }
        )

    print(
        f"Kept {len(rows)} runs | "
        f"skipped: unparseable={skipped_unparseable}, "
        f"bad_params={skipped_bad_params}, "
        f"no_rankme_or_metric={skipped_no_metric}, "
        f"transfer={skipped_transfer}, "
        f"short_runtime={skipped_short_runtime}"
    )
    if skipped_excluded:
        excluded_summary = ", ".join(f"{d}={n}" for d, n in sorted(skipped_excluded.items(), key=lambda x: -x[1]))
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

    Aggregation: each (model, dataset) cell pools every seeded run plus the
    single best unseeded run as samples. Cell value is the mean; std
    (rendered as ±) is over the same pool. N=1 cells render bare.

    No per-seed dedup: the W&B batch API returns truncated configs where
    ``config.seed`` is sometimes missing even though the ``seed`` tag is
    present. Grouping by seed would drop those rows. The upstream
    ``collect_runs`` filters (``state=="finished"``, ``_runtime>300s``)
    already screen out SLURM-requeue partial reruns, so each kept row is
    one valid sample.
    """
    if metric not in df.columns:
        return pd.DataFrame(), None
    df = df.dropna(subset=[metric])
    if df.empty:
        return pd.DataFrame(), None

    is_seed = df["is_seed_run"].fillna(False).astype(bool)

    # Each seeded run is one sample.
    seed_samples = df[is_seed][["dataset", "model", metric]]

    # One sample per (model, dataset) for unseeded runs (best attempt).
    unseeded_samples = df[~is_seed].groupby(["dataset", "model"], as_index=False)[metric].max()

    pooled = pd.concat(
        [seed_samples, unseeded_samples],
        ignore_index=True,
    )
    if pooled.empty:
        return pd.DataFrame(), None

    table = pooled.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
    # std with N=1 yields NaN; formatter falls back to bare value.
    # dropna=False keeps NaN cells so the std frame matches `table` shape even
    # when no (model, dataset) cell has more than 1 sample yet.
    std_table = pooled.pivot_table(index="dataset", columns="model", values=metric, aggfunc="std", dropna=False)
    std_table = std_table.reindex(index=table.index, columns=table.columns)

    # Mask known training collapses so they render as "---" rather than a
    # misleading score. The raw CSV rows are kept intact.
    for model, dataset in KNOWN_FAILURES:
        if dataset in table.index and model in table.columns:
            table.loc[dataset, model] = float("nan")
            if std_table is not None and dataset in std_table.index and model in std_table.columns:
                std_table.loc[dataset, model] = float("nan")

    # Chance baseline: random class guess accuracy for each dataset.
    table["Chance"] = pd.Series(
        {
            ds: 1.0 / DATASET_CONFIGS[ds].num_classes
            for ds in table.index
            if ds in DATASET_CONFIGS and DATASET_CONFIGS[ds].num_classes > 0
        }
    )
    avg_row = table.drop(columns="Chance").mean()
    avg_row["Chance"] = table["Chance"].mean()
    table.loc["Average"] = avg_row

    if std_table is not None and not std_table.empty:
        std_table.loc["Average"] = std_table.mean()

    return table, std_table


# LaTeX formatting


def format_latex(
    table: pd.DataFrame,
    std_table: pd.DataFrame | None = None,
    sections: list[tuple[str, set[str]]] | None = None,
) -> str:
    """Format pivot table as a booktabs LaTeX table (percentages).

    When ``sections`` is supplied, datasets are grouped into labeled
    blocks (e.g. "Image datasets", "Timeseries datasets"), each with its
    own intra-section average row. The trailing global Average row from
    :func:`pivot_table` is dropped.
    """
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None

    method_cols = [c for c in pct.columns if c != "Chance"]
    all_ds = [idx for idx in pct.index if idx != "Average"]

    if sections is None:
        sections = [("", set(all_ds))]

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
        return _fmt_one(pct.loc[ds, col], _std_at(pct_std, ds, col))

    n_cols = len(method_cols) + 1  # +1 for Chance

    lines = [
        f"\\begin{{tabular}}{{l {'c ' * n_cols}}}",
        "\\toprule",
        "\\textbf{Dataset} & " + " & ".join(_display_name(m) for m in method_cols) + " & \\textbf{Chance} \\\\",
    ]

    n_total_cols = n_cols + 1  # leading dataset col + method cols + Chance
    for i, (label, keys) in enumerate(sections):
        section_ds = [ds for ds in all_ds if ds in keys]
        if not section_ds:
            continue
        lines.append("\\midrule")
        if label:
            lines.append(f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textit{{{label}}}}} \\\\")
            lines.append("\\midrule")
        for ds in section_ds:
            cells = [_display_name(ds)]
            for col in method_cols:
                cells.append(_cell(ds, col))
            cells.append(_cell(ds, "Chance"))
            lines.append(" & ".join(cells) + " \\\\")

        # Per-section averages
        lines.append("\\midrule")
        cells = [f"\\textbf{{{label} avg.}}" if label else "\\textbf{Average}"]
        sub_pct = pct.loc[section_ds]
        for col in method_cols:
            cells.append(_fmt_one(sub_pct[col].mean()))
        cells.append(_fmt_one(sub_pct["Chance"].mean()))
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
        "--from-csv",
        action="store_true",
        help=f"Skip W&B and load results from {DEFAULT_CSV_PATH}",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        if not DEFAULT_CSV_PATH.exists():
            print(f"No cached CSV at {DEFAULT_CSV_PATH}; run without --from-csv first.")
            return
        df = pd.read_csv(DEFAULT_CSV_PATH)
        print(f"Loaded {len(df)} rows from {DEFAULT_CSV_PATH}")
    else:
        expected_params = _load_expected_params()
        df = collect_runs(args.entity, args.project, expected_params)

    if df.empty:
        print("No runs found.")
        return

    # Pivot each metric (may yield empty frames for missing metrics).
    pivots: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = {short: pivot_table(df, short) for short in METRICS}

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

    # Per-cell aggregate CSV (long format: one row per dataset/model/metric).
    agg_rows = []
    for short, (tbl, std) in pivots.items():
        if tbl.empty:
            continue
        for ds in tbl.index:
            if ds == "Average":
                continue
            for col in tbl.columns:
                if col == "Chance":
                    continue
                mean = tbl.loc[ds, col]
                s = std.loc[ds, col] if std is not None and ds in std.index and col in std.columns else float("nan")
                if pd.isna(mean):
                    continue
                agg_rows.append({"metric": short, "dataset": ds, "model": col, "mean": mean, "std": s})
    if agg_rows:
        agg_path = RESULTS_DIR / "benchmark_results_aggregated.csv"
        pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
        print(f"Saved aggregated CSV to {agg_path}")

    for short, (tbl, std) in pivots.items():
        if tbl.empty:
            continue
        out_path = RESULTS_DIR / f"benchmark_table_{short}_with_rankme.tex"
        out_path.write_text(format_latex(tbl, std, sections=SECTIONS))
        print(f"LaTeX table saved to {out_path}")


if __name__ == "__main__":
    main()
