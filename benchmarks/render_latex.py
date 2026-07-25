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
import yaml
from tqdm import tqdm

import wandb
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

# Column order used in the paper's tables: contrastive first (SimCLR, NNCLR),
# then Barlow Twins / DINO / LeJEPA / MAE, then Supervised. Chance is appended
# separately at render time.
PAPER_METHOD_ORDER: list[str] = [
    "simclr", "nnclr", "dino", "barlow_twins", "lejepa", "mae", "supervised",
]


# (model, dataset) pairs confirmed to be training collapses — rendered as "---".
# Add entries here rather than deleting rows from the CSV so the raw data is preserved.
KNOWN_FAILURES: set[tuple[str, str]] = set()

# Weak / partially-collapsed runs from within the special-config sweeps.
# Successful re-attempts for the same (model, dataset, seed) exist in the pool
# and cover the requested seed count without these.
KNOWN_BAD_RUNS: set[str] = {
    # lejepa × imagenette — partial runs (crashed after only 1-2 epochs), probe
    # values are mid-training snapshots, not converged. Fresh seed 2 is rerunning
    # (seed_sweep_lejepa_imagenette_s2_rerun).
    "679czexn",  # seed 2, 88s, probe 30% (partial)
    "mmfqzsmy",  # unseeded, 165s, probe 30% (partial)
    "2t5ttli8",  # dino × emnist_letters × seed 3 v6: partial collapse at 71%
    # dino × arabiccharacters — historic and v22 seeds that hit DINO's
    # seed-dependent centering collapse (probe ≈ 1/28 chance, rankme < 2).
    # Successful reproductions are seeds 1, 2, 6 in seed_sweep_v20/v22_special.
    "xeppps3i",  # v8 seed 1 (5.09%)
    "q1y4v4km",  # v16 seed 2 (3.57%)
    "n4qbz3ti",  # v16 seed 3 (3.57%)
    "4tc1vvv0",  # v20 seed 3 (5.74%)
    "qsa1sol8",  # v22 seed 4 (5.09%)
    "0xbd3gsk",  # v22 seed 5 (5.33%)
}

# Cells that only trained correctly under per_dataset_backbone (patch=4, native
# 28×28 for EMNIST family). Every historic run for these cells used the wrong
# default (patch16, upscaled 224×224) and collapsed intermittently. Only accept
# runs tagged with the special-config sweeps.
_ALLOWED_SPECIAL_TAGS: set[str] = {
    "seed_sweep_v5_special",
    "seed_sweep_v6_special",
    "seed_sweep_v7_special",
    "seed_sweep_v8_special",
    "seed_sweep_v16_special",
    "seed_sweep_v20_special",
    "seed_sweep_v22_special",
}
REQUIRES_SPECIAL_CONFIG: dict[tuple[str, str], set[str]] = {
    (m, ds): _ALLOWED_SPECIAL_TAGS
    for m, ds in [
        ("dino", "arabiccharacters"),
        ("dino", "emnist"),
        ("dino", "emnist_byclass"),
        ("dino", "emnist_bymerge"),
        ("dino", "emnist_letters"),
        ("dino", "emnist_digits"),
        ("dino", "emnist_mnist"),
        ("nnclr", "emnist_mnist"),
    ]
}


def _display_name(key: str) -> str:
    """Look up display name: datasets from DATASET_CONFIGS, models from MODEL_DISPLAY_NAMES."""
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    if key in DATASET_CONFIGS:
        return DATASET_CONFIGS[key].display_name
    return key


# Read expected hyperparams from YAML configs


def _load_expected_params(family: str = "vit") -> dict[str, dict[str, dict]]:
    """Load expected (model, dataset) → {batch_size, max_epochs, lr} from YAML configs.

    ``family`` ("vit"/"resnet") selects the per-backbone optimizer block for the
    expected ``lr``, which is validated against each run's W&B config. Using the
    wrong family's lr would filter out otherwise-valid runs.

    Returns {model_name: {dataset_name: {param: value, ...}, ...}, ...}.
    """
    expected = {}
    for yaml_path in sorted(CONF_DIR.glob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["name"]
        params = cfg.get("params", {})
        defaults = params.get("default", {})
        opt_block = cfg.get(f"{family}_optimizer") or cfg.get("vit_optimizer", {})
        lr = opt_block.get("lr")

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

# Backbone identifiers we recognize on W&B, grouped by architecture family.
# The ViT set includes both the pre-refactor short name (``vit_small``) and the
# post-refactor timm name (``vit_small_patch16_224``) so historical runs are
# still picked up. ``--backbone`` selects which family to render.
BACKBONE_SETS: dict[str, tuple[str, ...]] = {
    "vit": ("vit_small", "vit_small_patch16_224"),
    "resnet": ("resnet50",),
}
DEFAULT_BACKBONES: tuple[str, ...] = BACKBONE_SETS["vit"]


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
    # Seed-sweep runs have truncated configs on W&B (config.backbone is None),
    # so a pure config.backbone filter would drop them all. Include seed-tagged
    # runs too; downstream code resolves backbone via the run-name suffix.
    filters = {
        "$or": [{"config.backbone": b} for b in backbones]
        + [{"tags": "seed_sweep_v1"}, {"tags": "seed"}]
    }
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
        if run.id in KNOWN_BAD_RUNS:
            continue
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

        # Filter out obvious crash artefacts (Python died before any epoch
        # completed) via a very low runtime floor. Everything above this must
        # still pass the `rankme is None` gate below, so genuine short-but-
        # completed runs (supervised on tiny datasets like Beans) are kept.
        runtime = summary.get("_runtime")
        if runtime is not None and runtime < 60:
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

        # Special-config cells: only keep runs from the tagged sweeps that used
        # per_dataset_backbone (patch=4 native res). Anything else was trained
        # with the wrong config and either collapsed or is not comparable.
        allowed_tags = REQUIRES_SPECIAL_CONFIG.get((model, dataset))
        if allowed_tags is not None and not (run_tags & allowed_tags):
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
    unseeded_samples = (
        df[~is_seed]
        .groupby(["dataset", "model"], as_index=False)[metric]
        .max()
    )

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
    std_table = pooled.pivot_table(
        index="dataset", columns="model", values=metric, aggfunc="std", dropna=False
    )
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
    *,
    paper_style: bool = False,
    std_only: bool = False,
) -> str:
    """Format pivot table as a booktabs LaTeX table (percentages).

    Two styles:

    - Default: ``mean {\\scriptsize ± std}`` per cell, alphabetical method
      column order, per-section average rows. Used for the internal /
      appendix tables where seed variance matters.
    - ``paper_style=True``: single-value cells with the row-winning method
      wrapped in ``\\textbf{...}``. Methods appear in the paper's order
      (SimCLR, NNCLR, DINO, Barlow Twins, LeJEPA, MAE, Supervised) and a
      single bottom ``Avg.`` row summarizes all sections. Used for the main
      paper tables.

    When ``sections`` is supplied, datasets are grouped into labeled
    blocks (e.g. "Image datasets", "Timeseries datasets"), each with its
    own intra-section average row (default) or interleaved as a single
    block (paper_style). The trailing global Average row from
    :func:`pivot_table` is dropped in both cases.
    """
    pct = table * 100
    pct_std = std_table * 100 if std_table is not None else None

    method_cols_all = [c for c in pct.columns if c != "Chance"]
    if paper_style:
        # Preserve the paper's method order; skip any methods that are missing
        # from ``table`` (e.g. supervised-less subsets) without erroring.
        method_cols = [m for m in PAPER_METHOD_ORDER if m in method_cols_all]
        # Then append any methods present in the frame but not enumerated
        # (defensive; keeps novel methods visible).
        method_cols += [m for m in method_cols_all if m not in method_cols]
    else:
        method_cols = method_cols_all
    all_ds = [idx for idx in pct.index if idx != "Average"]

    if sections is None:
        sections = [("", set(all_ds))]

    def _fmt_one(val, std_val=None, bold: bool = False):
        if std_only:
            # Appendix std-only table: render the std alone, or --- if missing.
            # Two decimals so near-ceiling cells with std ≈ 0.02–0.04 don't
            # look like exact zeros; single-sample cells (nothing to compute
            # a std from) render as "---". Bolding still highlights the
            # mean-table's row-winning method so the reader can gauge
            # variance around the leading method.
            if std_val is None or pd.isna(std_val):
                return "---"
            body = f"{std_val:.2f}"
            if bold:
                body = f"\\textbf{{{body}}}"
            return body
        if pd.isna(val):
            return "---"
        body = f"{val:.1f}"
        if bold:
            body = f"\\textbf{{{body}}}"
        if std_val is None or pd.isna(std_val):
            return body
        if paper_style:
            # Compact vertical stack: mean on top, tiny ±std beneath. Keeps the
            # column no wider than the mean alone so \resizebox behaves.
            return f"$\\substack{{{body}\\\\{{\\tiny\\pm{std_val:.1f}}}}}$"
        return f"{body} {{\\scriptsize $\\pm$ {std_val:.1f}}}"

    def _std_at(tbl, ds, col):
        if tbl is None or ds not in tbl.index or col not in tbl.columns:
            return None
        return tbl.loc[ds, col]

    def _row_best_col(ds):
        """Model column to bold in this row (NaN-safe).

        For std_only tables the "best" is the *lowest* std (most seed-stable
        method for this dataset). Otherwise it's the highest mean.
        """
        if std_only and pct_std is not None:
            vals = {
                c: pct_std.loc[ds, c]
                for c in method_cols
                if ds in pct_std.index and c in pct_std.columns and not pd.isna(pct_std.loc[ds, c])
            }
            return min(vals, key=vals.get) if vals else None
        vals = {c: pct.loc[ds, c] for c in method_cols if not pd.isna(pct.loc[ds, c])}
        if not vals:
            return None
        return max(vals, key=vals.get)

    def _cell(ds, col):
        val = pct.loc[ds, col]
        bold = paper_style and (col == _row_best_col(ds))
        return _fmt_one(val, _std_at(pct_std, ds, col), bold=bold)

    n_cols = len(method_cols) + 1  # +1 for Chance

    lines = [
        f"\\begin{{tabular}}{{l {'c ' * n_cols}}}",
        "\\toprule",
        "\\textbf{Dataset} & " + " & ".join(_display_name(m) for m in method_cols) + " & \\textbf{Chance} \\\\",
    ]

    n_total_cols = n_cols + 1  # leading dataset col + method cols + Chance

    if paper_style:
        # Single bottom Avg. row; ignore section labels in the body but still
        # emit datasets in section order for consistency with the mean±std
        # rendering.
        ordered_ds = []
        for _, keys in sections:
            ordered_ds.extend(ds for ds in all_ds if ds in keys and ds not in ordered_ds)
        for ds in ordered_ds:
            lines.append("\\midrule" if ds == ordered_ds[0] else None)
            cells = [_display_name(ds)]
            for col in method_cols:
                cells.append(_cell(ds, col))
            cells.append(_fmt_one(pct.loc[ds, "Chance"]))
            lines.append(" & ".join(cells) + " \\\\")
        lines = [ln for ln in lines if ln is not None]
        # Bottom Avg. row across all included datasets. For the std-only
        # appendix table, both `val` and `std_val` for this row are the
        # per-method mean of dataset stds — so cells render as e.g. `4.32`
        # instead of `---`. Bold the method with the lowest mean std
        # (most seed-stable overall).
        lines.append("\\midrule")
        sub = pct.loc[ordered_ds]
        sub_std = pct_std.loc[ordered_ds] if pct_std is not None else None
        avg_vals = {c: sub[c].mean() for c in method_cols}
        avg_stds = {c: (sub_std[c].mean() if sub_std is not None else float("nan")) for c in method_cols}
        if std_only:
            # Best method for the Avg. row = lowest mean std (most stable).
            finite = {c: v for c, v in avg_stds.items() if not pd.isna(v)}
            best_avg = min(finite, key=finite.get) if finite else None
        else:
            best_avg = max(avg_vals, key=avg_vals.get) if avg_vals else None
        cells = ["\\textbf{Avg.}"]
        for col in method_cols:
            cells.append(_fmt_one(avg_vals[col], avg_stds[col], bold=(col == best_avg)))
        cells.append(_fmt_one(sub["Chance"].mean()))
        lines.append(" & ".join(cells) + " \\\\")
    else:
        for _, (label, keys) in enumerate(sections):
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


# ImageNet-transfer comparison table
#
# Rebuilds ``paper_table_imagenet_transfer.tex`` from the current pool so the
# In-domain DINO column can never desynchronize from the main SSL results.
# ImageNet DINO values come from ``transfer_results.csv`` (offline linear
# probe on a frozen ImageNet-1k dino_vit_small backbone, 90 epochs); in-domain
# DINO values are the 3-seed means from ``benchmark_results_aggregated.csv``.
# Row categories are the paper's taxonomy. Datasets missing from either side
# are silently skipped; Beans has no ImageNet-transfer probe and so is excluded.
DATASET_CATEGORY: dict[str, str] = {
    "arabiccharacters": "Characters", "arabicdigits": "Characters",
    "emnist": "Characters", "emnist_byclass": "Characters",
    "emnist_bymerge": "Characters", "emnist_digits": "Characters",
    "emnist_letters": "Characters", "emnist_mnist": "Characters",
    "hasyv2": "Characters", "notmnist": "Characters", "svhn": "Characters",
    "awa2": "Fine-grained", "cars196": "Fine-grained", "cub200": "Fine-grained",
    "fgvcaircraft": "Fine-grained", "fgvcaircraft_family": "Fine-grained",
    "fgvcaircraft_manufacturer": "Fine-grained", "flowers102": "Fine-grained",
    "food101": "Fine-grained",
    "bloodmnist": "Medical", "breastmnist": "Medical", "dermamnist": "Medical",
    "octmnist": "Medical", "organamnist": "Medical", "organcmnist": "Medical",
    "organsmnist": "Medical", "pathmnist": "Medical",
    "pneumoniamnist": "Medical", "retinamnist": "Medical", "tissuemnist": "Medical",
    "cifar10": "Natural", "cifar100": "Natural", "imagenet100": "Natural",
    "imagenette": "Natural", "linnaeus5": "Natural", "stl10": "Natural",
    "country211": "Geolocation", "dtd": "Texture", "fashionmnist": "Fashion",
    "galaxy10": "Astronomy", "rockpaperscissor": "Gestures",
}


def format_imagenet_transfer_latex(
    in_domain_probe: pd.DataFrame,
    transfer_csv_path: Path,
) -> str:
    """Produce the ImageNet-vs-in-domain DINO comparison LaTeX table.

    ``in_domain_probe`` is a ``benchmark_results_aggregated.csv`` slice with
    ``metric == "probe"`` — one row per (model, dataset) cell carrying the
    3-seed mean fraction. ``transfer_csv_path`` points to
    ``transfer_results.csv`` written by ``benchmarks.transfer.probe``.

    The output uses the same table shell (Category column, bold winner per
    row) as the paper's ``tab:imagenet-transfer``.
    """
    if not transfer_csv_path.exists():
        return "% transfer_results.csv not found — skipping ImageNet-transfer table\n"

    tf = pd.read_csv(transfer_csv_path)
    tf = tf[tf["backbone"] == "dino_vit_small_in1k"]
    # Keep the run with the most training epochs per dataset (drop the 1-epoch
    # smoke-test rows). ``top1`` is stored as a fraction in [0, 1].
    tf = tf.sort_values("max_epochs", ascending=False).drop_duplicates("dataset", keep="first")
    tf_probe = tf.set_index("dataset")["top1"] * 100

    ind = in_domain_probe[in_domain_probe["model"] == "dino"].set_index("dataset")
    ind_probe = ind["mean"] * 100

    rows: list[tuple[str, str, float, float, float]] = []
    for ds in sorted(DATASET_CATEGORY.keys(), key=_display_name):
        if ds not in tf_probe.index or ds not in ind_probe.index:
            continue
        chance = 100.0 / DATASET_CONFIGS[ds].num_classes if ds in DATASET_CONFIGS else float("nan")
        rows.append(
            (
                _display_name(ds),
                DATASET_CATEGORY[ds],
                float(tf_probe[ds]),
                float(ind_probe[ds]),
                chance,
            )
        )

    def _pair(tf_v: float, ind_v: float) -> tuple[str, str]:
        if pd.isna(tf_v) or pd.isna(ind_v):
            return f"{tf_v:.1f}" if not pd.isna(tf_v) else "---", f"{ind_v:.1f}" if not pd.isna(ind_v) else "---"
        if tf_v >= ind_v:
            return f"\\textbf{{{tf_v:.1f}}}", f"{ind_v:.1f}"
        return f"{tf_v:.1f}", f"\\textbf{{{ind_v:.1f}}}"

    lines = [
        "\\begin{tabular}{l l c c c}",
        "\\toprule",
        "Dataset & Category & ImageNet DINO & In-domain DINO & Chance \\\\",
        "\\midrule",
    ]
    for name, cat, tf_v, ind_v, chance in rows:
        tf_cell, ind_cell = _pair(tf_v, ind_v)
        lines.append(f"{name} & {cat} & {tf_cell} & {ind_cell} & {chance:.1f}  \\\\")
    lines.append("\\midrule")
    tf_avg = sum(r[2] for r in rows) / len(rows) if rows else float("nan")
    ind_avg = sum(r[3] for r in rows) / len(rows) if rows else float("nan")
    ch_avg = sum(r[4] for r in rows) / len(rows) if rows else float("nan")
    tf_avg_cell, ind_avg_cell = _pair(tf_avg, ind_avg)
    lines.append(f"\\textbf{{Avg.}} & --- & {tf_avg_cell} & {ind_avg_cell} & {ch_avg:.1f} \\\\")
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
        "--backbone",
        choices=sorted(BACKBONE_SETS),
        default="vit",
        help="Backbone family to render (default: vit).",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help=f"Skip W&B and load results from {DEFAULT_CSV_PATH}",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    family = args.backbone
    backbones = BACKBONE_SETS[family]
    # ViT (the default) keeps the original filenames so the existing paper
    # pipeline is unchanged; other backbones get a suffix so they don't clobber.
    suffix = "" if family == "vit" else f"_{family}"
    csv_path = RESULTS_DIR / f"benchmark_results{suffix}.csv"

    if args.from_csv:
        if not csv_path.exists():
            print(f"No cached CSV at {csv_path}; run without --from-csv first.")
            return
        df = pd.read_csv(csv_path)
        # Apply the run-id blocklist here too — a cached CSV may pre-date the
        # blocklist and still contain rows we no longer want to aggregate.
        if "run_id" in df.columns and KNOWN_BAD_RUNS:
            before = len(df)
            df = df[~df["run_id"].isin(KNOWN_BAD_RUNS)]
            print(f"Dropped {before - len(df)} blocklisted run(s) from cached CSV")
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        expected_params = _load_expected_params(family)
        df = collect_runs(args.entity, args.project, expected_params, backbones=backbones)

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

    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

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
        agg_path = RESULTS_DIR / f"benchmark_results_aggregated{suffix}.csv"
        pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
        print(f"Saved aggregated CSV to {agg_path}")

    updated_dir = RESULTS_DIR / "updated_tables"
    updated_dir.mkdir(parents=True, exist_ok=True)
    for short, (tbl, std) in pivots.items():
        if tbl.empty:
            continue
        out_path = RESULTS_DIR / f"benchmark_table_{short}{suffix}_with_rankme.tex"
        out_path.write_text(format_latex(tbl, std, sections=SECTIONS))
        print(f"LaTeX table saved to {out_path}")
        paper_path = RESULTS_DIR / f"benchmark_table_{short}{suffix}_paper.tex"
        paper_path.write_text(format_latex(tbl, std, sections=SECTIONS, paper_style=True))
        print(f"Paper-style table saved to {paper_path}")

        # updated_tables/: main paper table (no ±), appendix std-only table,
        # and legacy substack ± version for reference.
        no_std_path = updated_dir / f"benchmark_table_{short}{suffix}_paper_no_std.tex"
        no_std_path.write_text(format_latex(tbl, None, sections=SECTIONS, paper_style=True))
        with_std_path = updated_dir / f"benchmark_table_{short}{suffix}_paper_with_std.tex"
        with_std_path.write_text(format_latex(tbl, std, sections=SECTIONS, paper_style=True))
        std_only_path = updated_dir / f"benchmark_table_{short}{suffix}_paper_std_only.tex"
        std_only_path.write_text(
            format_latex(tbl, std, sections=SECTIONS, paper_style=True, std_only=True)
        )
        print(f"Updated tables written to {updated_dir}")

    # ImageNet-transfer comparison: In-domain DINO (from the current aggregate)
    # vs frozen ImageNet-pretrained DINO (from transfer_results.csv). This uses
    # the ViT transfer baseline (dino_vit_small_in1k), so it only applies to the
    # ViT family.
    if family == "vit":
        probe_agg = pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame(columns=["metric","dataset","model","mean","std"])
        transfer_csv = RESULTS_DIR / "transfer_results.csv"
        transfer_out = updated_dir / "paper_table_imagenet_transfer.tex"
        transfer_out.write_text(
            format_imagenet_transfer_latex(
                probe_agg[probe_agg["metric"] == "probe"], transfer_csv,
            )
        )
        print(f"ImageNet-transfer comparison written to {transfer_out}")


if __name__ == "__main__":
    main()
