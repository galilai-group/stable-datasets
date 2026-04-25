"""Pull per-run scalars from wandb.

Every training run produced by benchmarks/run.py logs, via the callbacks in
``benchmarks/models/__init__.py``:

* ``rankme`` / ``lidar`` — representation-geometry scalars (val-epoch).
* ``eval/linear_probe_top1_epoch`` / ``eval/knn_probe_top1`` — probe / KNN
  classification top-1 on the val split.

This script:
  1. Queries the wandb project for all runs under ``finalized-stable-datasets``.
  2. Parses ``(model, backbone, dataset, seed)`` out of the run ``config`` fields.
  3. For each run, takes the **final** step's scalars from ``run.summary``.
  4. Depending on ``--target``, merges into:
       * ``benchmarks/results/representation_scores.csv`` — adds
         ``rankme_wandb`` / ``lidar_wandb`` columns, deduped on (model, dataset).
       * ``benchmarks/results/benchmark_results.csv`` — refreshes the
         canonical accuracy table with columns (model, dataset, seed,
         is_seed_run, probe, knn), preserving per-seed rows.

Rows for (model, dataset) pairs without a wandb run keep NaN in the new columns.

Usage::

    python benchmarks/analysis/pull_wandb_metrics.py                 # both
    python benchmarks/analysis/pull_wandb_metrics.py --target representation_scores
    python benchmarks/analysis/pull_wandb_metrics.py --target benchmark_results
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from benchmarks.analysis.utils import BENCHMARK_RESULTS_CSV, REP_SCORES_CSV

WANDB_ENTITY = "samibg"
WANDB_PROJECT = "finalized-stable-datasets"

# Run names from benchmarks/run.py are {model}_{backbone}_{dataset}[_seed{N}].
# backbone is always vit_small in the current sweeps, so split on that.
_BACKBONE_TOKEN = "_vit_small_"


def _parse_run_name(name: str) -> tuple[str | None, str | None, int | None]:
    if name is None or _BACKBONE_TOKEN not in name:
        return None, None, None
    left, _, right = name.partition(_BACKBONE_TOKEN)
    model = left
    dataset = right
    seed = None
    if "_seed" in dataset:
        dataset, _, tail = dataset.rpartition("_seed")
        try:
            seed = int(tail)
        except ValueError:
            seed = None
    return model, dataset, seed


def _last_history_value(run, key: str) -> float | None:
    """Return the last non-null value of `key` logged during the run.

    Uses run.history(samples=N) rather than scan_history() so the call
    stays fast; 500 samples downsamples long runs uniformly, which is
    fine since we only want the final value.
    """
    try:
        df = run.history(keys=[key], samples=500, pandas=True)
    except Exception:
        return None
    if df is None or df.empty or key not in df.columns:
        return None
    s = df[key].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def fetch_runs(pull_history: bool = True) -> pd.DataFrame:
    """Pull per-run summary scalars from WandB.

    Parameters
    ----------
    pull_history
        If True, also fetch the last step's `rankme` / `lidar` from
        ``run.history()`` (those metrics aren't copied into
        ``run.summary`` by the callback). Set False for a fast probe/knn
        refresh.
    """
    import wandb

    api = wandb.Api(timeout=60)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    total = len(runs)
    print(f"wandb returned {total} runs; pull_history={pull_history}")
    rows = []
    for i, run in enumerate(runs, 1):
        cfg = run.config or {}
        model = cfg.get("model") or cfg.get("model/name")
        dataset = cfg.get("dataset")
        seed = cfg.get("seed")
        if not (model and dataset):
            m, d, s = _parse_run_name(run.name or "")
            model = model or m
            dataset = dataset or d
            seed = seed if seed is not None else s
        if model is None or dataset is None:
            continue
        summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
        rankme = summary.get("rankme")
        lidar = summary.get("lidar")
        if pull_history and rankme is None:
            rankme = _last_history_value(run, "rankme")
        if pull_history and lidar is None:
            lidar = _last_history_value(run, "lidar")
        rows.append({
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "wandb_run_id": run.id,
            "wandb_state": run.state,
            "rankme_wandb": rankme,
            "lidar_wandb": lidar,
            "probe": summary.get("eval/linear_probe_top1_epoch"),
            "knn": summary.get("eval/knn_probe_top1"),
        })
        if i % 25 == 0:
            print(f"  {i}/{total} runs processed")
    return pd.DataFrame(rows)


def merge_into_rep_scores(wandb_df: pd.DataFrame, csv_path: Path) -> None:
    """Merge rankme_wandb / lidar_wandb into representation_scores.csv keyed on (model, dataset).

    Dedups on (model, dataset) — takes the best run per pair, preferring
    finished runs with a non-null rankme.
    """
    wandb_df = wandb_df.copy()
    priority = wandb_df["rankme_wandb"].notna().astype(int) * 2 + wandb_df["wandb_state"].isin(["finished", "completed"]).astype(int)
    wandb_df = wandb_df.assign(_priority=priority)
    wandb_df = wandb_df.sort_values("_priority", ascending=False).drop_duplicates(["model", "dataset"], keep="first")
    wandb_df = wandb_df[["model", "dataset", "rankme_wandb", "lidar_wandb"]]

    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ("rankme_wandb", "lidar_wandb") if c in df.columns])
    df = df.merge(wandb_df, on=["model", "dataset"], how="left")
    df.to_csv(csv_path, index=False)

    n_with = df["rankme_wandb"].notna().sum()
    print(f"wrote {csv_path} ({len(df)} rows, {n_with} with rankme_wandb)")
    cov = df[df["rankme_wandb"].notna()].groupby("model").size()
    if len(cov):
        print("\nrankme_wandb coverage:")
        for m, n in cov.items():
            print(f"  {m:<16} {n}")


def merge_into_benchmark_results(wandb_df: pd.DataFrame, csv_path: Path) -> None:
    """Refresh benchmark_results.csv (model, dataset, seed, is_seed_run, probe, knn)
    from fresh wandb data.

    Keeps all finished runs that have a probe value — retries and reruns
    included. ssl_supervised_gap.py takes .max() per (model, dataset)
    downstream, so keeping duplicates is safe and ensures we don't silently
    drop the healthy run of a pair where a collapsed rerun exists
    (dino/arabiccharacters is the canonical example).

    The previous CSV is backed up with a timestamped suffix before overwrite.
    """
    wandb_df = wandb_df.copy()

    has_probe = wandb_df["probe"].notna()
    is_finished = wandb_df["wandb_state"].isin(["finished", "completed"])
    before = len(wandb_df)
    wandb_df = wandb_df[has_probe & is_finished]
    print(f"  filtered {before} -> {len(wandb_df)} (finished + has probe)")

    wandb_df["is_seed_run"] = wandb_df["seed"].notna()
    out = wandb_df[["model", "dataset", "seed", "is_seed_run", "probe", "knn"]].copy()
    out = out.sort_values(
        ["model", "dataset", "seed", "probe"],
        ascending=[True, True, True, False],
        na_position="first",
    ).reset_index(drop=True)

    if csv_path.exists():
        backup = csv_path.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}.csv")
        shutil.copy2(csv_path, backup)
        print(f"  backed up existing file -> {backup.name}")
    out.to_csv(csv_path, index=False)

    print(f"wrote {csv_path} ({len(out)} rows)")
    print("\nby model:")
    for m, n in out.groupby("model").size().items():
        print(f"  {m:<16} {n}")
    n_null_knn = out["knn"].isna().sum()
    if n_null_knn:
        print(f"\nnote: {n_null_knn} rows have null knn (run finished before knn callback logged)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--target",
        choices=["representation_scores", "benchmark_results", "both"],
        default="both",
        help="which CSV(s) to refresh (default: both)",
    )
    parser.add_argument("--rep-scores-csv", type=Path, default=REP_SCORES_CSV)
    parser.add_argument("--benchmark-results-csv", type=Path, default=BENCHMARK_RESULTS_CSV)
    parser.add_argument("--show", action="store_true", help="print fetched wandb rows and exit")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="skip the per-run run.history() fetch for rankme/lidar (faster, but "
             "those columns will be null for runs where the callback logged them "
             "as step metrics instead of summary values)",
    )
    args = parser.parse_args()

    wandb_df = fetch_runs(pull_history=not args.no_history)
    if wandb_df.empty:
        print("No runs returned from wandb.")
        return
    print(f"Fetched {len(wandb_df)} runs from {WANDB_ENTITY}/{WANDB_PROJECT}")
    if args.show:
        cols = ["model", "dataset", "seed", "wandb_state", "rankme_wandb", "lidar_wandb", "probe", "knn"]
        print(wandb_df[cols].to_string(index=False))
        return

    if args.target in ("representation_scores", "both"):
        print(f"\n=== merging into {args.rep_scores_csv.name} ===")
        merge_into_rep_scores(wandb_df, args.rep_scores_csv)
    if args.target in ("benchmark_results", "both"):
        print(f"\n=== writing {args.benchmark_results_csv.name} ===")
        merge_into_benchmark_results(wandb_df, args.benchmark_results_csv)


if __name__ == "__main__":
    main()
