"""Per-method correlations between normalized accuracy and RankMe/metadata.

This is an independent analysis entrypoint.  It intentionally avoids importing
the older ``benchmarks.analysis`` package.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


try:
    from .utils import (
        DATASET_ALIASES,
        DEFAULT_BACKBONE,
        DEFAULT_BACKBONES,
        DEFAULT_ENTITY,
        DEFAULT_PROJECT,
        SSL_METHODS,
        SUPERVISED_METHOD,
        collect_run_metrics,
        normalize_backbones,
        select_best_runs,
    )
except ImportError:  # pragma: no cover - supports direct ``python analysis.py`` use.
    from utils import (
        DATASET_ALIASES,
        DEFAULT_BACKBONE,
        DEFAULT_BACKBONES,
        DEFAULT_ENTITY,
        DEFAULT_PROJECT,
        SSL_METHODS,
        SUPERVISED_METHOD,
        collect_run_metrics,
        normalize_backbones,
        select_best_runs,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_METADATA_CSV = HERE / "dataset_metadata.csv"
DEFAULT_OUTPUT_CSV = HERE / "analysis_results.csv"


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "dataset" not in meta.columns or "num_classes" not in meta.columns:
        raise ValueError(f"{path} must contain 'dataset' and 'num_classes' columns")
    meta = meta.copy()
    meta["dataset"] = meta["dataset"].str.lower()
    meta["dataset"] = meta["dataset"].replace(DATASET_ALIASES)
    meta["num_classes"] = pd.to_numeric(meta["num_classes"], errors="coerce")
    if "mean_pixels" in meta.columns:
        meta["height_times_width"] = pd.to_numeric(meta["mean_pixels"], errors="coerce")
    elif {"mean_height", "mean_width"}.issubset(meta.columns):
        meta["height_times_width"] = pd.to_numeric(meta["mean_height"], errors="coerce") * pd.to_numeric(
            meta["mean_width"], errors="coerce"
        )
    meta["chance"] = 1.0 / meta["num_classes"]
    return meta


def metadata_metric_columns(meta: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    if requested is not None:
        cols = list(requested)
        missing = [c for c in cols if c not in meta.columns]
        if missing:
            raise ValueError(f"metadata columns not found: {missing}")
        return cols
    skip = {"dataset", "chance", "mean_pixels"}
    return [c for c in meta.columns if c not in skip and pd.api.types.is_numeric_dtype(meta[c])]


def _spearman(x: pd.Series, y: pd.Series) -> dict:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3:
        return {"n": int(len(df)), "spearman_rho": np.nan, "spearman_p": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(df["x"], df["y"])
    return {"n": int(len(df)), "spearman_rho": float(rho), "spearman_p": float(p)}


def build_analysis_frame(best_runs: pd.DataFrame, meta: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    """Attach chance, supervised baseline, and normalized LHS targets."""
    if best_runs.empty:
        return pd.DataFrame()

    df = best_runs.merge(meta, on="dataset", how="inner", suffixes=("", "_metadata"))
    sup = (
        df[df["model"] == SUPERVISED_METHOD][["dataset", "probe", "rankme_val"]]
        .rename(
            columns={
                "probe": "probe_sup",
                "rankme_val": "rankme_val_sup",
            }
        )
        .drop_duplicates("dataset")
    )
    df = df.merge(sup, on="dataset", how="left")
    df = df[df["model"].isin(set(methods))].copy()

    df["z"] = (df["probe"] - df["chance"]) / (1.0 - df["chance"])
    supervised_headroom = df["probe_sup"] - df["chance"]
    df["r"] = np.where(supervised_headroom > 0, (df["probe"] - df["chance"]) / supervised_headroom, np.nan)
    df["rel_rankme_val"] = df["rankme_val"] / df["rankme_val_sup"]
    return df


def correlation_rows(
    analysis_df: pd.DataFrame,
    *,
    methods: Iterable[str],
    metadata_columns: Iterable[str],
) -> list[dict]:
    rows: list[dict] = []
    lhs_columns = ("z", "r")

    for method in methods:
        method_df = analysis_df[analysis_df["model"] == method]
        if method_df.empty:
            continue
        for lhs in lhs_columns:
            for metric in metadata_columns:
                rows.append(
                    {
                        "method": method,
                        "lhs": lhs,
                        "rhs_family": "dataset_metadata",
                        "rhs_name": metric,
                        "split": "dataset",
                        **_spearman(method_df[metric], method_df[lhs]),
                    }
                )
            rows.append(
                {
                    "method": method,
                    "lhs": lhs,
                    "rhs_family": "rankme",
                    "rhs_name": "rankme",
                    "split": "val",
                    **_spearman(method_df["rankme_val"], method_df[lhs]),
                }
            )
            rows.append(
                {
                    "method": method,
                    "lhs": lhs,
                    "rhs_family": "relative_rankme",
                    "rhs_name": "rel_rankme",
                    "split": "val",
                    **_spearman(method_df["rel_rankme_val"], method_df[lhs]),
                }
            )
    return rows


def run_analysis(
    *,
    metadata_csv: Path = DEFAULT_METADATA_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    selected_runs_csv: Path | None = None,
    write_selected_runs_csv: Path | None = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    backbone: str | None = None,
    backbones: Iterable[str] | str | None = None,
    methods: Iterable[str] = SSL_METHODS,
    datasets: Iterable[str] | None = None,
    metadata_metrics: Iterable[str] | None = None,
    history_samples: int = 0,
) -> pd.DataFrame:
    meta = load_metadata(metadata_csv)
    metric_cols = metadata_metric_columns(meta, metadata_metrics)
    allowed_backbones = normalize_backbones(backbones, backbone=backbone)

    if selected_runs_csv is not None:
        best_runs = pd.read_csv(selected_runs_csv)
    else:
        fetch_methods = tuple(dict.fromkeys([*methods, SUPERVISED_METHOD]))
        fetched = collect_run_metrics(
            entity,
            project,
            backbones=allowed_backbones,
            methods=fetch_methods,
            datasets=datasets,
            history_samples=history_samples,
        )
        best_runs = select_best_runs(fetched)

    if write_selected_runs_csv is not None:
        write_selected_runs_csv.parent.mkdir(parents=True, exist_ok=True)
        best_runs.to_csv(write_selected_runs_csv, index=False)

    analysis_df = build_analysis_frame(best_runs, meta, methods)
    rows = correlation_rows(analysis_df, methods=methods, metadata_columns=metric_cols)
    out = pd.DataFrame(rows)
    columns = ["method", "lhs", "rhs_family", "rhs_name", "split", "n", "spearman_rho", "spearman_p"]
    if out.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = out[columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--backbones",
        default=",".join(DEFAULT_BACKBONES),
        help="comma-separated allowed backbones; best probe run is selected across them",
    )
    parser.add_argument("--backbone", default=None, help=f"single-backbone shortcut, e.g. {DEFAULT_BACKBONE}")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--selected-runs-csv", type=Path, default=None, help="reuse preselected W&B run metrics")
    parser.add_argument("--write-selected-runs-csv", type=Path, default=None)
    parser.add_argument("--methods", default=",".join(SSL_METHODS))
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--metadata-metrics", default=None)
    parser.add_argument(
        "--history-samples",
        type=int,
        default=0,
        help="samples to use when falling back from W&B summary to history; 0 disables fallback",
    )
    args = parser.parse_args()

    methods = tuple(_parse_csv_list(args.methods) or SSL_METHODS)
    backbones = tuple(_parse_csv_list(args.backbone or args.backbones) or DEFAULT_BACKBONES)
    datasets = _parse_csv_list(args.datasets)
    metadata_metrics = _parse_csv_list(args.metadata_metrics)

    out = run_analysis(
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
        selected_runs_csv=args.selected_runs_csv,
        write_selected_runs_csv=args.write_selected_runs_csv,
        entity=args.entity,
        project=args.project,
        backbones=backbones,
        methods=methods,
        datasets=datasets,
        metadata_metrics=metadata_metrics,
        history_samples=args.history_samples,
    )
    print(f"wrote {args.output_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
