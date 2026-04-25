"""End-to-end correlation pipeline.

Merges what used to be two scripts:

  * ssl_supervised_gap.py — compute ``ssl_advantage = max_ssl_top1 −
    sup_top1`` per dataset, attach hand-curated dataset metadata, rank
    metadata features by R² / leave-one-out R² / Spearman, render plots.
    Output: ``ssl_supervised_gap.csv``, ``ssl_supervised_gap_feature_ranking.csv``,
    plus PNGs under ``benchmarks/results/``.

  * correlation_report.py — Spearman / Pearson of RankMe and LiDAR with
    ``ssl_advantage`` across four framings (delta, ssl, sup, per-method)
    × two splits (val, train). Output: ``correlation_report.csv``.

CLI::

    python -m benchmarks.analysis.correlations              # both stages
    python -m benchmarks.analysis.correlations --stage gap
    python -m benchmarks.analysis.correlations --stage correlate

Stage 2 (``correlate``) requires Stage 1's ``ssl_supervised_gap.csv`` to
exist; ``--stage both`` produces it on the fly.
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.analysis.utils import (
    BENCHMARK_RESULTS_CSV,
    GAP_CSV,
    GAP_RANKING_CSV,
    HISTORY_PATHS,
    OUT_DIR,
    REPORT_CSV,
    SSL_METHODS,
    load_history,
)


# Stage 1: gap + dataset metadata
# -----------------------------------------------------------------------

# MAE is reconstruction-only and excluded from the "best SSL" max; treat it as
# a separate regime in plots / per-method analyses.
GAP_SSL_METHODS = {"simclr", "barlow_twins", "nnclr", "dino", "lejepa"}
SUPERVISED = "supervised"

# Hand-curated per-dataset metadata.
#   natural       : photos of real-world scenes/objects (vs synthetic digits,
#                   characters, medical slices).
#   fine_grained  : sub-category classification within one super-class.
#   grayscale     : single-channel input.
#   centered      : objects are centered/cropped on clean background — proxy
#                   for LOW intra-class variation.
METADATA: dict[str, dict] = {
    "arabiccharacters": dict(num_classes=28,  train_size=13440, natural=0, fine_grained=0, grayscale=1, centered=1),
    "arabicdigits":     dict(num_classes=10,  train_size=60000, natural=0, fine_grained=0, grayscale=1, centered=1),
    "beans":            dict(num_classes=3,   train_size=1034,  natural=1, fine_grained=1, grayscale=0, centered=1),
    "cifar10":          dict(num_classes=10,  train_size=50000, natural=1, fine_grained=0, grayscale=0, centered=0),
    "cifar100":         dict(num_classes=100, train_size=50000, natural=1, fine_grained=0, grayscale=0, centered=0),
    "country211":       dict(num_classes=211, train_size=31650, natural=1, fine_grained=1, grayscale=0, centered=0),
    "cub200":           dict(num_classes=200, train_size=5994,  natural=1, fine_grained=1, grayscale=0, centered=0),
    "dtd":              dict(num_classes=47,  train_size=1880,  natural=1, fine_grained=0, grayscale=0, centered=0),
    "emnist":           dict(num_classes=47,  train_size=112800,natural=0, fine_grained=0, grayscale=1, centered=1),
    "fashionmnist":     dict(num_classes=10,  train_size=60000, natural=0, fine_grained=0, grayscale=1, centered=1),
    "fgvcaircraft":     dict(num_classes=100, train_size=6667,  natural=1, fine_grained=1, grayscale=0, centered=0),
    "flowers102":       dict(num_classes=102, train_size=1020,  natural=1, fine_grained=1, grayscale=0, centered=0),
    "food101":          dict(num_classes=101, train_size=75750, natural=1, fine_grained=1, grayscale=0, centered=0),
    "imagenette":       dict(num_classes=10,  train_size=9469,  natural=1, fine_grained=0, grayscale=0, centered=0),
    "medmnist":         dict(num_classes=9,   train_size=89996, natural=0, fine_grained=0, grayscale=0, centered=1),
    "notmnist":         dict(num_classes=10,  train_size=18724, natural=0, fine_grained=0, grayscale=1, centered=1),
    "rockpaperscissor": dict(num_classes=3,   train_size=2520,  natural=1, fine_grained=0, grayscale=0, centered=1),
    "stl10":            dict(num_classes=10,  train_size=5000,  natural=1, fine_grained=0, grayscale=0, centered=0),
    "svhn":             dict(num_classes=10,  train_size=73257, natural=1, fine_grained=0, grayscale=0, centered=1),
}

FEATURES = [
    "num_classes",
    "log_train_size",
    "log_images_per_class",
    "supervised_ceiling",
    "supervised_normalized",
    "natural",
    "fine_grained",
    "grayscale",
    "centered",
]


def load_gaps() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_RESULTS_CSV)
    col = "probe"  # benchmark_results.csv column, renamed from the raw wandb key
    df = df.dropna(subset=[col])
    best = df.groupby(["dataset", "model"])[col].max().reset_index()
    rows = []
    for ds, g in best.groupby("dataset"):
        sup = g.loc[g.model == SUPERVISED, col]
        ssl = g.loc[g.model.isin(GAP_SSL_METHODS)]
        if sup.empty or ssl.empty:
            continue
        best_ssl_row = ssl.loc[ssl[col].idxmax()]
        rows.append(
            dict(
                dataset=ds,
                best_supervised=float(sup.max()),
                best_ssl=float(best_ssl_row[col]),
                best_ssl_method=best_ssl_row["model"],
                ssl_advantage=float(best_ssl_row[col]) - float(sup.max()),
            )
        )
    return pd.DataFrame(rows).sort_values("ssl_advantage", ascending=False).reset_index(drop=True)


def attach_metadata(gaps: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame([METADATA.get(ds, {}) for ds in gaps.dataset])
    out = pd.concat([gaps.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    out = out.dropna(subset=["num_classes"]).reset_index(drop=True)
    out["log_train_size"] = np.log10(out["train_size"])
    out["images_per_class"] = out["train_size"] / out["num_classes"]
    out["log_images_per_class"] = np.log10(out["images_per_class"])
    out["supervised_ceiling"] = out["best_supervised"]
    chance = 1.0 / out["num_classes"]
    out["supervised_normalized"] = (out["best_supervised"] - chance) / (1 - chance)
    return out


def fit_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, beta


def loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    preds = np.zeros_like(y)
    for i in range(len(y)):
        mask = np.ones(len(y), bool)
        mask[i] = False
        Xt = np.column_stack([np.ones(mask.sum()), X[mask]])
        beta, *_ = np.linalg.lstsq(Xt, y[mask], rcond=None)
        x1 = np.concatenate([[1.0], X[i]])
        preds[i] = x1 @ beta
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df["ssl_advantage"].to_numpy()
    rows = []
    for f in FEATURES:
        x = df[f].to_numpy().reshape(-1, 1)
        r2, _ = fit_r2(x, y)
        rho, p = stats.spearmanr(df[f], y)
        rows.append(dict(features=f, r2=r2, loo_r2=loo_r2(x, y),
                         spearman=float(rho), spearman_p=float(p)))
    for f1, f2 in itertools.combinations(FEATURES, 2):
        X = df[[f1, f2]].to_numpy()
        r2, _ = fit_r2(X, y)
        rows.append(dict(features=f"{f1} + {f2}", r2=r2, loo_r2=loo_r2(X, y),
                         spearman=np.nan, spearman_p=np.nan))
    return pd.DataFrame(rows).sort_values("loo_r2", ascending=False).reset_index(drop=True)


def _plot_advantage_bars(df, path):
    d = df.sort_values("ssl_advantage")
    colors = ["#1f77b4" if a >= 0 else "#d62728" for a in d["ssl_advantage"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["dataset"], d["ssl_advantage"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("best SSL − best supervised (top-1)")
    ax.set_title("SSL advantage by dataset (blue = SSL wins)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_headroom(df, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    x = df["supervised_ceiling"]; y = df["ssl_advantage"]
    ax.scatter(x, y, c=["#1f77b4" if a >= 0 else "#d62728" for a in y], s=60)
    ax.axhline(0, color="k", lw=0.5)
    for _, r in df.iterrows():
        ax.annotate(r.dataset, (r.supervised_ceiling, r.ssl_advantage), fontsize=8, alpha=0.8)
    r2, beta = fit_r2(x.to_numpy().reshape(-1, 1), y.to_numpy())
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, beta[0] + beta[1] * xs, color="gray", lw=1)
    ax.set_xlabel(f"supervised top-1 (ceiling)   —   R²={r2:.2f}")
    ax.set_ylabel("best SSL − best supervised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_gap_stage() -> pd.DataFrame:
    """Compute ssl_advantage + metadata + feature ranking. Writes 2 CSVs and 2 PNGs."""
    gaps = load_gaps()
    df = attach_metadata(gaps)
    ranked = rank_features(df)

    df.to_csv(GAP_CSV, index=False)
    ranked.to_csv(GAP_RANKING_CSV, index=False)
    _plot_advantage_bars(df, OUT_DIR / "ssl_supervised_gap_bars.png")
    _plot_headroom(df, OUT_DIR / "ssl_advantage_vs_ceiling.png")

    print(f"[gap] n datasets = {len(df)}  ->  {GAP_CSV.name}")
    print("Per-dataset SSL advantage (positive = SSL wins):")
    print(df[["dataset", "best_supervised", "best_ssl", "best_ssl_method", "ssl_advantage"]]
          .to_string(index=False))
    print("\nTop 10 features by leave-one-out R²:")
    print(ranked.head(10).to_string(index=False))
    return df


# Stage 2: rep-score correlations
# -----------------------------------------------------------------------


def _spearman_pearson(x: pd.Series, y: pd.Series) -> dict:
    # Align by index first — scipy.stats doesn't, so positional mismatch would
    # silently give wrong rhos if the two Series are indexed in different orders.
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return dict(n=int(len(df)), spearman_rho=None, spearman_p=None,
                    pearson_r=None, pearson_p=None)
    rho, p = stats.spearmanr(df["x"], df["y"])
    r, pr = stats.pearsonr(df["x"], df["y"])
    return dict(n=int(len(df)),
                spearman_rho=float(rho), spearman_p=float(p),
                pearson_r=float(r), pearson_p=float(pr))


def _collect_rows_for_split(hist: pd.DataFrame, gap: pd.DataFrame, split: str) -> list[dict]:
    out: list[dict] = []
    sup = hist[hist.model == "supervised"].set_index("dataset")

    # pick, per dataset, the rep-score row of the probe-best SSL method
    # (gap.best_ssl_method); fall back to any SSL row we have.
    best_ssl_rows = []
    for _, g in gap.iterrows():
        ds = g["dataset"]
        ssl_rows = hist[(hist.dataset == ds) & hist.model.isin(SSL_METHODS)]
        if ssl_rows.empty:
            continue
        picked = ssl_rows[ssl_rows.model == g.get("best_ssl_method")]
        if picked.empty:
            picked = ssl_rows
        best_ssl_rows.append(picked.iloc[0].to_dict())
    best_ssl = pd.DataFrame(best_ssl_rows).set_index("dataset") if best_ssl_rows else pd.DataFrame()

    gap_idx = gap.set_index("dataset")["ssl_advantage"]

    for metric in ("rankme", "lidar"):
        if not best_ssl.empty:
            out.append(dict(framing="delta", metric=metric, split=split, method="best_ssl",
                            **_spearman_pearson(best_ssl[metric] - sup[metric], gap_idx)))
            out.append(dict(framing="ssl", metric=metric, split=split, method="best_ssl",
                            **_spearman_pearson(best_ssl[metric], gap_idx)))
        out.append(dict(framing="sup", metric=metric, split=split, method="supervised",
                        **_spearman_pearson(sup[metric], gap_idx)))
        for method in SSL_METHODS:
            m_rows = hist[hist.model == method].set_index("dataset")
            if m_rows.empty:
                continue
            out.append(dict(framing="method", metric=metric, split=split, method=method,
                            **_spearman_pearson(m_rows[metric], gap_idx)))
    return out


def run_correlation_stage(gap: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute Spearman/Pearson of rep-scores against ssl_advantage. Writes correlation_report.csv."""
    if gap is None:
        if not GAP_CSV.exists():
            raise SystemExit(f"{GAP_CSV} missing — run with --stage both or --stage gap first")
        gap = pd.read_csv(GAP_CSV)

    val_hist = load_history(HISTORY_PATHS["val"], "val")
    train_hist = load_history(HISTORY_PATHS["train"], "train")

    print(f"[correlate] val: {len(val_hist)} rows ({val_hist.dataset.nunique()} datasets), "
          f"train: {len(train_hist)} rows ({train_hist.dataset.nunique()} datasets)")

    rows: list[dict] = []
    if len(val_hist) > 0:
        rows.extend(_collect_rows_for_split(val_hist, gap, "val"))
    if len(train_hist) > 0:
        rows.extend(_collect_rows_for_split(train_hist, gap, "train"))

    df = pd.DataFrame(rows)[["framing", "metric", "split", "method", "n",
                              "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]]
    df.to_csv(REPORT_CSV, index=False)
    print(f"[correlate] wrote {REPORT_CSV} ({len(df)} rows)")

    print("\nTop 15 by |Spearman ρ|:")
    printable = df.dropna(subset=["spearman_rho"]).copy()
    printable["abs_rho"] = printable["spearman_rho"].abs()
    for _, r in printable.sort_values("abs_rho", ascending=False).head(15).iterrows():
        print(f"  {r['split']:<5}  {r['metric']:<6}  {r['framing']:<6}  {r['method']:<14}"
              f"  n={int(r['n']):3d}  ρ={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}")
    return df


# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["gap", "correlate", "both"], default="both",
                        help="which pipeline stage(s) to run (default: both)")
    args = parser.parse_args()

    gap = None
    if args.stage in ("gap", "both"):
        gap = run_gap_stage()
    if args.stage in ("correlate", "both"):
        run_correlation_stage(gap)


if __name__ == "__main__":
    main()
