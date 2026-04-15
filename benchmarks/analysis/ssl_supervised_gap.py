"""Where does SSL beat supervised, and what dataset metadata predicts it?

For each dataset in benchmark_results.csv, compute
    ssl_advantage = best_ssl_top1 - best_supervised_top1
(positive = SSL wins) and regress it on hand-curated dataset metadata plus
two derived features that proved to matter more than naive metadata:

  - images_per_class  = train_size / num_classes
  - supervised_ceiling = best supervised top-1 (headroom proxy)
  - supervised_normalized = (supervised - 1/K) / (1 - 1/K), fraction of
    learnable signal that supervised already captured.

Figures and a summary are written to benchmarks/results/.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RESULTS_CSV = REPO / "benchmarks" / "results" / "benchmark_results.csv"
OUT_DIR = REPO / "benchmarks" / "results"

SSL_METHODS = {"simclr", "barlow_twins", "nnclr", "dino", "lejepa"}
SUPERVISED = "supervised"

# Hand-curated per-dataset metadata.
#   natural       : photos of real-world scenes/objects (vs synthetic digits,
#                   characters, medical slices).
#   fine_grained  : sub-category classification within one super-class
#                   (birds, planes, food, flowers, textures, countries).
#   grayscale     : single-channel input.
#   centered      : objects are centered/cropped on clean background
#                   (mnist-style) — proxy for LOW intra-class variation.
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
    df = pd.read_csv(RESULTS_CSV)
    col = "eval/linear_probe_top1_epoch"
    df = df.dropna(subset=[col])
    best = df.groupby(["dataset", "model"])[col].max().reset_index()
    rows = []
    for ds, g in best.groupby("dataset"):
        sup = g.loc[g.model == SUPERVISED, col]
        ssl = g.loc[g.model.isin(SSL_METHODS)]
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
        rows.append(dict(features=f, r2=r2, loo_r2=loo_r2(x, y), spearman=float(rho), spearman_p=float(p)))
    for f1, f2 in itertools.combinations(FEATURES, 2):
        X = df[[f1, f2]].to_numpy()
        r2, _ = fit_r2(X, y)
        rows.append(dict(features=f"{f1} + {f2}", r2=r2, loo_r2=loo_r2(X, y), spearman=np.nan, spearman_p=np.nan))
    return pd.DataFrame(rows).sort_values("loo_r2", ascending=False).reset_index(drop=True)


def plot_advantage_bars(df: pd.DataFrame, path: Path) -> None:
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


def plot_scatters(df: pd.DataFrame, path: Path) -> None:
    cont = ["num_classes", "log_train_size", "log_images_per_class", "supervised_ceiling", "supervised_normalized"]
    fig, axes = plt.subplots(1, len(cont), figsize=(4 * len(cont), 4), sharey=True)
    y = df["ssl_advantage"]
    colors = ["#1f77b4" if a >= 0 else "#d62728" for a in y]
    for ax, f in zip(axes, cont):
        ax.scatter(df[f], y, c=colors)
        ax.axhline(0, color="k", lw=0.5)
        for _, r in df.iterrows():
            ax.annotate(r.dataset, (r[f], r.ssl_advantage), fontsize=6, alpha=0.7)
        r2, beta = fit_r2(df[f].to_numpy().reshape(-1, 1), y.to_numpy())
        xs = np.linspace(df[f].min(), df[f].max(), 50)
        ax.plot(xs, beta[0] + beta[1] * xs, color="gray", lw=1)
        ax.set_xlabel(f"{f}\nR²={r2:.2f}")
    axes[0].set_ylabel("ssl_advantage")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_headroom_money(df: pd.DataFrame, path: Path) -> None:
    """The single-feature story: ssl_advantage vs supervised_ceiling."""
    fig, ax = plt.subplots(figsize=(7, 6))
    x = df["supervised_ceiling"]
    y = df["ssl_advantage"]
    ax.scatter(x, y, c=["#1f77b4" if a >= 0 else "#d62728" for a in y], s=60)
    ax.axhline(0, color="k", lw=0.5)
    for _, r in df.iterrows():
        ax.annotate(r.dataset, (r.supervised_ceiling, r.ssl_advantage), fontsize=8, alpha=0.8)
    r2, beta = fit_r2(x.to_numpy().reshape(-1, 1), y.to_numpy())
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, beta[0] + beta[1] * xs, color="gray", lw=1)
    ax.set_xlabel(f"supervised top-1 (ceiling)   —   R²={r2:.2f}")
    ax.set_ylabel("best SSL − best supervised")
    ax.set_title("SSL beats supervised where supervised has room to be beaten")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    gaps = load_gaps()
    df = attach_metadata(gaps)
    ranked = rank_features(df)

    df.to_csv(OUT_DIR / "ssl_supervised_gap.csv", index=False)
    ranked.to_csv(OUT_DIR / "ssl_supervised_gap_feature_ranking.csv", index=False)
    plot_advantage_bars(df, OUT_DIR / "ssl_supervised_gap_bars.png")
    plot_scatters(df, OUT_DIR / "ssl_supervised_gap_scatter.png")
    plot_headroom_money(df, OUT_DIR / "ssl_advantage_vs_ceiling.png")

    print(f"n datasets = {len(df)}")
    print("\nPer-dataset SSL advantage (positive = SSL wins):")
    print(df[["dataset", "best_supervised", "best_ssl", "best_ssl_method", "ssl_advantage"]].to_string(index=False))
    print("\nTop 15 features by leave-one-out R²:")
    print(ranked.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
