"""Analyze the gap between best SSL and best supervised methods per dataset.

Reads existing data files directly:
  - offline_probe_results.csv  (SSL linear probe results)
  - supervised.json            (supervised baselines)
  - conf/model/*.yaml          (pretraining epochs per method x dataset)

Outputs comprehensive tables and scatter plots for iterative exploration.

Usage:
    python analyze_gap.py [--output-dir figures/]
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONF_DIR = Path(__file__).parent / "stable_datasets" / "benchmarks" / "self_supervised" / "conf" / "model"

SSL_METHODS = ["dino", "lejepa", "mae", "nnclr", "simclr"]

# ── Dataset metadata ────────────────────────────────────────────────
# Hardcoded for the 10 benchmark datasets present in SSL results.
DATASET_META = pd.DataFrame([
    {"dataset": "arabiccharacters", "num_classes": 28,  "train_size": 13440, "balanced": True,  "grayscale": True,  "fine_grained": False},
    {"dataset": "arabicdigits",     "num_classes": 10,  "train_size": 60000, "balanced": True,  "grayscale": True,  "fine_grained": False},
    {"dataset": "cifar10",          "num_classes": 10,  "train_size": 50000, "balanced": True,  "grayscale": False, "fine_grained": False},
    {"dataset": "country211",       "num_classes": 211, "train_size": 31650, "balanced": True,  "grayscale": False, "fine_grained": True},
    {"dataset": "cub200",           "num_classes": 200, "train_size": 5994,  "balanced": False, "grayscale": False, "fine_grained": True},
    {"dataset": "dtd",              "num_classes": 47,  "train_size": 1880,  "balanced": True,  "grayscale": False, "fine_grained": True},
    {"dataset": "flowers102",       "num_classes": 102, "train_size": 1020,  "balanced": False, "grayscale": False, "fine_grained": True},
    {"dataset": "medmnist",         "num_classes": 2,   "train_size": 4708,  "balanced": False, "grayscale": True,  "fine_grained": False},
    {"dataset": "notmnist",         "num_classes": 10,  "train_size": 60000, "balanced": True,  "grayscale": True,  "fine_grained": False},
    {"dataset": "rockpaperscissor", "num_classes": 3,   "train_size": 2520,  "balanced": True,  "grayscale": False, "fine_grained": False},
]).set_index("dataset")


# =====================================================================
# Section 1: Load & Assemble
# =====================================================================

def load_ssl_results() -> pd.DataFrame:
    """Load SSL linear probe results, keeping best run per (model, dataset)."""
    df = pd.read_csv("offline_probe_results.csv")
    # Keep best top-1 per (model, dataset) in case of duplicate runs
    best = df.loc[df.groupby(["model", "dataset"])["eval/top1"].idxmax()]
    return best[["model", "dataset", "backbone", "eval/top1"]].rename(
        columns={"eval/top1": "accuracy", "model": "method"}
    ).reset_index(drop=True)


def load_supervised_results() -> pd.DataFrame:
    """Load supervised results. For multi-config datasets, keep only the matching config."""
    with open("supervised.json") as f:
        data = json.load(f)

    rows = []
    for model_name, datasets in data.items():
        for ds_name, ds_info in datasets.items():
            for entry in ds_info["entries"]:
                config = entry["hyperparams"].get("config_name")
                rows.append({
                    "backbone": model_name,
                    "dataset": ds_name,
                    "config_name": config,
                    "accuracy": entry["test_accuracy"],
                    "sup_epochs": entry["hyperparams"].get("max_epochs", 100),
                })

    df = pd.DataFrame(rows)
    # For medmnist, SSL uses pneumoniamnist; for emnist, skip (not in SSL)
    df = df[
        (df["config_name"].isna())
        | (df["config_name"] == "pneumoniamnist")
    ].copy()
    # Drop emnist — not in SSL benchmark datasets
    df = df[df["dataset"] != "emnist"]
    return df


def _parse_yaml_epochs(path: Path) -> dict:
    """Parse max_epochs from a YAML config without importing yaml.

    Returns {dataset_name: max_epochs, ...} including 'default'.
    """
    epochs = {}
    current_key = None
    in_params = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "params:":
            in_params = True
            continue
        if in_params:
            # New top-level key exits params block
            if stripped and not stripped.startswith("#") and not line.startswith(" ") and not line.startswith("\t"):
                in_params = False
                continue
            # Dataset key (indented once, ends with colon)
            if stripped.endswith(":") and "max_epochs" not in stripped:
                current_key = stripped.rstrip(":")
            elif "max_epochs:" in stripped and current_key is not None:
                val = stripped.split("max_epochs:")[-1].strip()
                epochs[current_key] = int(val)
    return epochs


def load_pretrain_epochs() -> pd.DataFrame:
    """Load pretraining epochs from conf/model/*.yaml files."""
    rows = []
    for method in SSL_METHODS:
        yaml_path = CONF_DIR / f"{method}.yaml"
        if not yaml_path.exists():
            continue
        epochs_map = _parse_yaml_epochs(yaml_path)
        default_epochs = epochs_map.get("default", 50)
        for ds in DATASET_META.index:
            ep = epochs_map.get(ds, default_epochs)
            rows.append({"method": method, "dataset": ds, "pretrain_epochs": ep})
    return pd.DataFrame(rows)


# =====================================================================
# Section 2: Full Results Tables
# =====================================================================

def print_ssl_table(ssl_df: pd.DataFrame) -> None:
    """Print all SSL results per method per dataset."""
    print("\n" + "=" * 90)
    print("SECTION 2a: ALL SSL LINEAR PROBE RESULTS (best run per method×dataset)")
    print("=" * 90)

    pivot = ssl_df.pivot_table(
        index="dataset", columns="method", values="accuracy", aggfunc="max"
    )
    # Add best column
    pivot["BEST"] = pivot.max(axis=1)
    pivot["best_method"] = pivot.drop(columns="BEST").idxmax(axis=1)

    # Format as percentages
    fmt = pivot.drop(columns="best_method").map(lambda x: f"{x * 100:.1f}" if pd.notna(x) else "---")
    fmt["best_method"] = pivot["best_method"]
    print(fmt.to_string())
    print()


def print_supervised_table(sup_df: pd.DataFrame) -> None:
    """Print all supervised results per backbone per dataset."""
    print("\n" + "=" * 90)
    print("SECTION 2b: ALL SUPERVISED RESULTS (per backbone×dataset)")
    print("=" * 90)

    pivot = sup_df.pivot_table(
        index="dataset", columns="backbone", values="accuracy", aggfunc="max"
    )
    fmt = pivot.map(lambda x: f"{x * 100:.1f}" if pd.notna(x) else "---")
    print(fmt.to_string())
    print()


def print_comparison_table(ssl_df: pd.DataFrame, sup_df: pd.DataFrame) -> None:
    """Side-by-side comparison with gap for each supervised backbone."""
    print("\n" + "=" * 90)
    print("SECTION 2c: SIDE-BY-SIDE COMPARISON (best SSL vs each supervised backbone)")
    print("=" * 90)

    ssl_best = ssl_df.groupby("dataset")["accuracy"].max().rename("ssl_best")
    ssl_method = ssl_df.loc[ssl_df.groupby("dataset")["accuracy"].idxmax()].set_index("dataset")["method"].rename("ssl_method")

    sup_backbones = sorted(sup_df["backbone"].unique())
    header_parts = [f"{'Dataset':20s}", f"{'SSL Best':>9s}", f"{'Method':>8s}"]
    for bb in sup_backbones:
        short = bb.replace("vit-base-patch16-224", "vit-base").replace("resnet-50", "rn50")
        header_parts.extend([f"{short:>9s}", f"{'gap':>7s}"])
    print("  ".join(header_parts))
    print("-" * (25 + 19 + len(sup_backbones) * 18))

    datasets = sorted(set(ssl_best.index) | set(sup_df["dataset"]))
    for ds in datasets:
        ssl_val = ssl_best.get(ds, np.nan)
        ssl_m = ssl_method.get(ds, "---")
        parts = [f"{ds:20s}", f"{ssl_val * 100:8.1f}%" if pd.notna(ssl_val) else f"{'---':>9s}", f"{ssl_m:>8s}"]
        for bb in sup_backbones:
            sup_val = sup_df[(sup_df["dataset"] == ds) & (sup_df["backbone"] == bb)]["accuracy"].max()
            if pd.isna(sup_val):
                parts.extend([f"{'---':>9s}", f"{'':>7s}"])
            else:
                gap = sup_val - ssl_val if pd.notna(ssl_val) else np.nan
                parts.append(f"{sup_val * 100:8.1f}%")
                parts.append(f"{gap * 100:+6.1f}%" if pd.notna(gap) else f"{'':>7s}")
        print("  ".join(parts))
    print()


# =====================================================================
# Section 3: Dataset Metadata
# =====================================================================

def print_metadata() -> None:
    print("\n" + "=" * 90)
    print("SECTION 3: DATASET METADATA")
    print("=" * 90)
    meta = DATASET_META.copy()
    meta["samples_per_class"] = meta["train_size"] / meta["num_classes"]
    meta["log_num_classes"] = np.log2(meta["num_classes"]).round(2)
    meta["log_train_size"] = np.log10(meta["train_size"]).round(2)
    print(meta.to_string())
    print()


# =====================================================================
# Section 4: Confounds Display
# =====================================================================

def print_confounds(ssl_df: pd.DataFrame, sup_df: pd.DataFrame, epochs_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("SECTION 4: POTENTIAL CONFOUNDS (pretraining compute, backbone mismatch)")
    print("=" * 90)

    ssl_best_idx = ssl_df.groupby("dataset")["accuracy"].idxmax()
    ssl_best = ssl_df.loc[ssl_best_idx].set_index("dataset")

    header = f"{'Dataset':20s}  {'SSL method':>8s}  {'SSL bb':>10s}  {'PT epochs':>9s}  {'train_sz':>8s}  {'PT compute':>10s}  {'Sup bb(s)':>22s}  {'Sup epochs':>10s}  {'Note':s}"
    print(header)
    print("-" * len(header))

    for ds in sorted(DATASET_META.index):
        if ds not in ssl_best.index:
            continue
        row = ssl_best.loc[ds]
        method = row["method"]
        backbone = row["backbone"]
        train_size = DATASET_META.loc[ds, "train_size"]

        # Get pretrain epochs for this method+dataset
        ep_row = epochs_df[(epochs_df["method"] == method) & (epochs_df["dataset"] == ds)]
        pt_epochs = int(ep_row["pretrain_epochs"].iloc[0]) if len(ep_row) > 0 else "?"
        pt_compute = f"{pt_epochs * train_size:,}" if isinstance(pt_epochs, int) else "?"

        # Supervised info
        sup_for_ds = sup_df[sup_df["dataset"] == ds]
        if len(sup_for_ds) > 0:
            sup_bbs = ", ".join(sorted(sup_for_ds["backbone"].unique()))
            sup_ep = str(int(sup_for_ds["sup_epochs"].iloc[0]))
            # Note confounds
            notes = []
            if backbone != "resnet-50" and "resnet-50" in sup_bbs:
                notes.append("backbone mismatch")
            if isinstance(pt_epochs, int) and pt_epochs != int(sup_for_ds["sup_epochs"].iloc[0]):
                notes.append(f"epoch mismatch ({pt_epochs} vs {sup_ep})")
            note = "; ".join(notes) if notes else "comparable"
        else:
            sup_bbs = "---"
            sup_ep = "---"
            note = "NO SUPERVISED BASELINE"

        print(f"{ds:20s}  {method:>8s}  {backbone:>10s}  {str(pt_epochs):>9s}  {train_size:>8d}  {pt_compute:>10s}  {sup_bbs:>22s}  {sup_ep:>10s}  {note}")
    print()


# =====================================================================
# Section 5: Gap Analysis
# =====================================================================

def gap_analysis(ssl_df: pd.DataFrame, sup_df: pd.DataFrame):
    """Compute gap for each supervised backbone, run regressions."""
    print("\n" + "=" * 90)
    print("SECTION 5: GAP ANALYSIS")
    print("=" * 90)

    ssl_best = ssl_df.groupby("dataset")["accuracy"].max()
    sup_backbones = sorted(sup_df["backbone"].unique())

    meta = DATASET_META.copy()
    meta["samples_per_class"] = meta["train_size"] / meta["num_classes"]
    meta["log_num_classes"] = np.log2(meta["num_classes"])
    meta["log_train_size"] = np.log10(meta["train_size"])

    numeric_features = [
        "num_classes", "train_size", "samples_per_class",
        "log_num_classes", "log_train_size",
    ]
    bool_features = ["balanced", "grayscale", "fine_grained"]
    all_features = numeric_features + bool_features

    results_by_backbone = {}

    for bb in sup_backbones:
        bb_short = bb.replace("vit-base-patch16-224", "vit-base").replace("resnet-50", "rn50")
        sup_bb = sup_df[sup_df["backbone"] == bb].groupby("dataset")["accuracy"].max()

        # Intersect datasets
        common = sorted(set(ssl_best.index) & set(sup_bb.index) & set(meta.index))
        if len(common) < 3:
            print(f"\n  [{bb_short}] Only {len(common)} overlapping datasets — skipping regression.")
            continue

        gap = pd.Series({ds: sup_bb[ds] - ssl_best[ds] for ds in common}, name="gap")
        df = meta.loc[common].copy()
        df["gap"] = gap

        print(f"\n{'─' * 70}")
        print(f"  Supervised backbone: {bb}  ({bb_short})")
        print(f"  Overlapping datasets: {len(common)}  {common}")
        print(f"{'─' * 70}")

        # Print gap values
        print(f"\n  {'Dataset':20s}  {'SSL':>8s}  {bb_short:>8s}  {'Gap':>8s}")
        for ds in common:
            print(f"  {ds:20s}  {ssl_best[ds]*100:7.1f}%  {sup_bb[ds]*100:7.1f}%  {gap[ds]*100:+7.1f}%")

        # Univariate correlations
        print(f"\n  --- Univariate correlations (Pearson r, p-value, R^2) ---")
        print(f"  {'Feature':30s}  {'r':>8s}  {'p':>8s}  {'R2':>8s}")
        y = df["gap"].values
        n = len(y)

        univar_results = []
        for feat in all_features:
            x = df[feat].astype(float).values
            if np.std(x) == 0:
                print(f"  {feat:30s}  {'no var':>8s}")
                continue
            r, p = stats.pearsonr(x, y)
            r2 = r ** 2
            print(f"  {feat:30s}  {r:8.4f}  {p:8.4f}  {r2:8.4f}")
            univar_results.append((feat, r, p, r2))

        # Best 1-feature OLS
        print(f"\n  --- Best single-feature OLS models ---")
        single_models = []
        for feat in all_features:
            x = df[feat].astype(float).values
            if np.std(x) == 0:
                continue
            X = np.column_stack([np.ones(n), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # LOO-CV R^2
            loo_preds = np.zeros(n)
            for i in range(n):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                b = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                loo_preds[i] = X[i] @ b
            ss_res_loo = np.sum((y - loo_preds) ** 2)
            r2_loo = 1 - ss_res_loo / ss_tot if ss_tot > 0 else 0

            single_models.append((feat, r2, r2_loo, beta))

        single_models.sort(key=lambda x: x[1], reverse=True)
        print(f"  {'Feature':30s}  {'R2':>8s}  {'LOO-CV R2':>10s}")
        for feat, r2, r2_loo, beta in single_models[:5]:
            marker = " **" if r2 >= 0.9 else ""
            print(f"  {feat:30s}  {r2:8.4f}  {r2_loo:10.4f}{marker}")

        # Best 2-feature OLS (exhaustive)
        if n >= 4:
            print(f"\n  --- Best 2-feature OLS models (top 10) ---")
            pair_models = []
            for f1, f2 in combinations(all_features, 2):
                x1 = df[f1].astype(float).values
                x2 = df[f2].astype(float).values
                if np.std(x1) == 0 or np.std(x2) == 0:
                    continue
                X = np.column_stack([np.ones(n), x1, x2])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                # LOO-CV R^2
                loo_preds = np.zeros(n)
                for i in range(n):
                    X_train = np.delete(X, i, axis=0)
                    y_train = np.delete(y, i)
                    try:
                        b = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                        loo_preds[i] = X[i] @ b
                    except np.linalg.LinAlgError:
                        loo_preds[i] = y.mean()
                ss_res_loo = np.sum((y - loo_preds) ** 2)
                r2_loo = 1 - ss_res_loo / ss_tot if ss_tot > 0 else 0

                pair_models.append((f1, f2, r2, r2_loo))

            pair_models.sort(key=lambda x: x[2], reverse=True)
            print(f"  {'Features':50s}  {'R2':>8s}  {'LOO-CV R2':>10s}")
            for f1, f2, r2, r2_loo in pair_models[:10]:
                marker = " **" if r2 >= 0.9 else ""
                print(f"  ({f1}, {f2}){'':>{48-len(f1)-len(f2)-4}s}  {r2:8.4f}  {r2_loo:10.4f}{marker}")
        else:
            print(f"\n  (Only {n} datasets — skipping 2-feature models)")

        results_by_backbone[bb] = {
            "common": common,
            "gap": gap,
            "df": df,
            "single_models": single_models,
        }

    return results_by_backbone


# =====================================================================
# Section 6: Scatter Plots
# =====================================================================

def make_scatter_plots(results_by_backbone: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"SECTION 6: SCATTER PLOTS (saving to {output_dir}/)")
    print("=" * 90)

    for bb, res in results_by_backbone.items():
        bb_short = bb.replace("vit-base-patch16-224", "vit-base").replace("resnet-50", "rn50")
        df = res["df"]
        gap = df["gap"]
        single_models = res["single_models"]

        if not single_models:
            continue

        # Plot top 4 features
        top_feats = [m[0] for m in single_models[:4]]
        fig, axes = plt.subplots(1, len(top_feats), figsize=(5 * len(top_feats), 4.5))
        if len(top_feats) == 1:
            axes = [axes]

        for ax, feat in zip(axes, top_feats):
            x = df[feat].astype(float)
            ax.scatter(x, gap * 100, s=60, edgecolors="k", linewidths=0.5, zorder=3)
            for ds in df.index:
                ax.annotate(ds, (x[ds], gap[ds] * 100), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))

            # Trend line
            slope, intercept, r, p, _ = stats.linregress(x, gap * 100)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7,
                    label=f"R²={r**2:.3f}, p={p:.3f}")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

            ax.set_xlabel(feat)
            ax.set_ylabel("Gap (sup - SSL) [%]")
            ax.legend(fontsize=8)
            ax.set_title(f"{feat} vs gap ({bb_short})")

        fig.tight_layout()
        fname = f"gap_vs_features_{bb_short}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {output_dir}/{fname}")

    print()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SSL vs Supervised gap analysis")
    parser.add_argument("--output-dir", default="figures/", help="Directory for scatter plots")
    args = parser.parse_args()

    print("=" * 90)
    print("SSL vs SUPERVISED GAP ANALYSIS")
    print("=" * 90)

    # Section 1: Load
    ssl_df = load_ssl_results()
    sup_df = load_supervised_results()
    epochs_df = load_pretrain_epochs()

    print(f"\nLoaded {len(ssl_df)} SSL results ({ssl_df['method'].nunique()} methods, {ssl_df['dataset'].nunique()} datasets)")
    print(f"Loaded {len(sup_df)} supervised results ({sup_df['backbone'].nunique()} backbones, {sup_df['dataset'].nunique()} datasets)")
    print(f"Loaded {len(epochs_df)} epoch configs ({epochs_df['method'].nunique()} methods)")

    # Section 2: Tables
    print_ssl_table(ssl_df)
    print_supervised_table(sup_df)
    print_comparison_table(ssl_df, sup_df)

    # Section 3: Metadata
    print_metadata()

    # Section 4: Confounds
    print_confounds(ssl_df, sup_df, epochs_df)

    # Section 5: Gap analysis
    results = gap_analysis(ssl_df, sup_df)

    # Section 6: Scatter plots
    make_scatter_plots(results, args.output_dir)

    # Summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"""
SSL methods: {', '.join(SSL_METHODS)} (all ViT-small, linear probe)
Supervised backbones: {', '.join(sorted(sup_df['backbone'].unique()))}
  (trained from scratch, {int(sup_df['sup_epochs'].iloc[0])} epochs)

SSL pretraining epochs vary by method and dataset ({epochs_df['pretrain_epochs'].min()}-{epochs_df['pretrain_epochs'].max()}).
Supervised always uses 100 epochs from random init.

Key confounds to consider:
  1. Backbone mismatch: SSL uses ViT-small; supervised uses ResNet-50 / ViT-base
  2. Epoch mismatch: SSL pretraining epochs differ from supervised training epochs
  3. Training paradigm: SSL = pretrain + linear probe; supervised = end-to-end fine-tune
""")


if __name__ == "__main__":
    main()
