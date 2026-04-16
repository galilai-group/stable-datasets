"""Generate paper-ready LaTeX tables and figures for the intra-class variation
sensitivity analysis.

Reads per-encoder intra-class variation CSVs from benchmarks/results/ and
the gap CSV from Stage 1, joins them, and writes:

  paper_fig_scatter_three_encoders.{pdf,png}  — 3-panel scatter with per-panel
                                                OLS fit and ρ annotation
  paper_table_sensitivity.tex                 — 3-row summary: extractor,
                                                objective, corpus, ρ, p
  paper_table_dataset.tex                     — full per-dataset booktabs
                                                table with all three v columns
  paper_table_cross_encoder_rho.tex           — 3×3 extractor-vs-extractor
                                                Spearman agreement matrix

All files land in benchmarks/results/. Run after each encoder's CSV exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "results"
GAP_CSV = OUT_DIR / "ssl_supervised_gap.csv"

ENCODERS: dict[str, dict] = {
    "dinov2": dict(
        label="DINOv2 ViT-S/14",
        objective="augmentation-invariance (DINO+iBOT)",
        corpus="LVD-142M",
    ),
    "in21k": dict(
        label="IN-21k ViT-S/16",
        objective="supervised cross-entropy",
        corpus="ImageNet-21k",
    ),
    "clip": dict(
        label="CLIP ViT-L/14",
        objective="image-text contrastive",
        corpus="OpenAI WIT-400M",
    ),
}

ORDER = ["dinov2", "in21k", "clip"]


def load_scores() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for enc in ORDER:
        path = OUT_DIR / f"intraclass_variation_{enc}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        out[enc] = pd.read_csv(path)[["dataset", "intraclass_variation"]].rename(
            columns={"intraclass_variation": f"v_{enc}"}
        )
    return out


def build_joined() -> pd.DataFrame:
    gaps = pd.read_csv(GAP_CSV)[
        ["dataset", "best_supervised", "best_ssl", "best_ssl_method", "ssl_advantage"]
    ]
    scores = load_scores()
    df = gaps
    for enc in ORDER:
        df = df.merge(scores[enc], on="dataset", how="inner")
    return df.sort_values("v_dinov2").reset_index(drop=True)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for enc in ORDER:
        rho, p_rho = stats.spearmanr(df[f"v_{enc}"], df["ssl_advantage"])
        r, p_r = stats.pearsonr(df[f"v_{enc}"], df["ssl_advantage"])
        rows.append(
            dict(
                encoder=enc,
                label=ENCODERS[enc]["label"],
                objective=ENCODERS[enc]["objective"],
                corpus=ENCODERS[enc]["corpus"],
                spearman=float(rho),
                spearman_p=float(p_rho),
                pearson=float(r),
                pearson_p=float(p_r),
            )
        )
    return pd.DataFrame(rows)


def compute_cross_encoder(df: pd.DataFrame) -> pd.DataFrame:
    cols = [f"v_{e}" for e in ORDER]
    mat = np.eye(len(ORDER))
    for i, a in enumerate(ORDER):
        for j, b in enumerate(ORDER):
            if i >= j:
                continue
            rho, _ = stats.spearmanr(df[f"v_{a}"], df[f"v_{b}"])
            mat[i, j] = rho
            mat[j, i] = rho
    return pd.DataFrame(mat, index=ORDER, columns=ORDER)


def write_scatter(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    y = df["ssl_advantage"].to_numpy()
    for ax, enc in zip(axes, ORDER):
        x = df[f"v_{enc}"].to_numpy()
        colors = ["#1f77b4" if a >= 0 else "#d62728" for a in y]
        ax.scatter(x, y, c=colors, s=55, edgecolors="k", linewidths=0.4)
        ax.axhline(0, color="k", lw=0.5)
        for _, r in df.iterrows():
            ax.annotate(
                r["dataset"],
                (r[f"v_{enc}"], r["ssl_advantage"]),
                fontsize=6.5,
                alpha=0.75,
                xytext=(3, 2),
                textcoords="offset points",
            )
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="gray", lw=1.0, linestyle="--")
        rho = float(summary.loc[summary.encoder == enc, "spearman"].iloc[0])
        p = float(summary.loc[summary.encoder == enc, "spearman_p"].iloc[0])
        ax.set_title(
            f"{ENCODERS[enc]['label']}\nρ = {rho:.2f}  (p = {p:.3f})",
            fontsize=11,
        )
        ax.set_xlabel(f"intra-class variation  $v_{{{enc}}}$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"ssl_advantage  (best SSL $-$ supervised)")
    fig.suptitle(
        "SSL advantage vs measured intra-class variation across three feature extractors",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "paper_fig_scatter_three_encoders.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "paper_fig_scatter_three_encoders.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_sensitivity_table(summary: pd.DataFrame) -> None:
    lines = [
        r"% intra-class variation sensitivity analysis — 3 feature extractors",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Extractor & Objective & Pretraining corpus & \# params & Spearman $\rho$ & $p$ \\",
        r"\midrule",
    ]
    params = {"dinov2": "22M", "in21k": "22M", "clip": "304M"}
    for _, r in summary.iterrows():
        lines.append(
            f"{r.label} & {r.objective} & {r.corpus} & {params[r.encoder]} & "
            f"{r.spearman:.3f} & {r.spearman_p:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "paper_table_sensitivity.tex").write_text("\n".join(lines) + "\n")


def write_cross_encoder_table(cross: pd.DataFrame) -> None:
    lines = [
        r"% Spearman rank agreement between per-dataset intra-class variation scores",
        r"% under the three feature extractors (all n=19 datasets).",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & " + " & ".join(ENCODERS[e]["label"] for e in ORDER) + r" \\",
        r"\midrule",
    ]
    for a in ORDER:
        row = [ENCODERS[a]["label"]]
        for b in ORDER:
            val = cross.loc[a, b]
            row.append("--" if a == b else f"{val:.2f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "paper_table_cross_encoder_rho.tex").write_text("\n".join(lines) + "\n")


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:.3f}"


def write_dataset_table(df: pd.DataFrame) -> None:
    lines = [
        r"% Per-dataset intra-class variation and SSL advantage.",
        r"% Rows sorted by v_dinov2.",
        r"\begin{tabular}{lrrrcr r r r}",
        r"\toprule",
        r"Dataset & sup top-1 & best SSL & SSL method & $\Delta$ & "
        r"$v_{\text{DINOv2}}$ & $v_{\text{IN21k}}$ & $v_{\text{CLIP}}$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        delta_str = f"{r.ssl_advantage:+.3f}"
        if r.ssl_advantage > 0:
            delta_str = r"\textbf{" + delta_str + r"}"
        name = r.dataset.replace("_", r"\_")
        method = r.best_ssl_method.replace("_", r"\_")
        lines.append(
            f"{name} & {r.best_supervised:.3f} & {r.best_ssl:.3f} & {method} & "
            f"{delta_str} & {r.v_dinov2:.3f} & {r.v_in21k:.3f} & {r.v_clip:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "paper_table_dataset.tex").write_text("\n".join(lines) + "\n")


METADATA_CSV = OUT_DIR / "dataset_metadata.csv"

# All candidate predictors of ssl_advantage, grouped and labelled for the
# big "predictor ranking" table. Each entry is (column_name, display_label,
# source_csv, category).
PREDICTORS: list[tuple[str, str, str, str]] = [
    # --- dataset metadata (counting / image-level) ---
    ("train_size",       r"Training-set size",                "metadata",  "Dataset metadata"),
    ("num_classes",      r"Number of classes $K$",            "gap",       "Dataset metadata"),
    ("images_per_class", r"Images per class",                 "gap",       "Dataset metadata"),
    ("class_balance",    r"Class balance (norm.\ entropy)",   "metadata",  "Dataset metadata"),
    ("mean_pixels",      r"Mean resolution ($H \times W$)",   "metadata",  "Dataset metadata"),
    ("mean_channels",    r"Mean channels",                    "metadata",  "Dataset metadata"),
    # --- hand-coded binary flags ---
    ("natural",          r"Natural images (binary)",          "gap",       "Hand-coded"),
    ("fine_grained",     r"Fine-grained (binary)",            "gap",       "Hand-coded"),
    ("grayscale",        r"Grayscale (binary)",               "gap",       "Hand-coded"),
    ("centered",         r"Centered / clean bg.\ (binary)",   "gap",       "Hand-coded"),
    # --- supervised performance ---
    ("supervised_ceiling", r"Supervised top-1 (ceiling)",     "gap",       "Supervised performance"),
    # --- measured intra-class variation ---
    ("v_dinov2",         r"Intra-class var.\ (DINOv2)",       "joined",    r"Intra-class variation"),
    ("v_in21k",          r"Intra-class var.\ (IN-21k sup.)",  "joined",    r"Intra-class variation"),
    ("v_clip",           r"Intra-class var.\ (CLIP)",         "joined",    r"Intra-class variation"),
]


def write_predictor_table(df_joined: pd.DataFrame) -> None:
    """Big table: every candidate predictor vs ssl_advantage."""
    gap = pd.read_csv(GAP_CSV)
    meta = pd.read_csv(METADATA_CSV) if METADATA_CSV.exists() else pd.DataFrame()

    # Build a single wide frame with all columns we need
    wide = df_joined.copy()
    for col in ["num_classes", "train_size", "natural", "fine_grained",
                "grayscale", "centered", "images_per_class",
                "supervised_ceiling"]:
        if col not in wide.columns and col in gap.columns:
            wide = wide.merge(gap[["dataset", col]], on="dataset", how="left")
    if not meta.empty:
        for col in ["class_balance", "mean_pixels", "mean_channels"]:
            if col not in wide.columns and col in meta.columns:
                wide = wide.merge(meta[["dataset", col]], on="dataset", how="left")

    lines = [
        r"% Comprehensive predictor ranking: Spearman ρ of each candidate",
        r"% feature against ssl_advantage (n = 19 datasets).",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Category & Predictor & Spearman $\rho$ & $p$ \\",
        r"\midrule",
    ]

    prev_cat = None
    rows_data = []
    for col, label, source, cat in PREDICTORS:
        if col not in wide.columns:
            continue
        vals = wide[col].dropna()
        if len(vals) < 5:
            continue
        idx = vals.index
        rho, p = stats.spearmanr(wide.loc[idx, col], wide.loc[idx, "ssl_advantage"])
        rows_data.append((cat, label, float(rho), float(p)))

    # Sort by |ρ| descending within each category
    for cat, label, rho, p in rows_data:
        if cat != prev_cat:
            if prev_cat is not None:
                lines.append(r"\addlinespace")
            prev_cat = cat
        sig = ""
        if p < 0.01:
            sig = r"$^{**}$"
        elif p < 0.05:
            sig = r"$^{*}$"
        rho_str = f"{rho:+.3f}"
        # Bold the intra-class variation rows (the winners)
        if "Intra-class" in cat:
            rho_str = r"\textbf{" + rho_str + r"}"
        lines.append(
            f"{cat} & {label} & {rho_str}{sig} & {p:.4f} \\\\"
        )
        # Don't repeat the category in subsequent rows
        cat = ""

    lines += [
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize $^{*}\,p < 0.05$, $^{**}\,p < 0.01$.}",
        r"\end{tabular}",
    ]
    out_path = OUT_DIR / "paper_table_predictors.tex"
    out_path.write_text("\n".join(lines) + "\n")

    # Also print to stdout
    print("\n=== predictor ranking (Spearman ρ vs ssl_advantage) ===")
    for cat, label, rho, p in sorted(rows_data, key=lambda x: -abs(x[2])):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {rho:+.3f} {sig:3s}  {label}")


def main() -> None:
    df = build_joined()
    summary = compute_summary(df)
    cross = compute_cross_encoder(df)

    print("=== primary ρ (intra-class vs ssl_advantage) ===")
    print(summary[["encoder", "spearman", "spearman_p", "pearson", "pearson_p"]].to_string(index=False))
    print("\n=== cross-encoder rank agreement (Spearman) ===")
    print(cross.round(3).to_string())

    write_scatter(df, summary)
    write_sensitivity_table(summary)
    write_cross_encoder_table(cross)
    write_dataset_table(df)
    write_predictor_table(df)
    print("\nwrote:")
    for name in [
        "paper_fig_scatter_three_encoders.pdf",
        "paper_fig_scatter_three_encoders.png",
        "paper_table_sensitivity.tex",
        "paper_table_cross_encoder_rho.tex",
        "paper_table_dataset.tex",
        "paper_table_predictors.tex",
    ]:
        print(f"  {OUT_DIR / name}")


if __name__ == "__main__":
    main()
