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
    print("\nwrote:")
    for name in [
        "paper_fig_scatter_three_encoders.pdf",
        "paper_fig_scatter_three_encoders.png",
        "paper_table_sensitivity.tex",
        "paper_table_cross_encoder_rho.tex",
        "paper_table_dataset.tex",
    ]:
        print(f"  {OUT_DIR / name}")


if __name__ == "__main__":
    main()
