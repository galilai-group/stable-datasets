"""Render the paper's Table 4 from analysis_new/analysis_results.csv.

This table keeps only the correlations between r_{d,m} and:
  - train size
  - number of classes
  - class balance
  - imbalance ratio
  - number of channels
  - image area (height * width)
  - RankMe (validation)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = HERE / "analysis_results.csv"
DEFAULT_OUTPUT_TEX = HERE / "table4_r_correlations.tex"
DEFAULT_OUTPUT_CSV = HERE / "table4_r_correlations.csv"

METHOD_ORDER = ["simclr", "nnclr", "dino", "barlow_twins", "lejepa", "mae"]
METHOD_LABELS = {
    "simclr": "SimCLR",
    "nnclr": "NNCLR",
    "dino": "DINO",
    "barlow_twins": "Barlow Twins",
    "lejepa": "LeJEPA",
    "mae": "MAE",
}

ROW_SPECS = [
    ("dataset_metadata", "train_size", "Train size"),
    ("dataset_metadata", "num_classes", "Num. classes"),
    ("dataset_metadata", "class_balance", "Class balance"),
    ("dataset_metadata", "imbalance_ratio", "Imbalance ratio"),
    ("dataset_metadata", "mean_channels", "Channels"),
    ("dataset_metadata", "height_times_width", "Image area ($H \\times W$)"),
    ("rankme", "rankme", "RankMe"),
]


def _star(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "^{**}"
    if p < 0.05:
        return "^{*}"
    return ""


def _cell(rho: float | None, p: float | None) -> str:
    if rho is None or pd.isna(rho):
        return "---"
    sign = "+" if rho >= 0 else "-"
    return f"${sign}{abs(rho):.2f}{_star(p)}$"


def build_table(results_csv: Path = DEFAULT_RESULTS_CSV) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(results_csv)
    subset = df[df["lhs"] == "r"].copy()

    rows: list[dict] = []
    latex_lines = [
        "% Table 4: Spearman correlations between r_{d,m} and selected predictors.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Predictor & SimCLR & NNCLR & DINO & Barlow Twins & LeJEPA & MAE \\\\",
        "\\midrule",
    ]

    for rhs_family, rhs_name, label in ROW_SPECS:
        row = {
            "predictor": label,
            "rhs_family": rhs_family,
            "rhs_name": rhs_name,
        }
        cells = [label]
        for method in METHOD_ORDER:
            hit = subset[
                (subset["method"] == method)
                & (subset["rhs_family"] == rhs_family)
                & (subset["rhs_name"] == rhs_name)
            ]
            if hit.empty:
                row[f"{method}_rho"] = None
                row[f"{method}_p"] = None
                row[f"{method}_n"] = None
                cells.append("---")
                continue

            rec = hit.iloc[0]
            row[f"{method}_rho"] = rec["spearman_rho"]
            row[f"{method}_p"] = rec["spearman_p"]
            row[f"{method}_n"] = rec["n"]
            cells.append(_cell(rec["spearman_rho"], rec["spearman_p"]))
        rows.append(row)
        latex_lines.append(" & ".join(cells) + " \\\\")

    latex_lines.extend(
        [
            "\\bottomrule",
            "\\multicolumn{7}{l}{\\footnotesize Entries are Spearman $\\rho$ for correlations with $r_{d,m}$.}",
            "\\multicolumn{7}{l}{\\footnotesize $^{*}\\,p < 0.05$, $^{**}\\,p < 0.01$. RankMe uses the validation split.}",
            "\\end{tabular}",
        ]
    )
    return pd.DataFrame(rows), latex_lines


def main() -> None:
    table_df, latex_lines = build_table()
    DEFAULT_OUTPUT_CSV.write_text(table_df.to_csv(index=False))
    DEFAULT_OUTPUT_TEX.write_text("\n".join(latex_lines) + "\n")
    print(f"wrote {DEFAULT_OUTPUT_CSV}")
    print(f"wrote {DEFAULT_OUTPUT_TEX}")


if __name__ == "__main__":
    main()
