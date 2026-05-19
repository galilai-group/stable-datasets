"""Render per-method linear probe results into a LaTeX summary table.

Reads best-run metrics from a CSV (produced by analysis.py --write-selected-runs-csv)
or re-fetches from W&B when no CSV is supplied.

Usage:
    # use cached runs (fast)
    python benchmarks/analysis_new/render_latex.py \
        --selected-runs-csv benchmarks/analysis_new/selected_runs.csv

    # re-fetch from W&B
    python benchmarks/analysis_new/render_latex.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
DEFAULT_SELECTED_RUNS_CSV = HERE / "selected_runs.csv"
DEFAULT_METADATA_CSV = HERE / "dataset_metadata.csv"
DEFAULT_OUTPUT_TEX = HERE / "benchmark_table_probe.tex"

DEFAULT_ENTITY = ""
DEFAULT_PROJECT = "finalized-anonymous-datasets"
DEFAULT_BACKBONES = ("vit_small_patch16_224", "vit_small")

SSL_METHODS = ("simclr", "barlow_twins", "nnclr", "dino", "lejepa", "mae")
SUPERVISED_METHOD = "supervised"
METHOD_ORDER = (*SSL_METHODS, SUPERVISED_METHOD)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "simclr": "SimCLR",
    "dino": "DINO",
    "mae": "MAE",
    "lejepa": "LeJEPA",
    "nnclr": "NNCLR",
    "barlow_twins": "Barlow Twins",
    "supervised": "Supervised",
}

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "arabiccharacters": "Arabic Characters",
    "arabicdigits": "Arabic Digits",
    "beans": "Beans",
    "bloodmnist": "BloodMNIST",
    "breastmnist": "BreastMNIST",
    "cars196": "Cars-196",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "country211": "Country-211",
    "cub200": "CUB-200",
    "dermamnist": "DermaMNIST",
    "dtd": "DTD",
    "emnist": "EMNIST",
    "emnist_byclass": "EMNIST ByClass",
    "emnist_bymerge": "EMNIST ByMerge",
    "emnist_digits": "EMNIST Digits",
    "emnist_letters": "EMNIST Letters",
    "emnist_mnist": "EMNIST MNIST",
    "fashionmnist": "FashionMNIST",
    "fgvcaircraft": "FGVC Aircraft",
    "fgvcaircraft_family": "FGVC Aircraft (family)",
    "fgvcaircraft_manufacturer": "FGVC Aircraft (mfr.)",
    "flowers102": "Flowers-102",
    "food101": "Food-101",
    "galaxy10": "Galaxy10",
    "hasyv2": "HASYv2",
    "imagenet100": "ImageNet-100",
    "imagenette": "Imagenette",
    "pneumoniamnist": "PneumoniaMNIST",
    "notmnist": "notMNIST",
    "octmnist": "OCTMNIST",
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
    "pathmnist": "PathMNIST",
    "retinamnist": "RetinaMNIST",
    "rockpaperscissor": "Rock Paper Scissors",
    "stl10": "STL-10",
    "svhn": "SVHN",
    "tissuemnist": "TissueMNIST",
}


def _display(key: str) -> str:
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    if key in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[key]
    return key.replace("_", " ").title()


# W&B fetch (only used when selected_runs.csv is absent)


def _retry(fn, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                time.sleep(2**attempt)
                continue
            raise
    return fn()


def _fetch_from_wandb(entity: str, project: str, backbones: tuple[str, ...]) -> pd.DataFrame:
    import wandb

    PROBE_KEY = "eval/linear_probe_top1_epoch"
    KNN_KEY = "eval/knn_probe_top1"
    RANKME_KEYS = ("rankme", "val/rankme", "eval/rankme", "rankme/val")
    DATASET_ALIASES = {"medmnist": "pneumoniamnist"}
    all_methods = set(METHOD_ORDER)

    api = wandb.Api(timeout=60)
    runs = _retry(lambda: list(api.runs(f"{entity}/{project}", per_page=1000)))
    print(f"fetched {len(runs)} runs from W&B", file=sys.stderr)

    rows = []
    for run in runs:
        if _retry(lambda: run.state) not in {"finished", "completed"}:
            continue
        config = _retry(lambda: run.config or {})
        summary = _retry(lambda: dict(run.summary._json_dict if hasattr(run.summary, "_json_dict") else run.summary))

        model = config.get("model") or ""
        dataset = (config.get("dataset") or "").lower()
        backbone = config.get("backbone") or ""

        if not model or model not in all_methods:
            continue
        if not dataset:
            continue
        if backbone not in backbones:
            continue

        dataset = DATASET_ALIASES.get(dataset, dataset)
        probe = summary.get(PROBE_KEY)
        knn = summary.get(KNN_KEY)
        rankme = next((summary.get(k) for k in RANKME_KEYS if summary.get(k) is not None), None)
        if probe is None:
            continue

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "probe": float(probe),
                "knn": float(knn) if knn is not None else None,
                "rankme_val": float(rankme) if rankme is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.dropna(subset=["probe"])
        .sort_values("probe", ascending=False)
        .drop_duplicates(subset=["model", "dataset"], keep="first")
        .reset_index(drop=True)
    )


def load_runs(
    selected_runs_csv: Path | None, *, entity: str, project: str, backbones: tuple[str, ...]
) -> pd.DataFrame:
    if selected_runs_csv is not None and selected_runs_csv.exists():
        return pd.read_csv(selected_runs_csv)
    print("no selected-runs CSV found — fetching from W&B…", file=sys.stderr)
    return _fetch_from_wandb(entity, project, backbones)


def pivot_probe(runs: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    df = runs.dropna(subset=["probe"]).copy()
    df = df.sort_values("probe", ascending=False).drop_duplicates(subset=["model", "dataset"], keep="first")
    table = df.pivot_table(index="dataset", columns="model", values="probe", aggfunc="max")
    table.columns.name = None
    table.index.name = None

    if metadata_csv.exists():
        meta = pd.read_csv(metadata_csv)
        meta["dataset"] = meta["dataset"].str.lower()
        chance = meta.set_index("dataset")["num_classes"].apply(lambda k: 1.0 / k if k > 0 else float("nan"))
        table["Chance"] = chance.reindex(table.index)
    else:
        table["Chance"] = float("nan")

    avg = table.drop(columns="Chance", errors="ignore").mean()
    avg["Chance"] = table["Chance"].mean()
    table.loc["Average"] = avg

    return table


def format_latex(table: pd.DataFrame) -> str:
    pct = table * 100
    method_cols = [m for m in METHOD_ORDER if m in pct.columns]
    all_ds = [idx for idx in pct.index if idx != "Average"]

    def _cell(ds: str, col: str) -> str:
        val = pct.loc[ds, col]
        return "---" if pd.isna(val) else f"{val:.1f}"

    n_cols = len(method_cols) + 1
    lines = [
        f"\\begin{{tabular}}{{l {'c ' * n_cols}}}",
        "\\toprule",
        "\\textbf{Dataset} & " + " & ".join(_display(m) for m in method_cols) + " & \\textbf{Chance} \\\\",
        "\\midrule",
    ]
    for ds in all_ds:
        cells = [_display(ds)] + [_cell(ds, col) for col in method_cols] + [_cell(ds, "Chance")]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    avg_cells = ["\\textbf{Average}"] + [_cell("Average", col) for col in method_cols] + [_cell("Average", "Chance")]
    lines.append(" & ".join(avg_cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--selected-runs-csv", type=Path, default=DEFAULT_SELECTED_RUNS_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    args = parser.parse_args()

    backbones = tuple(b.strip() for b in args.backbones.split(",") if b.strip())

    runs = load_runs(args.selected_runs_csv, entity=args.entity, project=args.project, backbones=backbones)
    if runs.empty:
        print("no runs found")
        return

    table = pivot_probe(runs, args.metadata_csv)

    print("\n=== Linear Probe Top-1 (%) ===")
    print((table * 100).round(1).to_string())

    tex = format_latex(table)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex)
    print(f"\nwrote {args.output_tex}")


if __name__ == "__main__":
    main()
