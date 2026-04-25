"""Shared globals, paths, and helpers for the analysis pipeline.

Everything other modules need that isn't a pipeline stage in itself lives here:
filesystem layout, training-sweep constants, the SSL method roster, the
collapsed-run skip list, output CSV/JSON paths, and a couple of small
loaders that more than one script wants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --- filesystem layout ---------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "results"
CONFIG_DIR = str(REPO / "benchmarks" / "conf")
CKPT_ROOT = Path("/oscar/home/sboughan/scratch/.stable-datasets/.pretrain_checkpoints")

# --- sweep parameters ----------------------------------------------------

BACKBONE = "vit_small"
MAX_SAMPLES = 4096
LIDAR_SAMPLES_PER_CLASS = 10  # paper default

# --- methods + filters ---------------------------------------------------

# Order matters for table rendering. MAE is reconstruction-only and excluded
# from "best SSL" maxes in ssl_supervised_gap.
SSL_METHODS = ["simclr", "barlow_twins", "nnclr", "dino", "lejepa", "mae"]

# Confirmed training collapses to exclude from extraction and correlation.
# val rankme ≈ 1, kNN at chance — rerunning yields garbage.
COLLAPSED: set[tuple[str, str]] = {("dino", "arabiccharacters")}

# --- output paths --------------------------------------------------------

BENCHMARK_RESULTS_CSV = OUT_DIR / "benchmark_results.csv"
REP_SCORES_CSV = OUT_DIR / "representation_scores.csv"
GAP_CSV = OUT_DIR / "ssl_supervised_gap.csv"
GAP_RANKING_CSV = OUT_DIR / "ssl_supervised_gap_feature_ranking.csv"
REPORT_CSV = OUT_DIR / "correlation_report.csv"

HISTORY_PATHS: dict[str, Path] = {
    "val": OUT_DIR / "representation_scores_history.json",
    "train": OUT_DIR / "representation_scores_train_history.json",
}

# --- small helpers other modules want -----------------------------------


def run_dir(model: str, dataset: str) -> Path:
    """Checkpoint directory for a (model, dataset) pair."""
    return CKPT_ROOT / f"{model}_{BACKBONE}_{dataset}"


def latest_checkpoint(model: str, dataset: str) -> Path | None:
    """Most recent .ckpt for (model, dataset), preferring last.ckpt."""
    d = run_dir(model, dataset)
    if not d.exists():
        return None
    last = d / "last.ckpt"
    if last.exists():
        return last
    ckpts = sorted(d.glob("*.ckpt"))
    return ckpts[-1] if ckpts else None


def discover_all() -> list[tuple[str, str]]:
    """All (model, dataset) pairs that have a checkpoint dir under CKPT_ROOT."""
    out: list[tuple[str, str]] = []
    if not CKPT_ROOT.exists():
        return out
    for d in sorted(CKPT_ROOT.iterdir()):
        if not d.is_dir() or f"_{BACKBONE}_" not in d.name:
            continue
        model, rest = d.name.split(f"_{BACKBONE}_", 1)
        out.append((model, rest))
    return out


def load_history(path: Path, split_label: str = "val") -> pd.DataFrame:
    """Load an ExperimentManager history JSON into a flat DataFrame.

    Columns: model, dataset, split, rankme, lidar, lidar_n_classes_used,
    n_classes, embed_dim. Rows in COLLAPSED are filtered out.
    """
    if not path.exists():
        return pd.DataFrame(columns=["model", "dataset", "split", "rankme", "lidar"])
    data = json.loads(path.read_text())
    rows = []
    for r in data.get("runs", []):
        if (r["model"], r["dataset"]) in COLLAPSED:
            continue
        res = r.get("results", {})
        rows.append(
            dict(
                model=r["model"],
                dataset=r["dataset"],
                split=r.get("split", split_label),
                rankme=res.get("rankme"),
                lidar=res.get("lidar"),
                lidar_n_classes_used=res.get("lidar_n_classes_used"),
                n_classes=res.get("n_classes"),
                embed_dim=res.get("embed_dim"),
            )
        )
    return pd.DataFrame(rows)
