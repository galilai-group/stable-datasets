"""Fill in the ONLINE metrics for cells whose checkpoints were cleared from scratch
(original-paper cells) by scraping their curated W&B run — the `wandb` origin.

These cells cannot be recomputed (no checkpoint on disk), so we fall back to the
run each cell was selected to in `benchmarks/analysis/selected_runs.csv`. Those are
the paper's curated, pre-contamination runs; we VERIFY each was created before
2026-07-10 (the post-July contamination window, e.g. run 7rbvv9tl on 07-26) and
loudly flag any that isn't.

Writes one file per cell into RECOVER_DIR in the SAME schema as
recover_online_probe.py, with origin="wandb", so the two sources merge cleanly.

    python -m benchmarks.scrape_wandb_online [wandb_orig_targets.tsv]
"""

from __future__ import annotations

import csv
import os
import sys

import wandb

from benchmarks.dataset import get_config

ENTITY_PROJECT = os.environ.get("SDS_WANDB_ENTITY_PROJECT", "samibg/stable-datasets-iclr")
CONTAMINATION_CUTOFF = "2026-07-10T00:00:00"  # runs on/after this may be contaminated
RECOVER_DIR = os.environ.get(
    "RECOVER_DIR",
    os.path.join(os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")),
                 "online_probe_recovered"),
)
SELECTED = os.path.join(os.path.dirname(__file__), "analysis", "selected_runs.csv")
CSV_HEADER = [
    "dataset", "method", "seed", "origin",
    "linear_top1", "linear_top5", "knn_top1", "knn_top5", "rankme", "lidar",
    "rankme_condition_number", "rankme_entropy", "lidar_entropy",
    "epoch", "max_epochs", "num_classes", "backbone", "checkpoint_or_run",
]


def _write_cell(method: str, dataset: str, seed: str, row: dict) -> None:
    os.makedirs(RECOVER_DIR, exist_ok=True)
    stem = f"{method}__{dataset}__seed{seed if seed else 'null'}.csv"
    with open(os.path.join(RECOVER_DIR, stem), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})


def main() -> None:
    targets_path = sys.argv[1] if len(sys.argv) > 1 else "/oscar/scratch/sboughan/wandb_orig_targets.tsv"
    targets = []
    for line in open(targets_path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0]:
            targets.append((p[0], p[1], p[2] if len(p) > 2 else ""))

    # map (method, dataset, seed) -> curated wandb_run_id from selected_runs.csv
    run_of = {}
    with open(SELECTED) as f:
        for r in csv.DictReader(f):
            run_of[(r["model"], r["dataset"], (r.get("seed") or "").strip())] = r["wandb_run_id"]

    api = wandb.Api(timeout=60)

    def num(v):
        try:
            return round(float(v), 6)
        except (TypeError, ValueError):
            return ""

    n_ok = n_missing_id = n_post_july = n_err = 0
    for method, dataset, seed in targets:
        rid = run_of.get((method, dataset, seed))
        if not rid:
            print(f"!! no run_id in selected_runs.csv for {method}/{dataset}/seed{seed or 'null'}")
            n_missing_id += 1
            continue
        try:
            run = api.run(f"{ENTITY_PROJECT}/{rid}")
            s = run.summary
            created = str(run.created_at)
            if created >= CONTAMINATION_CUTOFF:
                print(f"!! POST-JULY-10 run for {method}/{dataset} (id={rid}, created={created}) "
                      "— may be contaminated; review before trusting.")
                n_post_july += 1
            try:
                nc = get_config(dataset).num_classes
            except Exception:
                nc = ""
            row = {
                "dataset": dataset, "method": method, "seed": seed, "origin": "wandb",
                "linear_top1": num(s.get("eval/linear_probe_top1_epoch")),
                "linear_top5": num(s.get("eval/linear_probe_top5_epoch")),
                "knn_top1": num(s.get("eval/knn_probe_top1")),
                "knn_top5": num(s.get("eval/knn_probe_top5")),
                "rankme": num(s.get("rankme")),
                "lidar": num(s.get("lidar")),
                "rankme_condition_number": num(s.get("rankme/condition_number")),
                "rankme_entropy": num(s.get("rankme/entropy")),
                "lidar_entropy": num(s.get("lidar/entropy")),
                "epoch": s.get("epoch", ""),
                "max_epochs": "", "num_classes": nc,
                "backbone": "vit_small_patch16_224",
                "checkpoint_or_run": f"wandb:{rid}@{created[:10]}",
            }
            _write_cell(method, dataset, seed, row)
            n_ok += 1
        except Exception as e:
            print(f"!! error scraping {method}/{dataset} (id={rid}): {type(e).__name__}: {e}")
            n_err += 1

    print(f"\nwandb scrape: {n_ok} written | {n_missing_id} no-run-id | "
          f"{n_post_july} POST-July-10(flagged) | {n_err} errors  (of {len(targets)} targets)")


if __name__ == "__main__":
    main()
