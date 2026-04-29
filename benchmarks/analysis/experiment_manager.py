"""Manages representation-geometry experiment history.

Mirrors profiling/experiment_manager.py: one JSON file, append-only, atomic
writes, idempotent on a stable key. Each entry records one representation-
score run over a single (model, dataset, seed) checkpoint.

Usage from representation_scores.py::

    from experiment_manager import ExperimentManager

    mgr = ExperimentManager()   # default path
    mgr.append(
        model="lejepa",
        dataset="cifar10",
        seed=None,
        split="val",
        checkpoint="…/last.ckpt",
        config={"n_per_class_cap": 32, "batch_size": 64, ...},
        results={"rankme": 52.4, "lidar": 7.87, "lidar_n_classes_used": 10, ...},
    )

CLI::

    python benchmarks/analysis/experiment_manager.py --summary
    python benchmarks/analysis/experiment_manager.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "representation_scores_history.json"
)


def _run_key(model: str, dataset: str, seed: int | None, split: str) -> str:
    return f"{model}|{dataset}|{seed if seed is not None else 'none'}|{split}"


class ExperimentManager:
    """Append-only history backed by a single JSON file.

    Entries are keyed by ``(model, dataset, seed, split)``; ``append`` is
    idempotent on that key. Every successful append auto-saves via an
    atomic write-to-tmp-then-``os.replace``, so partial progress survives
    crashes or cancelled jobs.
    """

    def __init__(self, path: Path | str = DEFAULT_HISTORY_PATH):
        self.path = Path(path)
        self._history: dict | None = None

    @property
    def history(self) -> dict:
        if self._history is None:
            self._history = self._load()
        return self._history

    @property
    def runs(self) -> list[dict]:
        return self.history["runs"]

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if "runs" not in data:
                raise ValueError(f"{self.path} exists but has no 'runs' key")
            return data
        return {"schema_version": 1, "runs": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.history, indent=2))
        os.replace(tmp, self.path)

    def append(
        self,
        *,
        model: str,
        dataset: str,
        seed: int | None,
        split: str,
        checkpoint: str,
        config: dict,
        results: dict,
        overwrite: bool = False,
        timestamp: str | None = None,
    ) -> bool:
        """Append (or replace) a run. Returns True if stored, False if duplicate skipped."""
        key = _run_key(model, dataset, seed, split)
        existing_idx = None
        for i, r in enumerate(self.runs):
            if r.get("key") == key:
                existing_idx = i
                break
        if existing_idx is not None and not overwrite:
            return False
        entry = {
            "key": key,
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "split": split,
            "checkpoint": checkpoint,
            "timestamp": timestamp or datetime.utcnow().isoformat(timespec="seconds"),
            "config": config,
            "results": results,
        }
        if existing_idx is not None:
            self.runs[existing_idx] = entry
        else:
            self.runs.append(entry)
        self.save()
        return True

    def has(self, model: str, dataset: str, seed: int | None = None, split: str = "val") -> bool:
        key = _run_key(model, dataset, seed, split)
        return any(r.get("key") == key for r in self.runs)

    def as_rows(self) -> list[dict]:
        rows = []
        for r in self.runs:
            row = {
                "model": r["model"],
                "dataset": r["dataset"],
                "seed": r.get("seed"),
                "split": r.get("split", "val"),
                "n_samples": r.get("results", {}).get("n_samples"),
                "n_classes": r.get("results", {}).get("n_classes"),
                "embed_dim": r.get("results", {}).get("embed_dim"),
                "rankme": r.get("results", {}).get("rankme"),
                "lidar": r.get("results", {}).get("lidar"),
                "lidar_n_classes_used": r.get("results", {}).get("lidar_n_classes_used"),
            }
            rows.append(row)
        return rows

    def summary(self) -> str:
        rows = self.as_rows()
        if not rows:
            return f"{self.path}: empty history."
        models = sorted({r["model"] for r in rows})
        datasets = sorted({r["dataset"] for r in rows})
        by_key = {(r["model"], r["dataset"]): r for r in rows}
        out = [f"{len(rows)} runs over {len(models)} models × {len(datasets)} datasets"]
        out.append("")
        out.append(f"{'dataset':<20} " + " ".join(f"{m:>14}" for m in models))
        out.append("-" * (20 + 15 * len(models)))
        for ds in datasets:
            cells = []
            for m in models:
                r = by_key.get((m, ds))
                if r is None:
                    cells.append(f"{'--':>14}")
                else:
                    rm = r["rankme"]
                    ld = r["lidar"]
                    rm_s = f"{rm:>6.2f}" if rm is not None else "    --"
                    ld_s = f"{ld:>5.2f}" if ld is not None else "   --"
                    cells.append(f"{rm_s}/{ld_s}")
            out.append(f"{ds:<20} " + " ".join(cells))
        out.append("")
        out.append("cells show rankme / lidar  (— = missing)")
        return "\n".join(out)

    def export_csv(self, path: Path | str) -> None:
        rows = self.as_rows()
        if not rows:
            Path(path).write_text("")
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def gather_shards(self, shard_dir: Path | str, overwrite: bool = False) -> int:
        """Merge per-shard JSON files (written by parallel SLURM tasks) into this history.

        Each shard is a standalone ExperimentManager history file (same schema).
        Shards have one run entry each, written idempotently by the extractor.
        Returns the number of newly added runs.
        """
        shard_dir = Path(shard_dir)
        added = 0
        for shard_path in sorted(shard_dir.glob("*.json")):
            try:
                data = json.loads(shard_path.read_text())
            except Exception as e:
                print(f"  skip {shard_path.name}: {e}")
                continue
            for run in data.get("runs", []):
                was_added = self.append(
                    model=run["model"],
                    dataset=run["dataset"],
                    seed=run.get("seed"),
                    split=run.get("split", "val"),
                    checkpoint=run.get("checkpoint", ""),
                    config=run.get("config", {}),
                    results=run.get("results", {}),
                    timestamp=run.get("timestamp"),
                    overwrite=overwrite,
                )
                if was_added:
                    added += 1
        return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation-score experiment manager")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="export all runs to a flat CSV")
    parser.add_argument(
        "--gather",
        type=Path,
        default=None,
        help="merge per-task shard JSONs from this directory into the main history",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="with --gather, replace existing entries on key collision"
    )
    args = parser.parse_args()

    mgr = ExperimentManager(args.history)
    if args.gather:
        added = mgr.gather_shards(args.gather, overwrite=args.overwrite)
        print(f"gathered {added} new runs from {args.gather} → {args.history}  (total runs: {len(mgr.runs)})")
    if args.summary:
        print(mgr.summary())
    if args.csv:
        mgr.export_csv(args.csv)
        print(f"wrote {args.csv}  ({len(mgr.runs)} rows)")


if __name__ == "__main__":
    main()
