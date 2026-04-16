"""Manages the consolidated experiment history for profiling runs.

Single class, single file, single responsibility: load a history JSON,
append runs to it (idempotent on slurm_job_id, auto-saves on each
append so partial results survive crashes), and print summaries.

Usage from profile_iter.py::

    from experiment_manager import ExperimentManager

    mgr = ExperimentManager()          # default path
    # ... run profiling, collect results dict ...
    mgr.append(slurm_job_id="12345", config={...}, results={...})
    # auto-saved to disk; no separate save step

CLI (replaces the old consolidate_profile_results.py)::

    python profiling/experiment_manager.py --summary
    python profiling/experiment_manager.py old1.json old2.json   # migrate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path


DEFAULT_HISTORY_PATH = Path(__file__).parent.parent / "benchmarks" / "results" / "profile_iter_history.json"


class ExperimentManager:
    """Append-only experiment history backed by a single JSON file.

    Instantiate once with a path; call :meth:`append` after each
    profiling run. Writes are atomic (write-to-tmp then ``os.replace``)
    and each :meth:`append` auto-saves, so partial results survive
    crashes without a separate save step.
    """

    def __init__(self, path: Path | str = DEFAULT_HISTORY_PATH):
        self.path = Path(path)
        self._history: dict | None = None

    # -- Load / save

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
        """Atomic write: tmp file then os.replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.history, indent=2))
        os.replace(tmp, self.path)

    # -- Append ---------------------------------------------------------------

    def append(
        self,
        *,
        slurm_job_id: str,
        config: dict,
        results: dict,
        timestamp: str | None = None,
        source_file: str | None = None,
    ) -> bool:
        """Append a run entry. Idempotent on slurm_job_id.

        Auto-saves to disk on success. Returns True if added, False if
        the job ID was already present (duplicate).
        """
        existing_ids = {r.get("slurm_job_id") for r in self.runs}
        if slurm_job_id in existing_ids:
            return False
        entry = {
            "slurm_job_id": slurm_job_id,
            "timestamp": timestamp or datetime.utcnow().isoformat(timespec="seconds"),
            "config": config,
            "results": results,
        }
        if source_file:
            entry["_source"] = source_file
        self.runs.append(entry)
        self.save()
        return True

    # -- Summary --------------------------------------------------------------

    def summary(self) -> str:
        """Plain-text matrix view: datasets × backends, grouped by config."""
        if not self.runs:
            return f"{self.path}: empty history."

        table: dict = {}
        datasets: set = set()
        backends: set = set()
        configs: set = set()
        for run in self.runs:
            cfg = run.get("config", {})
            mem_gb = cfg.get("mem_gb")
            decode = cfg.get("decode", True)
            for ds_name, per_backend in run.get("results", {}).items():
                datasets.add(ds_name)
                for backend, data in per_backend.items():
                    if data is None:
                        continue
                    epoch_times = data.get("read", [])
                    if not epoch_times:
                        continue
                    mean = statistics.mean(epoch_times)
                    key = (ds_name, mem_gb, decode, backend)
                    table[key] = mean
                    backends.add(backend)
                    configs.add((mem_gb, decode))

        backend_order = [b for b in ("stable", "stable_lance", "hf", "tv") if b in backends]
        lines = []
        for ds_name in sorted(datasets):
            rows = sorted(
                [c for c in configs if any((ds_name, *c, b) in table for b in backends)],
                key=lambda c: (-(c[0] or 0), c[1]),
            )
            if not rows:
                continue
            lines.append(f"\n=== {ds_name} ===")
            header = f"{'mem':>6} {'decode':>8}  " + "".join(f"{b:>22}" for b in backend_order)
            lines.append(header)
            lines.append("-" * len(header))
            for mem_gb, decode in rows:
                mem_str = f"{mem_gb}G" if mem_gb else "?"
                dec_str = "on" if decode else "off"
                cells = []
                baseline = table.get((ds_name, mem_gb, decode, backend_order[0]))
                for b in backend_order:
                    val = table.get((ds_name, mem_gb, decode, b))
                    if val is None:
                        cells.append(f"{'--':>22}")
                    elif baseline and b != backend_order[0]:
                        ratio = val / baseline
                        cells.append(f"{val:>10.1f}s ({ratio:>5.2f}x) ")
                    else:
                        cells.append(f"{val:>10.1f}s{'':>11}")
                lines.append(f"{mem_str:>6} {dec_str:>8}  " + "".join(cells))
        return "\n".join(lines)

    # -- Migration (one-time, for old per-run JSONs) --------------------------

    @staticmethod
    def _infer_config(filename: str) -> dict:
        """Extract run config from a legacy per-run profile JSON filename."""
        name = Path(filename).stem

        m = re.match(r"profile_lance_(\d+)g(?:_nodecode)?_(\d+)$", name)
        if m:
            return {
                "slurm_job_id": m.group(2),
                "config": {
                    "mem_gb": int(m.group(1)),
                    "decode": "nodecode" not in name,
                    "shuffle": True,
                },
            }

        m = re.match(r"profile_lance_cold_(\d+)$", name)
        if m:
            return {
                "slurm_job_id": m.group(1),
                "config": {"mem_gb": 64, "decode": True, "shuffle": True},
            }

        m = re.match(r"profile_lance_matrix_(\d+)$", name)
        if m:
            return {
                "slurm_job_id": m.group(1),
                "config": {
                    "mem_gb": 256, "decode": True, "shuffle": True,
                    "num_runs": 1, "num_epochs": 3, "num_workers": 4,
                    "label": "matrix",
                },
            }

        m = re.match(r"profile_lance_small_(\d+)$", name)
        if m:
            return {
                "slurm_job_id": m.group(1),
                "config": {
                    "mem_gb": 256, "decode": True, "shuffle": True,
                    "num_runs": 2, "num_epochs": 3, "num_workers": 4,
                    "label": "small",
                },
            }

        raise ValueError(f"Cannot infer config from filename {filename!r}")

    def migrate(self, files: list[Path]) -> None:
        """Migrate legacy per-run JSON files into this history."""
        added = skipped = 0
        for f in files:
            try:
                parsed = self._infer_config(f.name)
            except ValueError as e:
                print(f"  skip {f.name}: {e}")
                skipped += 1
                continue
            with f.open() as fh:
                results = json.load(fh)
            ts = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
            was_added = self.append(
                slurm_job_id=parsed["slurm_job_id"],
                config=parsed["config"],
                results=results,
                timestamp=ts,
                source_file=f.name,
            )
            if was_added:
                print(f"  added {f.name} (job {parsed['slurm_job_id']})")
                added += 1
            else:
                print(f"  skip {f.name} (duplicate)")
                skipped += 1
        print(f"\n{self.path}: {len(self.runs)} total runs ({added} added, {skipped} skipped)")


# -- CLI entry point ----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment history manager")
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="Legacy per-run JSON files to migrate (glob profile_lance_*.json if empty)",
    )
    parser.add_argument(
        "--history", type=Path, default=DEFAULT_HISTORY_PATH,
        help=f"History file path (default: {DEFAULT_HISTORY_PATH})",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a plain-text summary of the history and exit",
    )
    args = parser.parse_args()
    mgr = ExperimentManager(args.history)

    if args.summary:
        print(mgr.summary())
        return

    files = args.files or sorted(Path(".").glob("profile_lance_*.json"))
    if not files:
        print("No per-run JSON files found.")
        return
    print(f"Migrating {len(files)} file(s) into {args.history}:")
    mgr.migrate(files)


if __name__ == "__main__":
    main()
