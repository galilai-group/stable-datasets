"""Consolidate per-run profile JSON files into a single history file.

``profile_iter.py`` used to write one JSON per invocation
(``profile_lance_64g_1679212.json`` etc.). That pattern scatters results
across the repo root, loses config metadata (only the filename carried
it), and makes cross-run analysis painful.

This tool reads any number of those files, infers the run config from
the filename, and appends each as an entry to a single consolidated
history file at ``benchmarks/results/profile_iter_history.json``. The
consolidated file is the canonical location going forward;
``profile_iter.py`` is updated to append directly rather than writing
its own one-off JSON.

Filename conventions understood:

    profile_lance_{mem}g_{slurm_id}.json             -- decode=True
    profile_lance_{mem}g_nodecode_{slurm_id}.json    -- decode=False
    profile_lance_matrix_{slurm_id}.json             -- 256G matrix run
    profile_lance_small_{slurm_id}.json              -- 256G small matrix
    profile_lance_cold_{slurm_id}.json               -- pre-rename, 64G

All are assumed shuffle=True (every run we care about from this
investigation used ``--shuffle``).

Idempotent: runs are keyed by ``slurm_job_id``; re-running does not
duplicate entries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path


HISTORY_PATH = Path("benchmarks/results/profile_iter_history.json")


def infer_config(filename: str) -> dict:
    """Extract run config from a per-run profile JSON filename."""
    name = Path(filename).stem

    # profile_lance_<size>g[_nodecode]_<jobid>
    m = re.match(r"profile_lance_(\d+)g(?:_nodecode)?_(\d+)$", name)
    if m:
        mem_gb = int(m.group(1))
        slurm_id = m.group(2)
        decode = "nodecode" not in name
        return {
            "slurm_job_id": slurm_id,
            "config": {
                "mem_gb": mem_gb,
                "decode": decode,
                "shuffle": True,
            },
        }

    # profile_lance_cold_<jobid> (pre-rename, always 64G decode=True)
    m = re.match(r"profile_lance_cold_(\d+)$", name)
    if m:
        return {
            "slurm_job_id": m.group(1),
            "config": {"mem_gb": 64, "decode": True, "shuffle": True},
        }

    # profile_lance_matrix_<jobid> (256G, all datasets, decode=on, 1 run x 3 ep)
    m = re.match(r"profile_lance_matrix_(\d+)$", name)
    if m:
        return {
            "slurm_job_id": m.group(1),
            "config": {
                "mem_gb": 256,
                "decode": True,
                "shuffle": True,
                "num_runs": 1,
                "num_epochs": 3,
                "num_workers": 4,
                "label": "matrix",
            },
        }

    # profile_lance_small_<jobid> (256G, small only, 2 runs x 3 ep)
    m = re.match(r"profile_lance_small_(\d+)$", name)
    if m:
        return {
            "slurm_job_id": m.group(1),
            "config": {
                "mem_gb": 256,
                "decode": True,
                "shuffle": True,
                "num_runs": 2,
                "num_epochs": 3,
                "num_workers": 4,
                "label": "small",
            },
        }

    raise ValueError(f"Cannot infer config from filename {filename!r}")


def load_history(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    with path.open() as f:
        data = json.load(f)
    if "runs" not in data:
        raise ValueError(f"{path} exists but is not a history file (no 'runs' key)")
    return data


def save_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, path)


def append_run(
    history: dict,
    *,
    slurm_job_id: str,
    config: dict,
    results: dict,
    timestamp: str | None = None,
    source_file: str | None = None,
) -> bool:
    """Append a run entry to the history. Idempotent on slurm_job_id.
    Returns True if added, False if skipped as duplicate."""
    existing_ids = {r.get("slurm_job_id") for r in history["runs"]}
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
    history["runs"].append(entry)
    return True


def migrate(files: list[Path], history_path: Path) -> None:
    history = load_history(history_path)
    added = 0
    skipped = 0
    for f in files:
        try:
            parsed = infer_config(f.name)
        except ValueError as e:
            print(f"  skip {f.name}: {e}")
            skipped += 1
            continue
        with f.open() as fh:
            results = json.load(fh)
        ts = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        was_added = append_run(
            history,
            slurm_job_id=parsed["slurm_job_id"],
            config=parsed["config"],
            results=results,
            timestamp=ts,
            source_file=f.name,
        )
        if was_added:
            print(f"  added {f.name} (job {parsed['slurm_job_id']}, mem={parsed['config'].get('mem_gb')}G, decode={parsed['config'].get('decode')})")
            added += 1
        else:
            print(f"  skip {f.name} (duplicate job id {parsed['slurm_job_id']})")
            skipped += 1

    save_history(history_path, history)
    print(f"\nhistory at {history_path}: {len(history['runs'])} total runs ({added} added, {skipped} skipped)")


def print_summary(history_path: Path) -> None:
    """Print a plain-text matrix view of the history file.

    Groups runs by dataset and config (mem_gb, decode), and for each
    (dataset, config) shows the per-backend mean epoch time so Arrow /
    Lance / HF can be eyeballed across the memory and decode axes
    without parsing JSON by hand.
    """
    import statistics

    history = load_history(history_path)
    if not history["runs"]:
        print(f"{history_path}: empty history.")
        return

    # Flatten into (dataset, mem_gb, decode, backend) -> mean epoch time
    table: dict = {}
    datasets: set = set()
    backends: set = set()
    configs: set = set()
    for run in history["runs"]:
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
    for ds_name in sorted(datasets):
        rows = sorted(
            [cfg for cfg in configs if any((ds_name, *cfg, b) in table for b in backends)],
            key=lambda c: (-(c[0] or 0), c[1]),
        )
        if not rows:
            continue
        print(f"\n=== {ds_name} ===")
        header = f"{'mem':>6} {'decode':>8}  " + "".join(f"{b:>22}" for b in backend_order)
        print(header)
        print("-" * len(header))
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
            print(f"{mem_str:>6} {dec_str:>8}  " + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Per-run profile JSON files to merge. If empty, globs profile_lance_*.json in cwd.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=HISTORY_PATH,
        help=f"Consolidated history file (default: {HISTORY_PATH})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a plain-text summary of the history file and exit. "
        "Does not require any input files.",
    )
    args = parser.parse_args()

    if args.summary:
        print_summary(args.history)
        return

    files = args.files or sorted(Path(".").glob("profile_lance_*.json"))
    if not files:
        print("No per-run JSON files found.")
        return
    print(f"Migrating {len(files)} file(s) into {args.history}:")
    migrate(files, args.history)


if __name__ == "__main__":
    main()
