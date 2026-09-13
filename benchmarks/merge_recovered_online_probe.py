"""Merge per-cell online-probe recovery files into one CSV, and audit coverage.

Reads every ``*.csv`` in RECOVER_DIR (each is header + one row written by
``recover_online_probe.py``) and concatenates them into a single CSV, keyed by
(method, dataset, seed, source). Reports counts by source, any duplicates, any
missing-checkpoint cells, and (if the manifest is given) which manifest cells are
still absent — so you can see coverage at a glance.

    python -m benchmarks.merge_recovered_online_probe \
        [RECOVER_DIR] [OUT.csv] [recover_manifest.txt]
"""

from __future__ import annotations

import collections
import csv
import glob
import os
import sys

DEFAULT_DIR = os.path.join(
    os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")),
    "online_probe_recovered",
)
# LiDAR dropped per Sami: it samples surrogate classes each call (not reproducible
# from a checkpoint), and rankme already covers representation geometry bit-exactly.
# Per-cell files still carry lidar/lidar_entropy columns; the merge simply omits them.
HEADER = [
    "dataset", "method", "seed", "origin",
    "linear_top1", "linear_top5", "knn_top1", "knn_top5", "rankme",
    "rankme_condition_number", "rankme_entropy",
    "epoch", "max_epochs", "num_classes", "backbone", "checkpoint_or_run",
]
KEY = ("dataset", "method", "seed")
ORIGIN = "origin"


def main() -> None:
    rec_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rec_dir, "online_probe_recovered.csv")
    manifest = sys.argv[3] if len(sys.argv) > 3 else None

    rows = []
    seen = collections.Counter()
    for f in sorted(glob.glob(os.path.join(rec_dir, "*__*.csv"))):
        with open(f, newline="") as fh:
            for d in csv.DictReader(fh):
                rows.append(d)
                seen[tuple(d.get(k, "") for k in KEY)] += 1

    rows.sort(key=lambda d: tuple(d.get(k, "") for k in KEY))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        for d in rows:
            w.writerow({k: d.get(k, "") for k in HEADER})

    by_origin = collections.Counter(d.get(ORIGIN, "") for d in rows)
    dups = {k: v for k, v in seen.items() if v > 1}
    need_wandb = [d for d in rows if d.get(ORIGIN) in ("missing_checkpoint", "incomplete")]

    print(f"merged {len(rows)} rows -> {out_csv}")
    print("by origin:", dict(by_origin))
    print("no duplicate keys" if not dups else f"!! {len(dups)} duplicate (dataset,method,seed) keys: {list(dups)[:10]}")
    if need_wandb:
        print(f"!! {len(need_wandb)} cells not recovered from a final checkpoint "
              f"(missing/incomplete): {[(d['dataset'], d['method'], d['seed']) for d in need_wandb][:10]}")

    if manifest and os.path.isfile(manifest):
        want = set()
        for line in open(manifest):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0]:
                # manifest is method<TAB>dataset<TAB>seed; key order is (dataset, method, seed)
                want.add((parts[1], parts[0], parts[2] if len(parts) > 2 else ""))
        have = {tuple(d.get(k, "") for k in KEY) for d in rows if d.get(ORIGIN) == "online_rerun"}
        absent = sorted(want - have)
        print(f"manifest coverage: {len(have)}/{len(want)} recovered (online_rerun); {len(absent)} absent")
        if absent:
            print("  first absent:", absent[:10])


if __name__ == "__main__":
    main()
