"""Report how much each run's linear probe has improved lately, to find converged runs.

The grid is capacity-bound, not time-bound: a run that stopped improving 200
epochs ago is holding a GPU that a queued cell could use. This surfaces those
so they can be cancelled deliberately, with the GPU-hours saved made explicit.

The tracked metric is ``eval/linear_probe_top1_epoch``, which since the probe
sweep landed is the BEST of the 9 (lr_scale x weight_decay) heads — so
"improvement" here means the best achievable linear probe got better, not that
one arbitrary probe setting did.

Two different questions get two different numbers, because probe top1 is noisy
and a single-point delta can be negative purely from run-to-run jitter:

  last_delta  latest value minus the value K epochs ago. Signed, jittery.
  best_gain   max over the last K epochs minus the best ever seen BEFORE that
              window. This is the convergence criterion: <= 0 means the run has
              not set a new high in K epochs, i.e. nothing was gained by the
              GPU-time spent.

A run is flagged CONVERGED when best_gain <= --threshold (default 0.0).

Usage:
    python -m benchmarks.check_converge                      # all running runs
    python -m benchmarks.check_converge --k 20
    python -m benchmarks.check_converge --dataset cifar10 --backbone vit
    python -m benchmarks.check_converge --method dino,lejepa --threshold 0.002
    python -m benchmarks.check_converge --all-states         # include finished
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import wandb

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None

ENTITY_PROJECT = os.environ.get("SDS_WANDB_ENTITY_PROJECT", "samibg/stable-datasets-iclr")
METRIC = "eval/linear_probe_top1_epoch"
METHODS = ["supervised", "simclr", "dino", "mae", "lejepa", "nnclr", "barlow_twins"]
_CONF = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")


def target_epochs(dataset: str, method: str) -> int | None:
    """Epoch budget for a cell, from conf/config.yaml's benchmark_epochs table.

    Not from run.config: W&B reports an EMPTY config for in-flight runs, so the
    remaining-hours estimate silently became "-" for every running job — which is
    exactly the column this script exists to produce.
    """
    if OmegaConf is None or not os.path.exists(_CONF):
        return None
    try:
        be = OmegaConf.load(_CONF).get("benchmark_epochs", None)
        if be is None or not be.get("enabled", False):
            return None
        base = (be.get("base", {}) or {}).get(dataset, None)
        if base is None:
            return None
        mult = int(be.get("mae_multiplier", 4)) if method == "mae" else 1
        return int(base) * mult
    except Exception:
        return None


def parse_name(name: str) -> tuple[str, str, str]:
    """run name -> (method, backbone, dataset). Names are
    ``{method}_{backbone}_{dataset}[_seed{n}]``, and both method and backbone
    contain underscores, so split on the known tokens rather than on '_'."""
    method = next((m for m in sorted(METHODS, key=len, reverse=True) if name.startswith(m + "_")), "?")
    rest = name[len(method) + 1:] if method != "?" else name
    backbone = "resnet50" if rest.startswith("resnet50") else ("vit" if rest.startswith("vit") else "?")
    dataset = rest.split("_", 1)[1] if "_" in rest else "?"
    if backbone == "vit":
        dataset = rest.replace("vit_small_patch16_224_", "", 1)
    elif backbone == "resnet50":
        dataset = rest.replace("resnet50_", "", 1)
    return method, backbone, dataset


def series(run) -> list[tuple[int, float]]:
    """(epoch, best-probe-top1) pairs, one per epoch, ascending."""
    out: dict[int, float] = {}
    try:
        for row in run.scan_history(keys=["epoch", METRIC], page_size=2000):
            e, v = row.get("epoch"), row.get(METRIC)
            if e is None or v is None:
                continue
            e = int(e)
            # keep the last value logged for an epoch
            out[e] = float(v)
    except Exception:
        return []
    return sorted(out.items())


def analyse(run, k: int, k_pct: float | None = None) -> dict | None:
    pts = series(run)
    if len(pts) < 2:
        return None
    method, backbone, dataset = parse_name(run.name)
    epochs = [e for e, _ in pts]
    vals = [v for _, v in pts]
    cur_ep, cur = pts[-1]
    if k_pct:
        # A fixed 10-epoch window is ~1% of an 800-epoch run: far too tight, so
        # ordinary epoch-to-epoch noise reads as "converged". Scaling the window
        # to the run's own length makes the verdict comparable across cells.
        t = target_epochs(dataset, method)
        k = max(5, int((t or cur_ep) * k_pct))

    # window = the last k epochs actually present in the history
    cutoff = cur_ep - k
    win = [v for e, v in pts if e > cutoff]
    pre = [v for e, v in pts if e <= cutoff]

    last_delta = cur - (pre[-1] if pre else vals[0])
    best_gain = (max(win) - max(pre)) if pre and win else float("nan")

    target = target_epochs(dataset, method)
    rt = run.summary.get("_runtime") or 0
    spe = rt / (cur_ep + 1) if cur_ep >= 0 else 0
    remaining_h = ((target - cur_ep - 1) * spe / 3600) if target else None

    return dict(
        name=run.name, method=method, backbone=backbone, dataset=dataset,
        epoch=cur_ep, target=target, k_used=k, cur=cur, best=max(vals),
        last_delta=last_delta, best_gain=best_gain,
        remaining_h=remaining_h, n_points=len(pts),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=10, help="window of epochs to measure improvement over (default 10)")
    p.add_argument("--k-pct", type=float, default=None,
                   help="instead use a window of this FRACTION of the cell's epoch budget, "
                        "e.g. 0.05. Recommended: a fixed k=10 is ~1%% of an 800-epoch run and "
                        "mostly measures noise.")
    p.add_argument("--dataset", default=None, help="comma-separated filter")
    p.add_argument("--method", default=None, help="comma-separated filter")
    p.add_argument("--backbone", default=None, help="'vit' or 'resnet50'")
    p.add_argument("--threshold", type=float, default=0.0, help="flag CONVERGED when best_gain <= this (default 0.0)")
    p.add_argument("--all-states", action="store_true", help="include finished/crashed runs, not just running")
    p.add_argument("--project", default=ENTITY_PROJECT)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    want = lambda s, f: (f is None) or (s in [x.strip() for x in f.split(",")])

    api = wandb.Api(timeout=90)
    runs = [r for r in api.runs(a.project, per_page=300) if a.all_states or r.state == "running"]
    sel = []
    for r in runs:
        m, b, d = parse_name(r.name)
        if want(d, a.dataset) and want(m, a.method) and want(b, a.backbone):
            sel.append(r)
    if not sel:
        print("no runs matched"); sys.exit(0)

    win_desc = (f"{a.k_pct:.0%} of each cell's epoch budget (per-run, see the k column)"
                if a.k_pct else f"last {a.k} epochs")
    print(f"{len(sel)} run(s); improvement window = {win_desc}; metric = {METRIC}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [x for x in ex.map(lambda r: analyse(r, a.k, a.k_pct), sel) if x]

    rows.sort(key=lambda r: (r["best_gain"] if r["best_gain"] == r["best_gain"] else 1e9))

    hdr = (f"{'method':<13}{'bb':<9}{'dataset':<12}{'epoch':>11}{'best':>8}"
           f"{'last_d':>9}{'gain@k':>9}{'k':>5}{'left_h':>8}  flag")
    print(hdr); print("-" * len(hdr))
    saved = 0.0
    for r in rows:
        conv = r["best_gain"] == r["best_gain"] and r["best_gain"] <= a.threshold
        if conv and r["remaining_h"]:
            saved += r["remaining_h"]
        ep = f"{r['epoch']}/{r['target']}" if r["target"] else str(r["epoch"])
        lh = f"{r['remaining_h']:.1f}" if r["remaining_h"] is not None else "-"
        print(f"{r['method']:<13}{r['backbone']:<9}{r['dataset']:<12}{ep:>11}"
              f"{r['best']:>8.4f}{r['last_delta']:>+9.4f}{r['best_gain']:>+9.4f}{r['k_used']:>5}{lh:>8}"
              f"  {'CONVERGED' if conv else ''}")
    n_conv = sum(1 for r in rows if r["best_gain"] == r["best_gain"] and r["best_gain"] <= a.threshold)
    print()
    print(f"{n_conv}/{len(rows)} runs set no new best within their window "
          f"({win_desc}; threshold {a.threshold:+.4f})")
    if saved:
        print(f"cancelling those would free ~{saved:,.0f} GPU-hours")
        print("cancel with:  scontrol update jobid=<id> requeue=0 && scancel --signal=KILL <id> && scancel <id>")


if __name__ == "__main__":
    main()
