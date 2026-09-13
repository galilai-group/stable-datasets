"""Report, for each resumed ViT pretrain checkpoint that has reached max_epochs:
the linear-probe at the resumed-from epoch (started) vs final, the delta, and the
gap to the paper number (benchmarks/results/benchmark_table_probe_paper.tex).

Reads each staged checkpoint (.pretrain_checkpoints) for its epoch + embedded W&B
run id, then pulls that run's linear-probe history. Writes resume_progress.csv.

    python -m benchmarks.resume_progress_report
"""
from __future__ import annotations
import csv, glob, os, re, time
import torch
import wandb
from benchmarks.render_latex import _display_name, MODEL_DISPLAY_NAMES

ENTITY_PROJECT = os.environ.get("SDS_WANDB_ENTITY_PROJECT", "samibg/stable-datasets-iclr")

CK = os.path.join(
    os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")),
    "checkpoints",
)
SWEEP = "/oscar/scratch/sboughan/epoch_sweep.csv"
PAPER = "benchmarks/results/benchmark_table_probe_paper.tex"
OUT = "/oscar/home/sboughan/stable-datasets-pyarrow/resume_progress.csv"
BB = "vit_small_patch16_224"
LP = "eval/linear_probe_top1_epoch"

def _retry(fn, t=6, w=8):
    for i in range(t):
        try: return fn()
        except Exception as e:
            if i == t-1: raise
            time.sleep(w)

# --- resumed-from epoch + max per (method,dataset) ---
sweep = {}
for r in csv.reader(open(SWEEP)):
    if len(r) < 6 or r[0] == "method": continue
    try: sweep[(r[0], r[1])] = (int(r[2]), int(r[3]), r[4])
    except: pass
keys = sorted({d for (_, d) in sweep})

# --- paper numbers (mean of each substack cell) ---
disp2key = {_display_name(k): k for k in keys}
PAPER_COLS = ["simclr", "nnclr", "dino", "barlow_twins", "lejepa", "mae", "supervised"]
paper = {}
for line in open(PAPER):
    if "substack" not in line: continue
    cells = line.split("&")
    dkey = disp2key.get(cells[0].strip())
    if not dkey: continue
    for i, m in enumerate(PAPER_COLS):
        if i + 1 >= len(cells): break
        mm = re.search(r"([0-9]+\.[0-9]+)", cells[i+1].replace("textbf", ""))
        if mm: paper[(m, dkey)] = float(mm.group(1))
print(f"paper cells parsed: {len(paper)}", flush=True)

api = wandb.Api(timeout=90)
hist_cache = {}
def run_hist(rid):
    if rid in hist_cache: return hist_cache[rid]
    try:
        run = _retry(lambda: api.run(f"{ENTITY_PROJECT}/{rid}"))
        h = _retry(lambda: run.history(keys=[LP, "epoch"], pandas=True))
        hist_cache[rid] = h
    except Exception:
        hist_cache[rid] = None
    return hist_cache[rid]

rows = []
for (method, dataset), (start_ep, maxep, status) in sorted(sweep.items()):
    if status != "EARLY": continue
    d = f"{CK}/{method}_{BB}_{dataset}"
    # Lightning writes the FINAL checkpoint as last-v1.ckpt (it won't overwrite the
    # pre-existing staged last.ckpt) or epoch=N-*.ckpt. Pick the most-advanced.
    cands = []
    if os.path.isfile(f"{d}/last-v1.ckpt"): cands.append(f"{d}/last-v1.ckpt")
    for p in glob.glob(f"{d}/epoch=*.ckpt"):
        cands.append(p)
    if os.path.isfile(f"{d}/last.ckpt"): cands.append(f"{d}/last.ckpt")
    if not cands: continue
    def _ep_of(p):
        m = re.search(r"epoch=(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    # prefer highest epoch=*.ckpt; last-v1/last need a load to know their epoch
    ecks = [(p, _ep_of(p)) for p in cands if _ep_of(p) >= 0]
    f = max(ecks, key=lambda x: x[1])[0] if ecks else (f"{d}/last-v1.ckpt" if os.path.isfile(f"{d}/last-v1.ckpt") else f"{d}/last.ckpt")
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception:
        continue
    now = ck.get("epoch"); wb = ck.get("wandb") or {}
    rid = wb.get("id")
    completed = isinstance(now, int) and now >= maxep - 1
    if not completed or not rid:
        del ck; continue
    del ck
    h = run_hist(rid)
    started = final = None
    if h is not None and LP in h:
        s = h.dropna(subset=[LP])
        if len(s):
            final = float(s[LP].iloc[-1]) * 100
            at = s[s["epoch"] <= start_ep] if "epoch" in s else s
            if len(at): started = float(at[LP].iloc[-1]) * 100
    pap = paper.get((method, dataset))
    rows.append((method, dataset, start_ep, now, started, final, pap))

# --- write + summarize ---
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "dataset", "resumed_from_ep", "final_ep", "started_lp", "final_lp", "delta_lp", "paper_lp", "gap_to_paper"])
    for m, d, se, ne, st, fi, pp in rows:
        delta = (fi - st) if (st is not None and fi is not None) else None
        gap = (fi - pp) if (fi is not None and pp is not None) else None
        w.writerow([m, d, se, ne, f"{st:.1f}" if st is not None else "",
                    f"{fi:.1f}" if fi is not None else "", f"{delta:+.1f}" if delta is not None else "",
                    f"{pp:.1f}" if pp is not None else "", f"{gap:+.1f}" if gap is not None else ""])
print(f"\nwrote {OUT} with {len(rows)} completed resumes", flush=True)
import statistics
d = [ (fi-st) for _,_,_,_,st,fi,_ in rows if st is not None and fi is not None ]
g = [ (fi-pp) for _,_,_,_,_,fi,pp in rows if fi is not None and pp is not None ]
if d: print(f"delta (final-started): mean={statistics.mean(d):+.2f}  min={min(d):+.1f} max={max(d):+.1f}")
if g: print(f"gap to paper (final-paper): mean={statistics.mean(g):+.2f}  within 1pt: {sum(1 for x in g if abs(x)<=1)}/{len(g)}  within 2pt: {sum(1 for x in g if abs(x)<=2)}/{len(g)}")
print("\n=== sample (method dataset start->final  delta  paper  gap) ===")
for m,d2,se,ne,st,fi,pp in rows[:25]:
    S=f"{st:.1f}" if st is not None else "-"; F=f"{fi:.1f}" if fi is not None else "-"
    D=f"{fi-st:+.1f}" if (st is not None and fi is not None) else "-"
    P=f"{pp:.1f}" if pp is not None else "-"; G=f"{fi-pp:+.1f}" if (fi is not None and pp is not None) else "-"
    print(f"  {m:13} {d2:18} {S:>5}->{F:>5}  d={D:>6}  paper={P:>5}  gap={G:>6}")
