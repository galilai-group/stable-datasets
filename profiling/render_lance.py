"""Render the Lance-migration investigation as a single LaTeX table.

Reads ``benchmarks/results/profile_iter_history.json`` (the consolidated
profile-iter history) and emits a single table covering both the
cross-dataset modality sweep and the ImageNet-1K cache-pressure +
decode-isolation study. The table body is split by ``\\midrule`` into
two blocks:

    Block A -- small/medium datasets, warm 256G, decode on
    Block B -- ImageNet-1K variants across memory tiers and decode state

Small datasets fit warm everywhere on our nodes, so one row each is
enough. ImageNet-1K is the only dataset that stresses the page cache,
so we give it the full memory sweep plus the decode=off row that
exposes Lance's storage-layer advantage once PIL decode is stripped.

Usage:
    python profiling/render_lance.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


HISTORY_FILE = Path(__file__).parent.parent / "benchmarks" / "results" / "profile_iter_history.json"
OUTPUT_FILE = Path(__file__).parent.parent / "benchmarks" / "results" / "lance_investigation_table.tex"

BACKEND_ORDER = ("stable", "stable_lance", "hf")
BACKEND_LABELS = {
    "stable": r"\textbf{stable-datasets}",
    "stable_lance": r"\textbf{Lance}",
    "hf": r"\textbf{HF Datasets}",
}

SMALL_DATASETS = (
    "CIFAR-10",
    "CIFAR-100",
    "FashionMNIST",
    "SVHN",
    "STL-10",
    "Flowers102",
)

# (row label, mem_gb, decode) for each ImageNet-1K variant row.
IMAGENET_VARIANTS = (
    (r"ImageNet-1K (256\,GB)", 256, True),
    (r"ImageNet-1K (64\,GB)", 64, True),
    (r"ImageNet-1K (32\,GB)", 32, True),
    (r"ImageNet-1K (8\,GB)", 8, True),
    (r"ImageNet-1K (64\,GB, no decode)", 64, False),
)


# -- Formatting helpers -------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return rf"{seconds * 1000:.0f}\,ms"
    if seconds < 60:
        return rf"{seconds:.2f}\,s"
    return rf"{seconds / 60:.1f}\,min"


def _fmt_mean_std(mean: float, std: float) -> str:
    return rf"${_fmt_time(mean)} \pm {_fmt_time(std)}$"


def _bold_if_best(cell: str, val: float, others: list[float | None]) -> str:
    valid = [o for o in others if o is not None]
    if not valid or val <= min(valid):
        return rf"\textbf{{{cell}}}"
    return cell


# -- History filtering and aggregation ---------------------------------------


def _pool_epoch_times(
    history: dict, *, dataset: str, backend: str, mem_gb: int | None, decode: bool
) -> list[float]:
    pool: list[float] = []
    for run in history["runs"]:
        cfg = run.get("config", {})
        if mem_gb is not None and cfg.get("mem_gb") != mem_gb:
            continue
        if cfg.get("decode", True) != decode:
            continue
        per_backend = run.get("results", {}).get(dataset, {})
        entry = per_backend.get(backend)
        if entry is None:
            continue
        pool.extend(entry.get("read", []))
    return pool


def _stat(pool: list[float]) -> tuple[float, float] | None:
    if not pool:
        return None
    mean = statistics.mean(pool)
    std = statistics.stdev(pool) if len(pool) > 1 else 0.0
    return mean, std


def _active_backends(history: dict) -> list[str]:
    """Return the backends that actually appear in at least one row."""
    active = []
    for b in BACKEND_ORDER:
        if any(_pool_epoch_times(history, dataset=d, backend=b, mem_gb=256, decode=True)
               for d in SMALL_DATASETS):
            active.append(b)
            continue
        for _, mem, decode in IMAGENET_VARIANTS:
            if _pool_epoch_times(history, dataset="ImageNet-1K", backend=b, mem_gb=mem, decode=decode):
                active.append(b)
                break
    return active


def _row(label: str, history: dict, dataset: str, mem_gb: int, decode: bool, backends: list[str]) -> str | None:
    stats = [
        _stat(_pool_epoch_times(history, dataset=dataset, backend=b, mem_gb=mem_gb, decode=decode))
        for b in backends
    ]
    if all(s is None for s in stats):
        return None
    means = [s[0] if s else None for s in stats]
    cells = []
    for i, s in enumerate(stats):
        if s is None:
            cells.append("---")
        else:
            mean, std = s
            others = [m for j, m in enumerate(means) if j != i]
            cells.append(_bold_if_best(_fmt_mean_std(mean, std), mean, others))
    return f"{label} & " + " & ".join(cells) + r" \\"


# -- Table assembly ----------------------------------------------------------


def render_table(history: dict) -> str:
    backends = _active_backends(history)
    col_spec = "l" + "c" * len(backends)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Profiling evaluation across backends. "
        r"Per-epoch read time through \texttt{torch.utils.data.DataLoader} "
        r"with \texttt{shuffle=True}, \texttt{batch\_size=128}, "
        r"\texttt{num\_workers=4}, on a Slurm compute node with the "
        r"indicated memory cgroup limit. Values are "
        r"mean~$\pm$~std over all available epochs; best-per-row in "
        r"\textbf{bold}. "
        r"The upper block reports small/medium datasets at 256\,GB "
        r"(warm cache, dataset fits entirely in page cache). "
        r"The lower block reports ImageNet-1K (147\,GB) across "
        r"memory tiers that constrain how much of the dataset the "
        r"page cache can hold, plus one row (\textit{no decode}) "
        r"where PIL JPEG decoding is disabled via "
        r"\texttt{StableDataset.set\_decode(False)} to isolate the "
        r"storage-layer cost from the decode-bound cost that "
        r"otherwise dominates every row above it.}",
        r"\label{tab:lance-investigation}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join(BACKEND_LABELS[b] for b in backends) + r" \\",
        r"\midrule",
    ]

    for ds in SMALL_DATASETS:
        row = _row(ds, history, ds, 256, True, backends)
        if row:
            lines.append(row)

    lines.append(r"\midrule")

    for label, mem, decode in IMAGENET_VARIANTS:
        row = _row(label, history, "ImageNet-1K", mem, decode, backends)
        if row:
            lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    if not HISTORY_FILE.exists():
        print(f"Missing: {HISTORY_FILE}")
        print("Run profile_iter.py or experiment_manager.py first.")
        return 1

    with HISTORY_FILE.open() as f:
        history = json.load(f)

    table_tex = render_table(history)
    OUTPUT_FILE.write_text(table_tex)
    print(f"Wrote {OUTPUT_FILE}\n")
    print(table_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
