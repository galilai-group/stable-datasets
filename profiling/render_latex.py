"""Render profiling results into a LaTeX table.

Combines prep-time and iteration-time USS results into one
LaTeX table comparing stable-datasets vs HuggingFace Datasets.

Usage:
    python profiling/render_latex.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"
PREP_FILE = RESULTS_DIR / "benchmark_prep_uss_results.json"
ITER_FILE = RESULTS_DIR / "benchmark_iter_uss_results.json"
OUTPUT_FILE = RESULTS_DIR / "profiling_table.tex"

BACKEND_LABELS = {
    "stable": r"\textbf{stable-datasets}",
    "hf": r"\textbf{HF Datasets}",
}


def _fmt_time(seconds: float) -> str:
    """Format seconds for LaTeX (e.g. '3.65\\,s' or '698\\,ms')."""
    if seconds < 1.0:
        return rf"{seconds * 1000:.0f}\,ms"
    if seconds < 60:
        return rf"{seconds:.2f}\,s"
    return rf"{seconds / 60:.1f}\,min"


def _fmt_mem(bytes_val: float) -> str:
    """Format bytes as MB or GB."""
    mb = bytes_val / 1024 / 1024
    if mb < 1024:
        return rf"{mb:.0f}\,MB"
    return rf"{mb / 1024:.2f}\,GB"


def _bold_if_best(val_str: str, val: float, other_val: float | None) -> str:
    """Wrap in \\textbf{} if val is the smaller (better) of the two."""
    if other_val is None or val < other_val:
        return rf"\textbf{{{val_str}}}"
    return val_str


def _load_prep(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_iter(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_prep(data: dict, ds_name: str, backend: str) -> tuple[float | None, float | None]:
    """Return (prep_time, uss_bytes) or (None, None)."""
    r = data.get(ds_name, {}).get(backend)
    if r is None or "error" in r:
        return None, None
    return r.get("prep"), r.get("uss")


def _get_iter(data: dict, ds_name: str, backend: str) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (mean_epoch_time, std, uss_total, rss_total) or all Nones."""
    r = data.get(ds_name, {}).get(backend)
    if r is None or "error" in r or r.get("read") is None:
        return None, None, None, None
    times = r["read"]
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    mem_list = r.get("memory", [])
    uss_total = mem_list[0]["uss_total"] if mem_list else None
    rss_total = mem_list[0]["rss_total"] if mem_list else None
    return mean_t, std_t, uss_total, rss_total


_SKIP_DATASETS = {"Beans"}  # profiling errors on this one


def render(prep_data: dict, iter_data: dict) -> str:
    # Dataset order: intersection of both files, preserved, skip excluded
    ds_names = [d for d in prep_data if d in iter_data and d not in _SKIP_DATASETS]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Profiling results: stable-datasets vs HuggingFace Datasets. "
        r"Prep time is cold-cache cache-build time (raw downloads pre-warmed). "
        r"Read time is mean~$\pm$~std per epoch over 5 epochs with "
        r"\texttt{batch\_size=128}, \texttt{num\_workers=4}. "
        r"Memory is USS (unique set size, excludes shared mmap pages).}",
        r"\label{tab:profiling}",
        r"\begin{tabular}{l" + "cc" * 3 + r"}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Prep time}} "
        r"& \multicolumn{2}{c}{\textbf{Read time / epoch}} "
        r"& \multicolumn{2}{c}{\textbf{USS (iter)}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"\textbf{Dataset} & stable & HF & stable & HF & stable & HF \\",
        r"\midrule",
    ]

    for ds in ds_names:
        prep_s, uss_s_prep = _get_prep(prep_data, ds, "stable")
        prep_h, uss_h_prep = _get_prep(prep_data, ds, "hf")

        iter_s_mean, iter_s_std, uss_s_iter, _ = _get_iter(iter_data, ds, "stable")
        iter_h_mean, iter_h_std, uss_h_iter, _ = _get_iter(iter_data, ds, "hf")

        row = [ds]

        # Prep times
        if prep_s is not None and prep_h is not None:
            row.append(_bold_if_best(_fmt_time(prep_s), prep_s, prep_h))
            row.append(_bold_if_best(_fmt_time(prep_h), prep_h, prep_s))
        else:
            row.append(_fmt_time(prep_s) if prep_s is not None else "---")
            row.append(_fmt_time(prep_h) if prep_h is not None else "---")

        # Iter times (show as mean ± std)
        def _fmt_iter(mean, std):
            if mean is None:
                return "---"
            return rf"${_fmt_time(mean)} \pm {_fmt_time(std)}$"

        if iter_s_mean is not None and iter_h_mean is not None:
            row.append(_bold_if_best(_fmt_iter(iter_s_mean, iter_s_std), iter_s_mean, iter_h_mean))
            row.append(_bold_if_best(_fmt_iter(iter_h_mean, iter_h_std), iter_h_mean, iter_s_mean))
        else:
            row.append(_fmt_iter(iter_s_mean, iter_s_std))
            row.append(_fmt_iter(iter_h_mean, iter_h_std))

        # Iter USS
        if uss_s_iter is not None and uss_h_iter is not None:
            row.append(_bold_if_best(_fmt_mem(uss_s_iter), uss_s_iter, uss_h_iter))
            row.append(_bold_if_best(_fmt_mem(uss_h_iter), uss_h_iter, uss_s_iter))
        else:
            row.append(_fmt_mem(uss_s_iter) if uss_s_iter is not None else "---")
            row.append(_fmt_mem(uss_h_iter) if uss_h_iter is not None else "---")

        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main():
    if not PREP_FILE.exists():
        print(f"Missing: {PREP_FILE}")
        print("Run profiling/profile_prep.py first.")
        return 1
    if not ITER_FILE.exists():
        print(f"Missing: {ITER_FILE}")
        print("Run profiling/profile_iter.py first.")
        return 1

    prep_data = _load_prep(PREP_FILE)
    iter_data = _load_iter(ITER_FILE)

    table_tex = render(prep_data, iter_data)
    OUTPUT_FILE.write_text(table_tex)
    print(f"Wrote {OUTPUT_FILE}")
    print()
    print(table_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
