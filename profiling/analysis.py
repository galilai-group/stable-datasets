#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKEND_ORDER = ["hf", "pyarrow", "lance"]
BACKEND_LABELS = {
    "hf": "HF Datasets",
    "pyarrow": "stable-datasets Arrow",
    "lance": "stable-datasets Lance",
}
PARALLELISM_ORDER = [0, 1, 4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-facing tables and figures from benchmark CSVs.")
    parser.add_argument("--decode-on-path", default="profiling/results/raw_runs_decode_on_long_workers.csv")
    parser.add_argument("--decode-off-path", default="profiling/results/raw_runs_decode_off_long_workers.csv")
    parser.add_argument("--lance-p32-path", default="profiling/results/lance_p32_decode_on_256g.csv")
    parser.add_argument("--output-dir", default="profiling/results/final_results")
    return parser.parse_args()


def load_csv(path: str):
    import pandas as pd

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Results file is empty: {csv_path}")
    return df


def aggregate_runs(df):
    group_cols = [
        "dataset",
        "backend",
        "regime",
        "decode_mode",
        "experiment_name",
        "parallelism_level",
        "num_workers",
        "lance_cpu_threads",
        "lance_io_threads",
        "batch_size",
        "warmup_batches",
        "measured_batches",
        "subset_fraction",
        "seed",
    ]
    aggregated = (
        df.groupby(group_cols, dropna=False)
        .agg(
            replicates=("replicate_id", "nunique"),
            total_images_mean=("total_images", "mean"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
            images_per_sec_mean=("images_per_sec", "mean"),
            images_per_sec_std=("images_per_sec", "std"),
            peak_total_host_uss_bytes_mean=("peak_total_host_uss_bytes", "mean"),
            peak_total_host_uss_bytes_std=("peak_total_host_uss_bytes", "std"),
            time_to_first_batch_sec_mean=("time_to_first_batch_sec", "mean"),
        )
        .reset_index()
    )
    aggregated["host_uss_gb_mean"] = aggregated["peak_total_host_uss_bytes_mean"] / (1024**3)
    aggregated["images_per_sec_std"] = aggregated["images_per_sec_std"].fillna(0.0)
    aggregated["host_uss_gb_std"] = aggregated["peak_total_host_uss_bytes_std"].fillna(0.0) / (1024**3)
    return aggregated


def merge_decode_on_with_lance_p32(decode_on_df, lance_p32_df):
    import pandas as pd

    base = decode_on_df.copy()
    mask = (
        (base["backend"] == "lance")
        & (base["decode_mode"] == "on")
        & (base["parallelism_level"] == 32)
        & (base["regime"].isin(["training", "sparse"]))
    )
    base = base.loc[~mask].copy()

    patch = lance_p32_df.copy()
    patch["decode_mode"] = "on"
    patch["parallelism_level"] = 32
    patch["experiment_name"] = "scaling-p32-256g"
    patch["annotation"] = "256G rerun"

    if "annotation" not in base.columns:
        base["annotation"] = ""
    patch["annotation"] = patch["annotation"].fillna("256G rerun")

    return pd.concat([base, patch], ignore_index=True, sort=False)


def render_training_curve_figure(decode_on_agg, decode_off_agg, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    panels = [("on", decode_on_agg, "Decode On"), ("off", decode_off_agg, "Decode Off")]

    for axis, (_, agg, title) in zip(axes, panels, strict=False):
        regime_df = agg[agg["regime"] == "training"].copy()
        for backend in BACKEND_ORDER:
            group = regime_df[regime_df["backend"] == backend].copy()
            if group.empty:
                continue
            group["parallelism_level"] = group["parallelism_level"].astype(int)
            group = group.sort_values("parallelism_level")

            axis.plot(
                group["parallelism_level"],
                group["images_per_sec_mean"],
                marker="o",
                linewidth=2.0,
                label=BACKEND_LABELS[backend],
            )
            axis.fill_between(
                group["parallelism_level"],
                group["images_per_sec_mean"] - group["images_per_sec_std"],
                group["images_per_sec_mean"] + group["images_per_sec_std"],
                alpha=0.15,
            )

            if backend == "lance" and "annotation" in group.columns:
                annotated = group[group["annotation"] == "256G rerun"]
                for _, row in annotated.iterrows():
                    axis.annotate(
                        "256G",
                        (row["parallelism_level"], row["images_per_sec_mean"]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=8,
                    )

        axis.set_title(title)
        axis.set_xlabel("Parallelism")
        axis.set_xticks(PARALLELISM_ORDER)
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("Images / sec")
    axes[1].legend()
    fig.suptitle("ImageNet-1K Training Throughput")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)


def write_curve_table(agg, output_path: Path, *, regime: str, decode_mode: str) -> None:
    import pandas as pd

    table = agg[(agg["regime"] == regime) & (agg["decode_mode"] == decode_mode)].copy()
    if table.empty:
        raise ValueError(f"No rows found for regime={regime!r}, decode_mode={decode_mode!r}")

    rows = []
    for backend in BACKEND_ORDER:
        group = table[table["backend"] == backend].copy()
        row = {"backend": BACKEND_LABELS[backend]}
        for parallelism in PARALLELISM_ORDER:
            match = group[group["parallelism_level"].astype(int) == parallelism]
            if match.empty:
                row[f"p{parallelism}"] = "OOM"
            else:
                value = match.iloc[0]["images_per_sec_mean"]
                note = match.iloc[0].get("annotation", "")
                cell = f"{value:.1f}"
                if note:
                    cell = f"{cell} ({note})"
                row[f"p{parallelism}"] = cell
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_best_point_table(decode_on_agg, decode_off_agg, output_path: Path) -> None:
    import pandas as pd

    rows = []
    for decode_mode, agg in [("on", decode_on_agg), ("off", decode_off_agg)]:
        subset = agg[agg["regime"] == "training"].copy()
        for backend in BACKEND_ORDER:
            group = subset[subset["backend"] == backend].copy()
            if group.empty:
                continue
            best = group.sort_values("images_per_sec_mean", ascending=False).iloc[0]
            rows.append(
                {
                    "backend": BACKEND_LABELS[backend],
                    "decode_mode": decode_mode,
                    "best_images_per_sec": round(float(best["images_per_sec_mean"]), 1),
                    "parallelism": int(best["parallelism_level"]),
                    "peak_host_uss_gb": round(float(best["host_uss_gb_mean"]), 2),
                    "replicates": int(best["replicates"]),
                    "note": best.get("annotation", ""),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_environment_table(output_path: Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                "field,value",
                "os,Linux 5.14.0-570.62.1.0.1.el9_6.x86_64",
                "cpu,2 x Intel Xeon Platinum 8268 @ 2.90GHz",
                "logical_cpus,48",
                "visible_ram,376 GiB",
                "filesystem,nfs",
                "nfs_version,3",
                "transport,rdma",
                "nconnect,8",
                "mount,hpcnfs:/oscar/home/sboughan/stable-datasets-pyarrow",
            ]
        )
        + "\n"
    )


def write_profile_summary(output_path: Path) -> None:
    import pandas as pd

    base = Path("profiling/results/profiles_delayed")
    rows = []
    for json_path in sorted(base.glob("*.stages.json")):
        payload = json.loads(json_path.read_text())
        meta = payload["metadata"]
        stages = payload["stages"]
        if meta["regime"] != "training":
            continue
        rows.append(
            {
                "backend": BACKEND_LABELS[meta["backend"]],
                "decode_mode": meta["decode_mode"],
                "source_fetch_total_sec": round(float(stages["source_fetch_total_sec"]), 3),
                "backend_take_total_sec": round(float(stages["backend_take_total_sec"]), 3),
                "format_batch_total_sec": round(float(stages["format_batch_total_sec"]), 3),
                "normalize_total_sec": round(float(stages["normalize_total_sec"]), 3),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decode_on_raw = load_csv(args.decode_on_path)
    decode_off_raw = load_csv(args.decode_off_path)
    lance_p32_raw = load_csv(args.lance_p32_path)

    decode_on_merged = merge_decode_on_with_lance_p32(decode_on_raw, lance_p32_raw)
    decode_on_agg = aggregate_runs(decode_on_merged)
    decode_off_agg = aggregate_runs(decode_off_raw)

    decode_on_agg.to_csv(output_dir / "aggregated_decode_on.csv", index=False)
    decode_off_agg.to_csv(output_dir / "aggregated_decode_off.csv", index=False)

    render_training_curve_figure(
        decode_on_agg,
        decode_off_agg,
        output_dir / "figure_training_decode_on_off.png",
    )
    write_curve_table(
        decode_on_agg,
        output_dir / "table_training_decode_on.csv",
        regime="training",
        decode_mode="on",
    )
    write_curve_table(
        decode_off_agg,
        output_dir / "table_training_decode_off.csv",
        regime="training",
        decode_mode="off",
    )
    write_best_point_table(decode_on_agg, decode_off_agg, output_dir / "table_best_points.csv")
    write_environment_table(output_dir / "table_environment.csv")
    write_profile_summary(output_dir / "table_profile_summary.csv")


if __name__ == "__main__":
    main()
