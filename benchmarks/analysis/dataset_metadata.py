"""Compute dataset-level metadata features for the 19 benchmark datasets.

For each dataset, computes:
  - train_size               # samples in the training split
  - num_classes              # label cardinality
  - class_balance            # normalized entropy of label distribution in [0,1]
                             #   1 = perfectly balanced, lower = more imbalanced
  - imbalance_ratio          # max_class_count / min_class_count
  - mean_height, mean_width  # native image resolution, averaged over a sample
                             # of 16 training images per dataset (before any
                             # resize). Grayscale images report 1 channel.
  - mean_channels            # 1 for grayscale datasets, 3 for RGB
  - mean_pixels              # mean(height * width) — rough "information per
                             # image" proxy independent of aspect ratio

Writes benchmarks/results/dataset_metadata.csv and prints a summary table of
correlations between each metadata feature and both (a) `ssl_advantage` and
(b) `intraclass_variation_dinov2`, so we can see directly whether any simple
counting-based metadata matches what the DINOv2 measurement captures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.dataset import DATASET_CONFIGS, INCLUDED_IMAGE_DATASETS, DatasetConfig, _get_dataset_class


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "results"
OUT_CSV = OUT_DIR / "dataset_metadata.csv"

DATA_ROOT = Path("/oscar/home/sboughan/scratch/.stable-datasets")
DATA_KWARGS = {
    "download_dir": str(DATA_ROOT / "downloads"),
    "processed_cache_dir": str(DATA_ROOT / "processed"),
}

DATASETS: dict[str, DatasetConfig] = {
    name: cfg for name, cfg in DATASET_CONFIGS.items() if name in INCLUDED_IMAGE_DATASETS
}

N_RESOLUTION_SAMPLES = 16


def get_labels(ds) -> np.ndarray:
    try:
        return np.asarray(ds._table.column("label").to_numpy(zero_copy_only=False))
    except Exception:
        return np.array([int(ds[i]["label"]) for i in range(len(ds))])


def image_hwc(img) -> tuple[int, int, int]:
    # PIL path
    if hasattr(img, "size") and hasattr(img, "getbands"):
        w, h = img.size
        c = len(img.getbands())
        return h, w, c
    # ndarray / tensor fallback
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.shape[0], arr.shape[1], 1
    return arr.shape[0], arr.shape[1], arr.shape[2]


def sample_resolution(ds, n: int, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    hs, ws, cs = [], [], []
    for i in idx:
        img = ds[int(i)]["image"]
        h, w, c = image_hwc(img)
        hs.append(h)
        ws.append(w)
        cs.append(c)
    return float(np.mean(hs)), float(np.mean(ws)), float(np.mean(cs))


def one_dataset(name: str, cfg: DatasetConfig) -> dict:
    print(f"[{name}] loading…", flush=True)
    cls = _get_dataset_class(cfg)
    ds = cls(split="train", **DATA_KWARGS, **cfg.builder_kwargs)
    labels = get_labels(ds)

    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    k = int(len(counts))
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum())
    class_balance = entropy / float(np.log(k)) if k > 1 else 1.0
    imbalance_ratio = float(counts.max() / counts.min())

    mean_h, mean_w, mean_c = sample_resolution(ds, N_RESOLUTION_SAMPLES)
    mean_pixels = mean_h * mean_w

    row = {
        "dataset": name,
        "train_size": int(len(ds)),
        "num_classes": k,
        "class_balance": class_balance,
        "imbalance_ratio": imbalance_ratio,
        "mean_height": mean_h,
        "mean_width": mean_w,
        "mean_channels": mean_c,
        "mean_pixels": mean_pixels,
    }
    print(
        f"[{name}] K={k}  n={len(ds)}  balance={class_balance:.3f}  "
        f"imb_ratio={imbalance_ratio:.2f}  res={mean_h:.0f}x{mean_w:.0f}x{mean_c:.0f}",
        flush=True,
    )
    return row


def correlations_report(meta: pd.DataFrame) -> None:
    gaps_path = OUT_DIR / "ssl_supervised_gap.csv"
    dinov2_path = OUT_DIR / "intraclass_variation_dinov2.csv"
    if not gaps_path.exists() or not dinov2_path.exists():
        print("(missing gap or dinov2 CSV — skipping correlation report)")
        return
    gaps = pd.read_csv(gaps_path)[["dataset", "ssl_advantage"]]
    dino = pd.read_csv(dinov2_path)[["dataset", "intraclass_variation"]].rename(
        columns={"intraclass_variation": "intraclass_dinov2"}
    )
    df = meta.merge(gaps, on="dataset").merge(dino, on="dataset")

    features = [
        "train_size",
        "num_classes",
        "class_balance",
        "imbalance_ratio",
        "mean_height",
        "mean_pixels",
        "mean_channels",
    ]
    rows = []
    for f in features:
        rho_adv, p_adv = stats.spearmanr(df[f], df["ssl_advantage"])
        rho_int, p_int = stats.spearmanr(df[f], df["intraclass_dinov2"])
        rows.append(
            {
                "feature": f,
                "rho_vs_ssl_advantage": float(rho_adv),
                "p_vs_ssl_advantage": float(p_adv),
                "rho_vs_intraclass_dinov2": float(rho_int),
                "p_vs_intraclass_dinov2": float(p_int),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "dataset_metadata_correlations.csv", index=False)
    print("\n=== Spearman correlations (n = 19) ===")
    print(out.to_string(index=False))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="compute one dataset (for job-array mode)")
    parser.add_argument("--gather", action="store_true", help="merge per-dataset shards + run correlations")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUT_DIR / "metadata_shards"
    shard_dir.mkdir(exist_ok=True)

    if args.gather:
        import glob as g

        shards = sorted(g.glob(str(shard_dir / "*.csv")))
        if not shards:
            print("no shards found")
            return
        meta = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        meta = meta.sort_values("dataset").reset_index(drop=True)
        meta.to_csv(OUT_CSV, index=False)
        print(f"gathered {len(meta)} datasets → {OUT_CSV}")
        correlations_report(meta)
        return

    if args.dataset:
        targets = {args.dataset: DATASETS[args.dataset]}
    else:
        targets = DATASETS

    for name, cfg in targets.items():
        row = one_dataset(name, cfg)
        shard = shard_dir / f"{name}.csv"
        pd.DataFrame([row]).to_csv(shard, index=False)
        print(f"  wrote {shard}")

    if not args.dataset:
        meta = pd.DataFrame(
            [pd.read_csv(shard_dir / f"{n}.csv").iloc[0] for n in DATASETS if (shard_dir / f"{n}.csv").exists()]
        )
        meta.to_csv(OUT_CSV, index=False)
        print(f"\nwrote {OUT_CSV}")
        correlations_report(meta)


if __name__ == "__main__":
    main()
