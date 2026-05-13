"""Compute dataset-level metadata for the RankMe correlation analysis.

The output schema is the metadata contract consumed by ``analysis.py``:

  - dataset
  - train_size
  - num_classes
  - class_balance
  - imbalance_ratio
  - mean_height
  - mean_width
  - mean_channels
  - mean_pixels

``mean_pixels`` is the average of ``height * width`` over sampled training
images.  ``analysis.py`` exposes this as ``height_times_width`` in correlation
outputs.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dataset import DATASET_CONFIGS, INCLUDED_IMAGE_DATASETS, DatasetConfig, _get_dataset_class


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = HERE / "dataset_metadata.csv"
DEFAULT_SHARD_DIR = HERE / "metadata_shards"
DEFAULT_DATA_ROOT = Path("./.anonymous-datasets-cache")
N_RESOLUTION_SAMPLES = 16

DATASETS: dict[str, DatasetConfig] = {
    name: cfg for name, cfg in DATASET_CONFIGS.items() if name in INCLUDED_IMAGE_DATASETS
}


def data_kwargs(data_root: Path) -> dict[str, str]:
    return {
        "download_dir": str(data_root / "downloads"),
        "processed_cache_dir": str(data_root / "processed"),
    }


def get_labels(ds) -> np.ndarray:
    try:
        return np.asarray(ds._table.column("label").to_numpy(zero_copy_only=False))
    except Exception:
        return np.array([int(ds[i]["label"]) for i in range(len(ds))])


def image_hwc(img) -> tuple[int, int, int]:
    if hasattr(img, "size") and hasattr(img, "getbands"):
        w, h = img.size
        c = len(img.getbands())
        return h, w, c
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.shape[0], arr.shape[1], 1
    return arr.shape[0], arr.shape[1], arr.shape[2]


def sample_resolution(ds, n: int, seed: int = 0) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    hs, ws, cs, pixels = [], [], [], []
    for i in idx:
        h, w, c = image_hwc(ds[int(i)]["image"])
        hs.append(h)
        ws.append(w)
        cs.append(c)
        pixels.append(h * w)
    return float(np.mean(hs)), float(np.mean(ws)), float(np.mean(cs)), float(np.mean(pixels))


def one_dataset(name: str, cfg: DatasetConfig, *, data_root: Path) -> dict:
    print(f"[{name}] loading...", flush=True)
    cls = _get_dataset_class(cfg)
    ds = cls(split="train", **data_kwargs(data_root), **cfg.builder_kwargs)
    labels = get_labels(ds)

    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    k = int(len(counts))
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum())
    class_balance = entropy / float(np.log(k)) if k > 1 else 1.0
    imbalance_ratio = float(counts.max() / counts.min())

    mean_h, mean_w, mean_c, mean_pixels = sample_resolution(ds, N_RESOLUTION_SAMPLES)
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
        f"[{name}] K={k} n={len(ds)} balance={class_balance:.3f} "
        f"imb_ratio={imbalance_ratio:.2f} res={mean_h:.0f}x{mean_w:.0f}x{mean_c:.0f}",
        flush=True,
    )
    return row


def gather_shards(shard_dir: Path, output_csv: Path) -> pd.DataFrame:
    shards = sorted(glob.glob(str(shard_dir / "*.csv")))
    if not shards:
        raise FileNotFoundError(f"no metadata shards found in {shard_dir}")
    meta = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
    meta = meta.sort_values("dataset").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None, help="compute one dataset shard")
    parser.add_argument("--gather", action="store_true", help="merge shards into dataset_metadata.csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    args = parser.parse_args()

    args.shard_dir.mkdir(parents=True, exist_ok=True)

    if args.gather:
        meta = gather_shards(args.shard_dir, args.output_csv)
        print(f"gathered {len(meta)} datasets -> {args.output_csv}")
        return

    if args.dataset is not None:
        if args.dataset not in DATASETS:
            raise KeyError(f"unknown dataset {args.dataset!r}; available: {sorted(DATASETS)}")
        targets = {args.dataset: DATASETS[args.dataset]}
    else:
        targets = DATASETS

    for name, cfg in targets.items():
        row = one_dataset(name, cfg, data_root=args.data_root)
        shard = args.shard_dir / f"{name}.csv"
        pd.DataFrame([row]).to_csv(shard, index=False)
        print(f"  wrote {shard}")

    if args.dataset is None:
        meta = gather_shards(args.shard_dir, args.output_csv)
        print(f"\nwrote {args.output_csv} ({len(meta)} datasets)")


if __name__ == "__main__":
    main()
