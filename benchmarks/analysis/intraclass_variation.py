"""Measure per-dataset intra-class variation in DINOv2 feature space.

For each benchmark dataset, embed up to N_PER_CLASS training images per
class with frozen DINOv2-small, then compute the mean pairwise cosine
distance within each class and average across classes. That scalar
replaces the hand-coded `centered` flag in ssl_supervised_gap.py with
a real measurement.

Outputs (in benchmarks/results/):
  intraclass_variation.csv          per-dataset scores + n_sampled
  intraclass_variation_joined.csv   joined with ssl_advantage
  intraclass_variation_scatter.png  ssl_advantage vs intraclass score

Run on a GPU node for sanity — CPU works but is slow on big datasets.
Re-running is cheap: scores already in the output CSV are skipped unless
--overwrite is passed. Use --dataset NAME to run one dataset (debugging).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats

from benchmarks.dataset import DATASET_CONFIGS, INCLUDED_IMAGE_DATASETS, DatasetConfig, _get_dataset_class


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "benchmarks" / "results"
GAP_CSV = OUT_DIR / "ssl_supervised_gap.csv"

# Match benchmarks/conf/config.yaml so we reuse the existing scratch cache
# instead of re-downloading multi-GB tarballs into $HOME.
DATA_ROOT = Path("/oscar/home/sboughan/scratch/.stable-datasets")
DATA_KWARGS = {
    "download_dir": str(DATA_ROOT / "downloads"),
    "processed_cache_dir": str(DATA_ROOT / "processed"),
}

# Encoder registry. Each entry maps a short tag (used in output filenames
# and CLI) to a timm model id. The tag gets embedded in all per-encoder
# artifacts so runs don't clobber each other.
#
# - dinov2 : augmentation-invariance objective (DINO+iBOT); same family as
#            most of the benchmark SSL methods. Primary run.
# - in21k  : supervised cross-entropy on ImageNet-21k labels. Different
#            objective family; sensitivity-analysis baseline.
# - clip   : image-text contrastive on 400M web pairs. Third objective,
#            different data, different scale.
ENCODERS: dict[str, str] = {
    "dinov2": "vit_small_patch14_dinov2",
    "in21k": "vit_small_patch16_224.augreg_in21k",
    "clip": "vit_large_patch14_clip_224.openai",
}

N_PER_CLASS = 32
BATCH_SIZE = 32  # conservative; ViT-L CLIP at 224 still fits comfortably

DATASETS: dict[str, DatasetConfig] = {
    name: cfg for name, cfg in DATASET_CONFIGS.items() if name in INCLUDED_IMAGE_DATASETS
}


def build_model(encoder: str, device: torch.device) -> torch.nn.Module:
    model_name = ENCODERS[encoder]
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    return model


def build_transform(model: torch.nn.Module) -> Callable[[Image.Image], torch.Tensor]:
    cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**cfg, is_training=False)


def sample_indices_by_class(labels: np.ndarray, n_per_class: int, seed: int) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        by_class.setdefault(int(lbl), []).append(idx)
    out: dict[int, list[int]] = {}
    for cls, idxs in by_class.items():
        if len(idxs) <= n_per_class:
            out[cls] = idxs
        else:
            out[cls] = rng.choice(idxs, size=n_per_class, replace=False).tolist()
    return out


def get_labels(ds) -> np.ndarray:
    try:
        col = ds._table.column("label").to_numpy(zero_copy_only=False)
        return np.asarray(col)
    except Exception:
        return np.array([int(ds[i]["label"]) for i in range(len(ds))])


def embed_samples(
    model: torch.nn.Module,
    transform,
    ds,
    sampled: dict[int, list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    flat_idx: list[int] = []
    flat_cls: list[int] = []
    for cls, idxs in sampled.items():
        flat_idx.extend(idxs)
        flat_cls.extend([cls] * len(idxs))

    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(flat_idx), BATCH_SIZE):
            batch_idx = flat_idx[start : start + BATCH_SIZE]
            imgs = []
            for i in batch_idx:
                sample = ds[i]
                img: Image.Image = sample["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                imgs.append(transform(img))
            x = torch.stack(imgs).to(device)
            e = model(x)
            e = F.normalize(e, dim=-1)
            embs.append(e.cpu())
    return torch.cat(embs, dim=0), np.array(flat_cls)


def intraclass_variation(embs: torch.Tensor, cls: np.ndarray) -> float:
    scores: list[float] = []
    for c in np.unique(cls):
        mask = cls == c
        if mask.sum() < 2:
            continue
        e = embs[mask]
        sim = e @ e.T
        m = e.shape[0]
        off_sum = sim.sum() - sim.diagonal().sum()
        mean_sim = off_sum / (m * (m - 1))
        scores.append(1.0 - float(mean_sim))
    return float(np.mean(scores))


def scores_csv_path(encoder: str) -> Path:
    return OUT_DIR / f"intraclass_variation_{encoder}.csv"


def load_existing(encoder: str) -> pd.DataFrame:
    path = scores_csv_path(encoder)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset", "intraclass_variation", "n_sampled", "n_classes"])


def process_one(name: str, model, transform, device, seed: int) -> dict | None:
    cfg = DATASETS[name]
    cls = _get_dataset_class(cfg)
    print(f"[{name}] loading…", flush=True)
    ds = cls(split="train", **DATA_KWARGS, **cfg.builder_kwargs)
    labels = get_labels(ds)
    sampled = sample_indices_by_class(labels, N_PER_CLASS, seed)
    n_sampled = sum(len(v) for v in sampled.values())
    print(f"[{name}] {len(ds)} rows, {len(sampled)} classes, sampling {n_sampled}", flush=True)
    embs, cls_arr = embed_samples(model, transform, ds, sampled, device)
    score = intraclass_variation(embs, cls_arr)
    print(f"[{name}] intraclass_variation = {score:.4f}", flush=True)
    return {"dataset": name, "intraclass_variation": score, "n_sampled": n_sampled, "n_classes": len(sampled)}


def analyze_and_plot(scores: pd.DataFrame, encoder: str) -> None:
    if not GAP_CSV.exists():
        print(f"(no {GAP_CSV.name} yet; skipping join/plot)")
        return
    gaps = pd.read_csv(GAP_CSV)
    df = gaps.merge(scores, on="dataset", how="inner")
    if df.empty:
        print("(no overlap between scores and gap CSV yet)")
        return
    df = df.sort_values("intraclass_variation").reset_index(drop=True)
    df.to_csv(OUT_DIR / f"intraclass_variation_{encoder}_joined.csv", index=False)

    if len(df) < 3:
        print(f"(only {len(df)} datasets scored so far; skipping correlation/plot)")
        return

    rho, p = stats.spearmanr(df["intraclass_variation"], df["ssl_advantage"])
    r_pearson, p_pearson = stats.pearsonr(df["intraclass_variation"], df["ssl_advantage"])
    print(f"\nencoder = {encoder}  ({ENCODERS[encoder]})")
    print(f"n = {len(df)}")
    print(f"Spearman ρ(intraclass, ssl_advantage) = {rho:.3f}  (p = {p:.4f})")
    print(f"Pearson  r(intraclass, ssl_advantage) = {r_pearson:.3f}  (p = {p_pearson:.4f})")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    x = df["intraclass_variation"]
    y = df["ssl_advantage"]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in y]
    ax.scatter(x, y, c=colors, s=60)
    ax.axhline(0, color="k", lw=0.5)
    for _, r in df.iterrows():
        ax.annotate(r.dataset, (r.intraclass_variation, r.ssl_advantage), fontsize=8, alpha=0.8)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color="gray", lw=1)
    ax.set_xlabel(f"intra-class variation (mean 1 − cos_sim, {encoder})  ρ={rho:.2f}")
    ax.set_ylabel("best SSL − best supervised")
    ax.set_title(f"SSL advantage vs intra-class variation — encoder: {encoder}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"intraclass_variation_{encoder}_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="dinov2", choices=sorted(ENCODERS.keys()))
    parser.add_argument("--dataset", type=str, default=None, help="run only this dataset")
    parser.add_argument("--overwrite", action="store_true", help="recompute even if already in CSV")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true", help="skip embedding; just re-run the join/plot")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = args.encoder
    print(f"encoder = {encoder}  ({ENCODERS[encoder]})")
    existing = load_existing(encoder)
    done = set(existing.dataset) if not args.overwrite else set()

    if not args.analyze_only:
        targets = [args.dataset] if args.dataset else list(DATASETS.keys())
        todo = [t for t in targets if t not in done]
        if not todo:
            print("nothing to compute; all targets already in CSV. Pass --overwrite to redo.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"device = {device}")
            model = build_model(encoder, device)
            transform = build_transform(model)
            rows = existing.to_dict("records")
            csv_path = scores_csv_path(encoder)
            for name in todo:
                row = process_one(name, model, transform, device, args.seed)
                if row is not None:
                    rows = [r for r in rows if r["dataset"] != name] + [row]
                    pd.DataFrame(rows).to_csv(csv_path, index=False)

    scores = load_existing(encoder)
    analyze_and_plot(scores, encoder)


if __name__ == "__main__":
    main()
