"""CLIP-protocol linear probe on frozen ImageNet ResNet-50 features.

Reproduces the linear-probe evaluation of Radford et al. 2021 (CLIP), Appendix A.3, so new datasets
can be checked against the "ResNet 50" row of the paper's Table 10:

* penultimate-layer features of torchvision's ImageNet ResNet-50 (``IMAGENET1K_V1``), fc removed;
* scikit-learn L-BFGS logistic regression, ``max_iter=1000``;
* L2 strength lambda (``C = 1 / lambda``) picked on the validation split by a parametric binary search
  over ``[1e-6, 1e6]``: start at ``10^{-6, -4, ..., 6}`` and halve the interval around the peak down to
  8 steps per decade. Without a validation split, 10% of train is held out for the search;
* the final classifier is refit on train + validation and scored on test.

Usage::

    python -m benchmarks.transfer.clip_probe --dataset kitti_distance --data-dir <root>
    python -m benchmarks.transfer.clip_probe --dataset cifar10 --data-dir <root>   # calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import stable_datasets as sds


log = logging.getLogger(__name__)

# dataset -> (builder class, CLIP Table 10 "ResNet 50" top-1, default preprocessing)
DATASETS = {
    "cifar10": ("CIFAR10", 91.8, "center_crop"),
    # Wide 1242x375 driving frames: a center crop keeps only the middle ~26% of the width and drops the
    # vehicles at the image edges, which are often the closest ones. Resize the whole frame, as VTAB does.
    "kitti_distance": ("KITTIDistance", 67.1, "resize_full"),
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _transform(preprocess: str) -> T.Compose:
    if preprocess == "center_crop":  # torchvision ImageNet eval transform
        return torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])


class _Split(Dataset):
    def __init__(self, ds, transform, limit: int | None = None):
        self.ds, self.transform = ds, transform
        self.n = len(ds) if limit is None else min(limit, len(ds))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ex = self.ds[i]
        return self.transform(ex["image"].convert("RGB")), int(ex["label"])


@torch.no_grad()
def _extract(model, ds, transform, batch_size, num_workers, limit, device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(_Split(ds, transform, limit), batch_size=batch_size, num_workers=num_workers)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def _fit_score(lam: float, x_fit, y_fit, x_eval, y_eval) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1.0 / lam, max_iter=1000, solver="lbfgs").fit(x_fit, y_fit)
    return 100.0 * clf.score(x_eval, y_eval)


def _sweep(x_fit, y_fit, x_val, y_val) -> tuple[float, float]:
    """Binary search over log10(lambda) in [-6, 6] down to 1/8 decade; returns (log10 lambda, val acc)."""
    scores: dict[float, float] = {}

    def evaluate(e: float) -> None:
        e = round(e, 4)
        if -6 <= e <= 6 and e not in scores:
            scores[e] = _fit_score(10.0**e, x_fit, y_fit, x_val, y_val)
            log.info("  lambda=1e%+.3f  val_acc=%.2f", e, scores[e])

    for e in range(-6, 7, 2):
        evaluate(e)
    step = 2.0
    while step > 1 / 8:
        step /= 2
        best = max(scores, key=scores.get)
        evaluate(best - step)
        evaluate(best + step)
    best = max(scores, key=scores.get)
    return best, scores[best]


def run(
    dataset: str,
    data_dir: str | None = None,
    preprocess: str | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    feature_dir: str | None = None,
    limit: int | None = None,
) -> dict:
    builder_name, paper_top1, default_preprocess = DATASETS[dataset]
    preprocess = preprocess or default_preprocess

    kwargs = {}
    if data_dir is not None:
        kwargs = {
            "download_dir": str(Path(data_dir) / "downloads"),
            "processed_cache_dir": str(Path(data_dir) / "processed"),
        }
    splits = getattr(sds.images, builder_name)(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    transform = _transform(preprocess)
    log.info("%s: %s | preprocess=%s | device=%s", dataset, {k: len(v) for k, v in splits.items()}, preprocess, device)

    feats = {}
    for name, split in splits.items():
        cache = None
        if feature_dir is not None and limit is None:
            os.makedirs(feature_dir, exist_ok=True)
            cache = Path(feature_dir) / f"{dataset}_resnet50_{preprocess}_{name}.npz"
        if cache is not None and cache.exists():
            z = np.load(cache)
            feats[name] = (z["X"], z["y"])
            continue
        t0 = time.time()
        feats[name] = _extract(model, split, transform, batch_size, num_workers, limit, device)
        log.info("extracted %s: %s in %.0fs", name, feats[name][0].shape, time.time() - t0)
        if cache is not None:
            np.savez(cache, X=feats[name][0], y=feats[name][1])

    x_tr, y_tr = feats["train"]
    x_te, y_te = feats["test"]
    if "validation" in feats:
        x_val, y_val = feats["validation"]
        x_fit, y_fit = x_tr, y_tr
        x_final, y_final = np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val])
    else:  # CLIP: hold out part of train for the hyperparameter search
        idx = np.random.RandomState(0).permutation(len(y_tr))
        n_val = len(y_tr) // 10
        x_val, y_val = x_tr[idx[:n_val]], y_tr[idx[:n_val]]
        x_fit, y_fit = x_tr[idx[n_val:]], y_tr[idx[n_val:]]
        x_final, y_final = x_tr, y_tr

    log.info("sweeping L2 strength on validation:")
    best_e, best_val = _sweep(x_fit, y_fit, x_val, y_val)
    test_acc = _fit_score(10.0**best_e, x_final, y_final, x_te, y_te)

    return {
        "dataset": dataset,
        "backbone": "torchvision resnet50 IMAGENET1K_V1",
        "preprocess": preprocess,
        "best_lambda": f"1e{best_e:+.3f}",
        "val_acc": round(best_val, 2),
        "test_acc": round(test_acc, 2),
        "test_majority": round(100.0 * np.bincount(y_te).max() / len(y_te), 2),
        "clip_paper": paper_top1,
        "n_test": int(len(y_te)),
        "limit": limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--data-dir", default=None, help="Root with downloads/ and processed/ (default: library cache)"
    )
    parser.add_argument(
        "--preprocess",
        choices=["center_crop", "resize_full"],
        default=None,
        help="center_crop: resize 256 + center-crop 224; resize_full: whole image to 224x224 (default: per dataset)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-dir", default=None, help="Cache extracted features here")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per split (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(
        args.dataset,
        data_dir=args.data_dir,
        preprocess=args.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_dir=args.feature_dir,
        limit=args.limit,
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
