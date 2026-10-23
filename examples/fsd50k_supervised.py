"""Supervised FSD50K baseline: the VGG-like model of Fonseca et al. (TASLP 2022).

Reproduces the VGG-like row of Table VI in "FSD50K: An Open Dataset of Human-Labeled Sound Events"
(https://arxiv.org/abs/2010.00475) on top of the ``FSD50K`` builder. Two stages, run from the
repository root with the ``benchmarks`` extra:

.. code-block:: bash

    python examples/fsd50k_supervised.py features --out runs/fsd50k
    python examples/fsd50k_supervised.py train --features runs/fsd50k --out runs/fsd50k/vgg_seed0 --seed 0

Protocol (paper Sec. V-B): 22.05 kHz audio, 96-band log-mel and 1 s patches (101 frames), the front end
of ``fsd_simclr.py``. Clips shorter than 1 s are replicated; longer clips are sliced into patches with
50% overlap that inherit the clip labels. The VGG-like network (0.27M weights) has sigmoid outputs for
the 200 classes and is trained with binary cross-entropy, Adam (lr 3e-4) and batches of 64. The learning
rate is halved after 5 epochs without a better validation score and training stops after 10 (at most
100 epochs). The best validation checkpoint is evaluated once on the eval set (``test`` split). Clip
scores average the patch outputs; the metrics are macro mAP, d' and lwlrap. The paper averages lwlrap
over classes, reported here as ``lwlrap_class``; ``lwlrap`` is the DCASE definition (every label weighs
the same), which comes out about 0.1 higher on FSD50K because frequent classes are easier.
Published: mAP 0.434 ± 0.002, d' 2.167 ± 0.011, lwlrap (class-averaged) 0.514 ± 0.003.

The paper's baseline repository is no longer online, so a few details follow the text and Keras
defaults: model selection uses validation mAP (the paper used TensorFlow's PR-AUC), inputs are
standardised with training-split statistics, the 256-unit layer uses ReLU, and layers use Glorot-uniform
initialisation and Keras batch-norm settings.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import fsd_simclr as simclr  # shared front end; resolved from this directory when run as a script
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")


# -- Feature store -----------------------------------------------------------


def expected_frames(n_samples: int) -> int:
    """Frames produced by ``LogMel`` for a clip of ``n_samples`` (clips under 1 s are replicated to 1 s)."""
    return 1 + max(n_samples, simclr.SAMPLE_RATE) // simclr.HOP_LENGTH


def build_features(out: Path, builder=None):
    """Compute log-mel features for every FSD50K clip and store them per split.

    Writes ``{split}.npy`` (all frames, ``[F, N_MELS]`` float32) and ``{split}.json`` (per clip:
    ``offset``, ``frames``, ``labels``, ``filename``) for ``train``, ``validation`` and ``test``. Frame
    counts are known from the clip lengths, so frames are written straight into a memory-mapped file.
    """
    import pyarrow.compute as pc

    if builder is None:
        from stable_datasets.timeseries import FSD50K as builder

    out.mkdir(parents=True, exist_ok=True)
    front_end = simclr.LogMel()
    counts = {}
    for split in SPLITS:
        dataset = builder(split=split, config_name="22khz")
        label_names = dataset.features["labels"].feature.names
        samples = pc.list_value_length(dataset.table.column("series")).to_numpy(zero_copy_only=False)
        lengths = [expected_frames(int(n)) for n in samples]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        frames = np.lib.format.open_memmap(
            out / f"{split}.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), simclr.N_MELS)
        )
        clips = []
        for i, (row, wave) in enumerate(simclr._waveforms(dataset)):
            if row["sample_rate"] != simclr.SAMPLE_RATE:
                raise ValueError(
                    f"Expected {simclr.SAMPLE_RATE} Hz audio, got {row['sample_rate']} for {row['filename']}."
                )
            mel = front_end(wave)
            if len(mel) != lengths[i]:
                raise ValueError(f"{row['filename']}: {len(mel)} frames, expected {lengths[i]}.")
            frames[offsets[i] : offsets[i + 1]] = mel
            clips.append(
                {
                    "filename": row["filename"],
                    "labels": [int(label) for label in row["labels"]],
                    "frames": lengths[i],
                    "offset": int(offsets[i]),
                }
            )
        frames.flush()
        del frames
        (out / f"{split}.json").write_text(json.dumps({"label_names": label_names, "clips": clips}) + "\n")
        counts[split] = len(clips)
    return counts


class Features(simclr.Features):
    """Memory-mapped log-mel frames of one split with per-clip offsets and multi-hot ``targets``."""

    def __init__(self, root: Path, split: str):
        self.path = root / f"{split}.npy"
        self.frames = np.load(self.path, mmap_mode="r")
        meta = json.loads((root / f"{split}.json").read_text())
        self.label_names = meta["label_names"]
        self.offsets = np.asarray([c["offset"] for c in meta["clips"]], dtype=np.int64)
        self.lengths = np.asarray([c["frames"] for c in meta["clips"]], dtype=np.int64)
        self.targets = np.zeros((len(meta["clips"]), len(self.label_names)), dtype=np.float32)
        for i, clip in enumerate(meta["clips"]):
            self.targets[i, clip["labels"]] = 1.0
        self.filenames = [c["filename"] for c in meta["clips"]]


def patch_index(features: Features):
    """Absolute start frame and clip of every 1 s patch (50% overlap): ``(starts [P], clips [P])``."""
    starts, clips = [], []
    for clip, (offset, length) in enumerate(zip(features.offsets, features.lengths)):
        patch = np.asarray(simclr.patch_starts(int(length)), dtype=np.int64)
        starts.append(offset + patch)
        clips.append(np.full(len(patch), clip, dtype=np.int64))
    return np.concatenate(starts), np.concatenate(clips)


class Patches(Dataset):
    """Every training patch with the labels of its clip (false strong labelling)."""

    def __init__(self, features: Features, mean: float, std: float):
        self.features, self.mean, self.std = features, mean, std
        self.starts, self.clips = patch_index(features)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        patch = np.array(self.features.frames[start : start + simclr.PATCH_LEN], dtype=np.float32)
        target = torch.from_numpy(self.features.targets[self.clips[index]])
        return (torch.from_numpy(patch)[None] - self.mean) / self.std, target


# -- Model -------------------------------------------------------------------


class VGGLike(nn.Module):
    """Paper's VGG-like network: 3x conv32, pool, 2x conv64, pool, conv128, [GMP, GAP], FC 256, FC classes."""

    def __init__(self, n_classes: int = 200):
        super().__init__()

        def conv(c_in, c_out):
            # Keras BatchNormalization defaults: momentum 0.99 (0.01 in PyTorch terms), epsilon 1e-3.
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01), nn.ReLU()]

        self.convs = nn.Sequential(
            *conv(1, 32),
            *conv(32, 32),
            *conv(32, 32),
            nn.MaxPool2d(2),
            *conv(32, 64),
            *conv(64, 64),
            nn.MaxPool2d(2),
            *conv(64, 128),
        )
        self.hidden = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.classifier = nn.Linear(256, n_classes)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.convs(x)
        x = torch.cat([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))], dim=1)
        return self.classifier(self.hidden(x))


# -- Evaluation --------------------------------------------------------------


@torch.no_grad()
def clip_scores(model: nn.Module, features: Features, mean, std, device, batch_size: int = 1024) -> np.ndarray:
    """Average the sigmoid outputs of every patch of each clip: ``[clips, classes]``."""
    model.eval()
    starts, clips = patch_index(features)
    window = np.arange(simclr.PATCH_LEN)
    totals = torch.zeros(len(features), features.targets.shape[1], dtype=torch.float64)
    for begin in range(0, len(starts), batch_size):
        chunk = slice(begin, begin + batch_size)
        patches = torch.from_numpy(np.asarray(features.frames[starts[chunk, None] + window], dtype=np.float32))
        scores = torch.sigmoid(model(((patches[:, None] - mean) / std).to(device))).cpu().double()
        totals.index_add_(0, torch.from_numpy(clips[chunk]), scores)
    counts = np.bincount(clips, minlength=len(features))
    return (totals / torch.from_numpy(counts)[:, None]).numpy()


def lwlrap(targets: np.ndarray, scores: np.ndarray, average: str = "label") -> float:
    """Label-weighted label-ranking average precision (DCASE 2019 Task 2; paper Eq. 3).

    ``average="label"`` weighs every (clip, label) pair equally, the DCASE definition. ``average="class"``
    first averages the pairs of each class and then the classes with equal weight, which is how the
    FSD50K paper reports it ("all metrics are computed on a per-class basis", Sec. V-A).
    """
    precisions, classes = [], []
    for target, score in zip(targets, scores):
        labels = np.flatnonzero(target > 0)
        positive = score[labels]
        rank = (score[None, :] >= positive[:, None]).sum(1)
        hits = (positive[None, :] >= positive[:, None]).sum(1)
        precisions.append(hits / rank)
        classes.append(labels)
    precisions, classes = np.concatenate(precisions), np.concatenate(classes)
    if average == "label":
        return float(precisions.mean())
    if average == "class":
        return float(np.mean([precisions[classes == c].mean() for c in np.unique(classes)]))
    raise ValueError(f"average must be 'label' or 'class', got {average!r}")


def metrics(targets: np.ndarray, scores: np.ndarray) -> dict:
    """Macro mAP and d' over classes with both positive and negative clips, and lwlrap (DCASE and class-averaged)."""
    from scipy.stats import norm
    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = targets.sum(0)
    valid = (positives > 0) & (positives < len(targets))
    ap = average_precision_score(targets[:, valid], scores[:, valid], average=None)
    auc = roc_auc_score(targets[:, valid], scores[:, valid], average=None)
    return {
        "mAP": float(np.mean(ap)),
        "d_prime": float(np.mean(np.sqrt(2) * norm.ppf(auc))),
        "lwlrap": lwlrap(targets, scores),
        "lwlrap_class": lwlrap(targets, scores, average="class"),
        "classes": int(valid.sum()),
    }


# -- Training ----------------------------------------------------------------


class Plateau:
    """Keras ``ReduceLROnPlateau`` + ``EarlyStopping`` on a score to maximise (no tolerance, no cooldown)."""

    def __init__(self, optimizer, patience: int = 5, stop: int = 10, factor: float = 0.5):
        self.optimizer, self.patience, self.stop, self.factor = optimizer, patience, stop, factor
        self.best, self.since_best, self.since_decay = None, 0, 0

    def step(self, score: float) -> bool:
        """Record an epoch's validation score; returns whether it is a new best."""
        if self.best is None or score > self.best:
            self.best, self.since_best, self.since_decay = score, 0, 0
            return True
        self.since_best += 1
        self.since_decay += 1
        if self.since_decay >= self.patience:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.factor
            self.since_decay = 0
        return False

    @property
    def should_stop(self) -> bool:
        return self.since_best >= self.stop


def train(
    root: Path,
    out: Path,
    seed: int,
    device,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    workers: int = 8,
):
    torch.manual_seed(seed)
    data = {split: Features(root, split) for split in SPLITS}
    mean, std = simclr.normalization_stats(data["train"], "standard")
    loader = DataLoader(
        Patches(data["train"], mean, std),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    model = VGGLike(data["train"].targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)  # Keras epsilon
    plateau = Plateau(optimizer)
    history, best, started = [], None, time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_lr, total, steps = optimizer.param_groups[0]["lr"], 0.0, 0
        for patches, targets in loader:
            loss = F.binary_cross_entropy_with_logits(model(patches.to(device)), targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss.detach()), steps + 1
        validation = metrics(data["validation"].targets, clip_scores(model, data["validation"], mean, std, device))
        row = {"epoch": epoch, "loss": total / steps, "lr": epoch_lr, "validation": validation}
        if plateau.step(validation["mAP"]):
            best = row
            torch.save(model.state_dict(), out / "model.pt")
        row["seconds"] = time.monotonic() - started
        history.append(row)
        print(json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if plateau.should_stop:
            break

    model.load_state_dict(torch.load(out / "model.pt", map_location=device, weights_only=True))
    scores = clip_scores(model, data["test"], mean, std, device)
    np.savez(out / "predictions.npz", scores=scores, targets=data["test"].targets)
    result = {
        "seed": seed,
        "selected_epoch": best["epoch"],
        "epochs_run": len(history),
        "validation": best["validation"],
        "test": metrics(data["test"].targets, scores),
        "normalization": {"mean": mean, "std": std},
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# -- CLI ---------------------------------------------------------------------


def provenance(args) -> dict:
    record = simclr.provenance(args)
    record["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    features = commands.add_parser("features", help="Compute log-mel features from the FSD50K builder.")
    features.add_argument("--out", type=Path, required=True)

    fit = commands.add_parser("train", help="Train the VGG-like baseline and evaluate it on the eval set.")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--workers", type=int, default=8)
    fit.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.command == "features":
        counts = build_features(args.out)
        (args.out / "features_config.json").write_text(json.dumps({**provenance(args), "counts": counts}, indent=2))
        print(json.dumps(counts))
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "config.json").write_text(json.dumps(provenance(args), indent=2) + "\n")
        result = train(
            args.features,
            args.out,
            args.seed,
            simclr.device_of(args.device),
            args.epochs,
            args.batch_size,
            args.lr,
            args.workers,
        )
        print(json.dumps({"selected_epoch": result["selected_epoch"], "test": result["test"]}))


if __name__ == "__main__":
    main()
