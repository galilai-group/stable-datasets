"""SimCLR sound event representations on FSDnoisy18k, following Fonseca et al. (ICASSP 2021).

Reproduces "Unsupervised Contrastive Learning of Sound Event Representations"
(https://arxiv.org/abs/2011.07616, code: https://github.com/edufonseca/uclser20) on top of the
``FSDnoisy18k`` builder. Three stages, run from the repository root with the ``benchmarks`` extra:

.. code-block:: bash

    python examples/fsd_simclr.py features --out runs/fsdnoisy18k
    python examples/fsd_simclr.py pretrain --features runs/fsdnoisy18k --out runs/fsdnoisy18k/simclr_seed0 --seed 0
    python examples/fsd_simclr.py probe --features runs/fsdnoisy18k --encoder runs/fsdnoisy18k/simclr_seed0/encoder.pt \
        --out runs/fsdnoisy18k/simclr_seed0/probe

Protocol (paper Sec. 3): log-mel patches of 1 s; SimCLR is trained on the *noisy* train subset without
labels; the linear probe is trained on the noisy subset with its labels, the epoch is selected on the
manually verified *clean* subset, and the test set is evaluated once with the selected classifier.
The metric is balanced (class-averaged) clip accuracy, with clip predictions obtained by averaging
patch softmax outputs. ``--encoder random`` gives the random-init floor. ``--encoder`` also accepts
the authors' released ``ResNet_best.pth`` to check this data pipeline against their weights.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Front end of the paper (uclser20 ``config/*.yaml``, ``extract``).
SAMPLE_RATE = 22050
N_FFT = 2048
WIN_LENGTH = 660
HOP_LENGTH = 220
N_MELS = 96
F_MIN, F_MAX = 50.0, 10500.0
LOG_EPS = 1.1e-8
PATCH_LEN = 101  # frames, 1 s
PATCH_HOP = 50  # frames between evaluation patches

SUBSETS = ("noisy", "clean", "test")


# -- Front end ---------------------------------------------------------------


def _hz_to_mel(freq):
    """Slaney mel scale (linear below 1 kHz, logarithmic above), as ``librosa`` with ``htk=False``."""
    freq = np.array(freq, dtype=np.float64, ndmin=1)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels):
    mels = np.array(mels, dtype=np.float64, ndmin=1)
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    freqs = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


def mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
    """Triangular mel filters ``[n_mels, n_fft // 2 + 1]`` without area normalisation (``norm=None``)."""
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_min, mel_max = _hz_to_mel([f_min, f_max])
    mel_freqs = _mel_to_hz(np.linspace(mel_min, mel_max, n_mels + 2))
    widths = np.diff(mel_freqs)
    ramps = mel_freqs[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    return np.maximum(0, np.minimum(lower, upper)).astype(np.float32)


class LogMel:
    """Log10 mel energies of a mono waveform at ``SAMPLE_RATE``, as ``[frames, N_MELS]``.

    Mirrors uclser20 ``extract/wav2spec_utils.py``: remove the mean and peak-normalise the clip,
    replicate clips shorter than 1 s, power STFT with a 30 ms periodic Hamming window and reflect
    padding, Slaney mel filters (50 Hz-10.5 kHz), ``log10(x + 1.1e-8)``.
    """

    def __init__(self):
        self.window = torch.hamming_window(WIN_LENGTH, periodic=True, dtype=torch.float64)
        self.filters = torch.from_numpy(mel_filterbank()).double()

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        wave = np.asarray(wave, dtype=np.float64).reshape(-1)
        if wave.size == 0:
            raise ValueError("Empty waveform.")
        wave = wave - wave.mean()
        wave = wave / (np.abs(wave).max() + 0.005)
        if wave.size < SAMPLE_RATE:
            wave = np.tile(wave, math.ceil(SAMPLE_RATE / wave.size))[:SAMPLE_RATE]
        spec = torch.stft(
            torch.from_numpy(wave + LOG_EPS),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mel = self.filters @ spec.abs().pow(2)
        return torch.log10(mel + LOG_EPS).T.float().numpy()


def patch_starts(n_frames: int) -> list[int]:
    """Start frames of the evaluation patches (uclser20 ``fetch_file_2_tensor``)."""
    return list(range(0, n_frames - PATCH_LEN + 1, PATCH_HOP))


# -- Feature store -----------------------------------------------------------


def _waveforms(dataset):
    """Yield ``(row_metadata, waveform)`` from the Arrow table, without building Python lists."""
    import pyarrow.compute as pc

    table = dataset.table
    meta_columns = [name for name in table.column_names if name != "series"]
    for batch in table.to_batches(max_chunksize=64):
        series = batch.column("series")
        meta = batch.select(meta_columns).to_pylist()
        for i, row in enumerate(meta):
            frames = series[i].values  # [T] list of [1] channels
            yield row, pc.list_flatten(frames).to_numpy(zero_copy_only=False).astype(np.float32)


def build_features(out: Path, builder=None):
    """Compute log-mel features for every FSDnoisy18k clip and store them per subset.

    Writes ``{subset}.npy`` (all frames, ``[F, N_MELS]`` float32) and ``{subset}.json`` (per clip:
    ``offset``, ``frames``, ``label``, ``filename``) for ``noisy``, ``clean`` and ``test``.
    """
    if builder is None:
        from stable_datasets.timeseries import FSDnoisy18k as builder

    out.mkdir(parents=True, exist_ok=True)
    front_end = LogMel()
    clips = {subset: [] for subset in SUBSETS}
    frames = {subset: [] for subset in SUBSETS}
    for split in ("train", "test"):
        dataset = builder(split=split, config_name="22khz")
        label_names = dataset.features["label"].names
        for row, wave in _waveforms(dataset):
            if row["sample_rate"] != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {row['sample_rate']} for {row['filename']}.")
            subset = "test" if split == "test" else "clean" if row["manually_verified"] else "noisy"
            mel = front_end(wave)
            clips[subset].append({"filename": row["filename"], "label": int(row["label"]), "frames": len(mel)})
            frames[subset].append(mel)
    for subset in SUBSETS:
        offsets = np.concatenate([[0], np.cumsum([c["frames"] for c in clips[subset]])])
        for clip, offset in zip(clips[subset], offsets[:-1]):
            clip["offset"] = int(offset)
        np.save(out / f"{subset}.npy", np.concatenate(frames[subset]).astype(np.float32))
        (out / f"{subset}.json").write_text(json.dumps({"label_names": label_names, "clips": clips[subset]}) + "\n")
    return {subset: len(clips[subset]) for subset in SUBSETS}


class Features:
    """Memory-mapped log-mel frames of one subset with per-clip offsets and labels."""

    def __init__(self, root: Path, subset: str):
        self.path = root / f"{subset}.npy"
        self.frames = np.load(self.path, mmap_mode="r")
        meta = json.loads((root / f"{subset}.json").read_text())
        self.label_names = meta["label_names"]
        self.offsets = np.asarray([c["offset"] for c in meta["clips"]], dtype=np.int64)
        self.lengths = np.asarray([c["frames"] for c in meta["clips"]], dtype=np.int64)
        self.labels = np.asarray([c["label"] for c in meta["clips"]], dtype=np.int64)
        self.filenames = [c["filename"] for c in meta["clips"]]

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        # Data-loader workers reopen the memory map instead of receiving a copy of every frame.
        return {k: v for k, v in self.__dict__.items() if k != "frames"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.frames = np.load(self.path, mmap_mode="r")

    def clip(self, index: int) -> np.ndarray:
        start = self.offsets[index]
        return self.frames[start : start + self.lengths[index]]


def normalization_stats(features: Features, mode: str = "reference"):
    """Mean and scale fit on the given (training) subset only.

    ``"standard"`` returns the population standard deviation. ``"reference"`` reproduces uclser20, whose
    ``spec_std = sqrt(sum((x - mean)^2)) / N`` equals ``std / sqrt(N)``: inputs end up scaled by a large
    constant. Batch norm after the bias-free first convolution absorbs most of this, but the authors'
    checkpoint expects it, so it is the default.
    """
    frames = features.frames
    count = frames.size
    total, total_sq = 0.0, 0.0
    for start in range(0, len(frames), 1 << 20):
        block = np.asarray(frames[start : start + (1 << 20)], dtype=np.float64)
        total += block.sum()
        total_sq += np.square(block).sum()
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean**2, 0.0))
    if mode == "standard":
        return mean, std
    if mode == "reference":
        return mean, std / math.sqrt(count)
    raise ValueError(f"Unknown normalization mode: {mode!r}")


# -- Augmentations (paper Sec. 2; uclser20 ``da/`` and ``FSDnoisy18k.py``) -----


def random_patch(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random 1 s patch; clips are at least ``PATCH_LEN`` frames long after replication."""
    start = rng.integers(0, len(clip) - PATCH_LEN + 1)
    return np.asarray(clip[start : start + PATCH_LEN], dtype=np.float32)


def mix_back(patch: np.ndarray, background: np.ndarray, alpha: float) -> np.ndarray:
    """Mix a background patch into ``patch`` in the linear domain (paper Eq. 1).

    The background is scaled to carry ``alpha`` times the energy of the patch, so the patch stays
    dominant whatever the relative loudness; the result is returned in the log10 domain.
    """
    patch_lin = np.power(10.0, patch, dtype=np.float64)
    back_lin = np.power(10.0, background, dtype=np.float64)
    beta = alpha * patch_lin.sum() / back_lin.sum()
    return np.log10((1 - alpha) * patch_lin + beta * back_lin).astype(np.float32)


def random_resized_crop(spec: torch.Tensor, rng: np.random.Generator, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)):
    """Crop a random time-frequency region of ``spec`` ``[T, F]`` and resize it back to ``[T, F]``.

    Same sampling as torchvision's ``RandomResizedCrop`` (uclser20 ``RandomResizedCrop_diy``), with
    width = frequency and height = time; bilinear resizing.
    """
    height, width = spec.shape
    area = height * width
    for _ in range(10):
        target_area = rng.uniform(*scale) * area
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            break
    else:
        h, w, top, left = height, width, 0, 0
    crop = spec[top : top + h, left : left + w][None, None]
    return F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)[0, 0]


class SimCLRPatches(Dataset):
    """Two augmented views per clip: random patches, mix-back, RRC, compander, Gaussian noise.

    No labels are returned: pretraining is unsupervised. Randomness comes from torch's per-worker
    generator, so a run is reproducible for a given seed and number of workers.
    """

    def __init__(self, features: Features, mean: float, std: float, alpha: float = 0.05):
        if len(features) < 3:
            raise ValueError("Mix-back needs at least three clips.")
        self.features, self.mean, self.std, self.alpha = features, mean, std, alpha

    def __len__(self):
        return len(self.features)

    def _view(self, index, background, rng):
        patch = random_patch(self.features.clip(index), rng)
        patch = mix_back(patch, random_patch(self.features.clip(background), rng), rng.uniform(0, self.alpha))
        spec = random_resized_crop(torch.from_numpy(patch), rng)
        spec = spec * rng.uniform(0.75, 1.0)  # compander (uclser20 ``comp_alpha=0.75``)
        spec = spec + torch.from_numpy(rng.normal(0, rng.uniform(0, 0.01), spec.shape).astype(np.float32))
        return ((spec - self.mean) / self.std)[None]  # [1, T, F]

    def __getitem__(self, index):
        rng = np.random.default_rng(int(torch.randint(0, 2**62, ()).item()))
        # Two distinct background clips, both different from the anchor clip (uclser20 ``mix_out_clip``).
        others = rng.choice(len(self.features) - 1, size=2, replace=False)
        backgrounds = [int(o) + (o >= index) for o in others]
        return {"views": [{"image": self._view(index, b, rng)} for b in backgrounds]}


# -- Encoder -----------------------------------------------------------------


def make_encoder() -> nn.Module:
    """ResNet-18 with a one-channel stem and global average pooling; outputs 512-d embeddings."""
    import torchvision

    model = torchvision.models.resnet18(num_classes=20)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()
    return model


def make_projector() -> nn.Module:
    """One-hidden-layer MLP head of the paper (512-512-512); the loss L2-normalises its output."""
    return nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, 512))


def load_encoder(spec: str, seed: int) -> nn.Module:
    """``random`` (seeded init), a state dict saved by ``pretrain``, or the uclser20 ``ResNet_best.pth``."""
    torch.manual_seed(seed)
    encoder = make_encoder()
    if spec == "random":
        return encoder
    state = torch.load(spec, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}  # uclser20 keeps its 20-way layer
    encoder.load_state_dict(state)
    return encoder


# -- Pretraining -------------------------------------------------------------


def pretrain(
    features: Features,
    mean,
    std,
    out: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    precision,
    milestones=(325, 425),
):
    """SimCLR with the paper's optimisation: SGD (lr 0.03, momentum 0.9, wd 1e-4), lr /10 at ``milestones`` epochs."""
    import lightning.pytorch as pl
    import stable_pretraining as spt
    from stable_pretraining.forward import simclr
    from stable_pretraining.losses import NTXEntLoss

    class History(pl.Callback):
        def __init__(self):
            self.rows, self.total, self.count, self.started = [], 0.0, 0, time.monotonic()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = float((outputs["loss"] if isinstance(outputs, dict) else outputs).detach())
            if not math.isfinite(loss):
                raise ValueError("Non-finite contrastive loss.")
            self.total, self.count = self.total + loss, self.count + 1

        def on_train_epoch_end(self, trainer, module):
            row = {
                "epoch": trainer.current_epoch + 1,
                "loss": self.total / self.count,
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "seconds": time.monotonic() - self.started,
            }
            self.rows.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(json.dumps(self.rows, indent=2) + "\n")
            self.total, self.count = 0.0, 0
            if row["epoch"] % 50 == 0:
                torch.save(module.backbone.state_dict(), out / f"encoder_epoch{row['epoch']}.pt")

    spt.set(verbose="ERROR")
    pl.seed_everything(seed, workers=True)
    dataset = SimCLRPatches(features, mean, std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    encoder = make_encoder()
    module = spt.Module(
        forward=simclr,
        backbone=encoder,
        projector=make_projector(),
        simclr_loss=NTXEntLoss(temperature=0.2),
        optim={
            "optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9, "weight_decay": 1e-4},
            # spt.Module steps schedulers after every optimizer step, so milestones are given in steps.
            "scheduler": {"type": "MultiStepLR", "milestones": [m * len(loader) for m in milestones], "gamma": 0.1},
            "interval": "step",
        },
    )
    history = History()
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[history],
        default_root_dir=str(out),
    )
    trainer.fit(module, train_dataloaders=loader)
    torch.save(encoder.state_dict(), out / "encoder.pt")
    return history.rows


# -- Linear evaluation -------------------------------------------------------


def _patches(features: Features, index: int, mean, std) -> torch.Tensor:
    clip = np.asarray(features.clip(index), dtype=np.float32)
    patches = np.stack([clip[s : s + PATCH_LEN] for s in patch_starts(len(clip))])
    return (torch.from_numpy(patches)[:, None] - mean) / std


@torch.no_grad()
def recalibrate_batch_norm(encoder: nn.Module, features: Features, mean, std, device, seed: int, batches: int = 200):
    """Re-estimate batch-norm statistics on un-augmented training patches.

    uclser20 runs its frozen encoder in train mode during linear evaluation, so its batch-norm
    statistics move to the clean (non-mixed, non-augmented) input distribution; this does the same
    once, up front, and then keeps the encoder in eval mode.
    """
    rng = np.random.default_rng(seed)
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None  # cumulative average
    encoder.train()
    for _ in range(batches):
        index = rng.integers(0, len(features), 128)
        batch = np.stack([random_patch(features.clip(int(i)), rng) for i in index])
        encoder(((torch.from_numpy(batch)[:, None] - mean) / std).to(device))
    encoder.eval()


@torch.no_grad()
def embed(encoder: nn.Module, features: Features, mean, std, device, batch_size: int = 512):
    """Embeddings of every evaluation patch: ``(embeddings [P, 512], clip_of_patch [P])``."""
    encoder.eval()
    embeddings, clip_ids, pending = [], [], []

    def flush():
        embeddings.append(encoder(torch.cat(pending).to(device)).float().cpu())
        pending.clear()

    for index in range(len(features)):
        patches = _patches(features, index, mean, std)
        pending.append(patches)
        clip_ids.extend([index] * len(patches))
        if sum(len(p) for p in pending) >= batch_size:
            flush()
    if pending:
        flush()
    return torch.cat(embeddings), torch.as_tensor(clip_ids)


def clip_predictions(logits: torch.Tensor, clip_ids: torch.Tensor, n_clips: int) -> torch.Tensor:
    """Average patch softmax outputs per clip and return the clip-level class."""
    probs = torch.zeros(n_clips, logits.shape[1]).index_add_(0, clip_ids, logits.softmax(1))
    return probs.argmax(1)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over the classes present in ``labels`` (the paper's accuracy)."""
    recalls = [np.mean(predictions[labels == c] == c) for c in range(n_classes) if np.any(labels == c)]
    return float(np.mean(recalls))


def _metrics(labels, predictions, n_classes):
    return {
        "balanced_accuracy": balanced_accuracy(labels, predictions, n_classes),
        "accuracy": float(np.mean(labels == predictions)),
    }


def linear_probe(train, val, test, n_classes: int, seed: int, epochs: int = 200, batch_size: int = 128):
    """Train a linear classifier on frozen patch embeddings (uclser20 ``lin_eval``).

    Each split is ``(embeddings, clip_ids, clip_labels)``. Every epoch draws one random evaluation
    patch per training clip; SGD (lr 0.1, momentum 0.9, weight decay 1e-4, /10 at epochs 80 and 160).
    The epoch is selected on validation balanced accuracy and only that classifier sees the test set.
    """
    torch.manual_seed(seed)  # classifier initialisation
    generator = torch.Generator().manual_seed(seed)
    train_emb, train_clips, train_labels = train
    patches_of_clip = [torch.nonzero(train_clips == c).flatten() for c in range(len(train_labels))]
    first = torch.as_tensor([p[0] for p in patches_of_clip])
    count = torch.as_tensor([len(p) for p in patches_of_clip])
    targets = torch.as_tensor(train_labels)

    classifier = nn.Linear(train_emb.shape[1], n_classes)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], gamma=0.1)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        pick = first + (torch.rand(len(first), generator=generator) * count).long()
        order = torch.randperm(len(pick), generator=generator)
        for batch in order.split(batch_size):
            loss = F.cross_entropy(classifier(train_emb[pick[batch]]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            prediction = clip_predictions(classifier(val[0]), val[1], len(val[2])).numpy()
        row = {"epoch": epoch, **_metrics(val[2], prediction, n_classes)}
        history.append(row)
        if best is None or row["balanced_accuracy"] > best[0]["balanced_accuracy"]:
            best = (row, {k: v.clone() for k, v in classifier.state_dict().items()})
    classifier.load_state_dict(best[1])
    with torch.no_grad():
        prediction = clip_predictions(classifier(test[0]), test[1], len(test[2])).numpy()
    result = {
        "selected_epoch": best[0]["epoch"],
        "validation": best[0],
        "test": _metrics(test[2], prediction, n_classes),
    }
    return result, history, prediction, classifier


def probe(root: Path, encoder_spec: str, out: Path, seed: int, normalization: str, device, bn_batches: int = 200):
    subsets = {name: Features(root, name) for name in SUBSETS}
    mean, std = normalization_stats(subsets["noisy"], normalization)
    encoder = load_encoder(encoder_spec, seed).to(device)
    recalibrate_batch_norm(encoder, subsets["noisy"], mean, std, device, seed, bn_batches)
    splits = {}
    for name, features in subsets.items():
        embeddings, clip_ids = embed(encoder, features, mean, std, device)
        splits[name] = (embeddings, clip_ids, features.labels)
    n_classes = len(subsets["noisy"].label_names)
    result, history, prediction, classifier = linear_probe(
        splits["noisy"], splits["clean"], splits["test"], n_classes, seed
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "predictions.npz", target=subsets["test"].labels, prediction=prediction)
    np.savez(out / "classifier.npz", weight=classifier.weight.detach().numpy(), bias=classifier.bias.detach().numpy())
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {
        "encoder": encoder_spec,
        "seed": seed,
        "normalization": normalization,
        "bn_batches": bn_batches,
        **result,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# -- Provenance and CLI ------------------------------------------------------


def provenance(args) -> dict:
    source = Path(__file__).read_bytes()
    versions = {}
    for package in ("torch", "torchvision", "numpy", "lightning", "stable-pretraining", "stable-datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    features = commands.add_parser("features", help="Compute log-mel features from the FSDnoisy18k builder.")
    features.add_argument("--out", type=Path, required=True)

    train = commands.add_parser("pretrain", help="SimCLR pretraining on the noisy train subset.")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--precision", default="32-true", help="Lightning precision, e.g. bf16-mixed on GPUs.")
    train.add_argument("--normalization", choices=("reference", "standard"), default="reference")

    evaluate = commands.add_parser("probe", help="Frozen linear evaluation (clean = validation, test once).")
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--encoder", required=True, help="'random', an encoder.pt, or uclser20 ResNet_best.pth.")
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--normalization", choices=("reference", "standard"), default="reference")
    evaluate.add_argument("--bn-batches", type=int, default=200, help="Batches of 128 for batch-norm recalibration.")

    args = parser.parse_args(argv)
    if args.command == "features":
        counts = build_features(args.out)
        (args.out / "features_config.json").write_text(json.dumps({**provenance(args), "counts": counts}, indent=2))
        print(json.dumps(counts))
    elif args.command == "pretrain":
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "config.json").write_text(json.dumps(provenance(args), indent=2) + "\n")
        noisy = Features(args.features, "noisy")
        mean, std = normalization_stats(noisy, args.normalization)
        pretrain(noisy, mean, std, args.out, args.seed, args.epochs, args.batch_size, args.workers, args.precision)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "config.json").write_text(json.dumps(provenance(args), indent=2) + "\n")
        result = probe(
            args.features,
            args.encoder,
            args.out,
            args.seed,
            args.normalization,
            device_of(args.device),
            args.bn_batches,
        )
        print(json.dumps({"selected_epoch": result["selected_epoch"], "test": result["test"]}))


if __name__ == "__main__":
    main()
