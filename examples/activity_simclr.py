"""Subject-disjoint SimCLR + frozen linear probes for UCI HAR and MHEALTH.

Run from the repository root; install the ``benchmarks`` extra first.
Outputs (including checkpoints and predictions) belong outside the repository.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_datasets.timeseries import MHEALTH, UCIHAR


SPLITS = ("train", "validation", "test")
MOTION_CHANNELS = [i for i in range(23) if i not in (3, 4)]


def mhealth_windows(record):
    """Keep pure non-null windows on the original subject's 64-sample grid."""
    signals = np.asarray(record["series"], dtype=np.float32)
    labels = np.asarray(record["label"], dtype=np.int64)
    for start in range(0, len(labels) - 127, 64):
        target = labels[start : start + 128]
        if target[0] != 0 and np.all(target == target[0]):
            yield signals[start : start + 128, MOTION_CHANNELS], int(target[0]) - 1, start


def prepare_data(name):
    rows = {split: [] for split in SPLITS}
    if name == "uci_har":
        data = UCIHAR()
        for source_split in ("train", "test"):
            for index, row in enumerate(data[source_split]):
                subject = row["subject_id"]
                split = "test" if source_split == "test" else "validation" if subject in (27, 28, 29, 30) else "train"
                rows[split].append((row["series"], row["label"], subject, index))
    elif name == "mhealth":
        for row in MHEALTH(split="all"):
            subject = row["subject_id"]
            split = "train" if subject <= 6 else "validation" if subject <= 8 else "test"
            for series, label, start in mhealth_windows(row):
                rows[split].append((series, label, subject, start))
    else:
        raise ValueError(f"Unknown activity dataset: {name}")
    arrays = {}
    for split, examples in rows.items():
        series, labels, subjects, indices = zip(*examples)
        arrays[split] = {
            "series": np.asarray(series, dtype=np.float32),
            "label": np.asarray(labels, dtype=np.int64),
            "subject_id": np.asarray(subjects, dtype=np.int64),
            "source_index": np.asarray(indices, dtype=np.int64),
        }
    subjects = [set(arrays[s]["subject_id"]) for s in SPLITS]
    assert all(subjects[i].isdisjoint(subjects[j]) for i in range(3) for j in range(i))
    return arrays


def standardize(arrays):
    """Fit channel statistics only on training windows, including their overlap."""
    train = arrays["train"]["series"]
    mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1
    tensors = {
        split: torch.from_numpy(np.ascontiguousarray(((rows["series"] - mean) / std).transpose(0, 2, 1)))
        for split, rows in arrays.items()
    }
    return tensors, mean, std


class TwoViews(Dataset):
    def __init__(self, series):
        self.series = series

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        x = self.series[index]
        views = []
        for _ in range(2):
            scale = torch.empty(x.shape[0], 1).uniform_(0.9, 1.1)
            views.append({"image": x * scale + torch.randn_like(x) * 0.02})
        # The upstream forward calls its input "image", but accepts 1D signals.
        return {"views": views}


def make_encoder(channels):
    layers = []
    for output, kernel in ((32, 7), (64, 5), (128, 3)):
        layers.extend(
            [
                nn.Conv1d(channels, output, kernel, stride=2, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(output),
                nn.ReLU(),
            ]
        )
        channels = output
    return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())


@torch.inference_mode()
def embeddings(encoder, tensors):
    encoder.eval()
    return {
        split: torch.cat([encoder(batch) for batch in series.split(256)]).numpy() for split, series in tensors.items()
    }


def metrics(y, prediction, classes):
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, prediction, labels=classes, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist(),
    }


def linear_probe(features, arrays):
    """Select C using validation macro-F1, then evaluate the test set once."""
    classes = np.unique(arrays["train"]["label"])
    candidates, best, best_score = [], None, -1
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, solver="lbfgs"))
        model.fit(features["train"], arrays["train"]["label"])
        validation = metrics(arrays["validation"]["label"], model.predict(features["validation"]), classes)
        candidates.append({"C": c, "validation": validation})
        if validation["macro_f1"] > best_score:
            best, best_score, chosen_c = model, validation["macro_f1"], c
    prediction = best.predict(features["test"])
    result = {"C": chosen_c, "candidates": candidates, "test": metrics(arrays["test"]["label"], prediction, classes)}
    return result, prediction, best


def train(encoder, series, seed, epochs, output):
    import lightning.pytorch as pl
    import stable_pretraining as spt
    from stable_pretraining.forward import simclr
    from stable_pretraining.losses import NTXEntLoss

    class History(pl.Callback):
        def __init__(self):
            self.rows = []
            self.total, self.count = 0.0, 0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = float(loss.detach())
            if not np.isfinite(loss):
                raise ValueError("Nonfinite contrastive loss")
            self.total += loss
            self.count += 1

        def on_train_epoch_end(self, trainer, module):
            self.rows.append({"epoch": trainer.current_epoch + 1, "loss": self.total / self.count})
            print(f"epoch={trainer.current_epoch + 1} loss={self.rows[-1]['loss']:.5f}", flush=True)
            (output / "history.json").write_text(json.dumps(self.rows, indent=2) + "\n")
            self.total, self.count = 0.0, 0

    spt.set(verbose="ERROR")
    pl.seed_everything(seed, workers=True)
    loader = DataLoader(TwoViews(series), batch_size=128, shuffle=True, drop_last=True, num_workers=0)
    module = spt.Module(
        forward=simclr,
        backbone=encoder,
        projector=nn.Sequential(nn.Linear(128, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64)),
        simclr_loss=NTXEntLoss(temperature=0.5),
        optim={
            "optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
            "scheduler": {"type": "CosineAnnealingLR", "T_max": epochs * len(loader)},
            "interval": "step",
        },
    )
    history = History()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[history],
        default_root_dir=str(output),
    )
    module.train()
    trainer.fit(module, train_dataloaders=loader)
    assert encoder[1].num_batches_tracked.item() == 2 * epochs * len(loader)
    torch.save(module.state_dict(), output / "simclr.pt")
    return history.rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci_har", "mhealth"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Epochs must be positive and seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    arrays = prepare_data(args.dataset)
    tensors, mean, std = standardize(arrays)
    np.savez(args.output / "normalization.npz", mean=mean, std=std)
    np.savez(
        args.output / "split_manifest.npz",
        **{
            f"{split}_{key}": values[key]
            for split, values in arrays.items()
            for key in ("label", "subject_id", "source_index")
        },
    )
    counts = {
        split: {
            "windows": len(rows["label"]),
            "subjects": np.unique(rows["subject_id"]).tolist(),
            "class_counts": np.bincount(rows["label"]).tolist(),
        }
        for split, rows in arrays.items()
    }
    source = Path(__file__).read_bytes()
    (args.output / "activity_simclr.py").write_bytes(source)
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "device": "cpu",
        "threads": 4,
        "batch_size": 128,
        "drop_last": True,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.5,
        "scaling": [0.9, 1.1],
        "noise_std": 0.02,
        "scheduler": "cosine per step",
        "selection": "final fixed epoch; linear C selected on validation macro-F1 only",
        "counts": counts,
        "dataset_sha256": next(
            iter((UCIHAR if args.dataset == "uci_har" else MHEALTH).SOURCE["assets"].values())
        ).checksum,
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "scikit-learn", "lightning", "stable-pretraining")
        },
        "stable_pretraining_source": json.loads(
            importlib.metadata.distribution("stable-pretraining").read_text("direct_url.json") or "null"
        ),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(counts), flush=True)
    results = []
    for seed in args.seeds:
        started = time.monotonic()
        run_dir = args.output / f"seed_{seed}"
        run_dir.mkdir()
        torch.manual_seed(seed)
        encoder = make_encoder(tensors["train"].shape[1])
        torch.save(encoder.state_dict(), run_dir / "initial_encoder.pt")
        initial = encoder[0].weight.detach().clone()
        baseline, initial_prediction, baseline_model = linear_probe(embeddings(encoder, tensors), arrays)
        history = train(encoder, tensors["train"], seed, args.epochs, run_dir)
        assert not torch.equal(initial, encoder[0].weight.detach())
        trained, prediction, trained_model = linear_probe(embeddings(encoder, tensors), arrays)
        torch.save(encoder.state_dict(), run_dir / "encoder.pt")
        # Store numeric probe parameters, avoiding pickle for the classifier.
        for name, model in (("random", baseline_model), ("simclr", trained_model)):
            scaler, classifier = model.steps[0][1], model.steps[1][1]
            np.savez(
                run_dir / f"{name}_probe.npz",
                mean=scaler.mean_,
                scale=scaler.scale_,
                coef=classifier.coef_,
                intercept=classifier.intercept_,
                classes=classifier.classes_,
            )
        np.savez(
            run_dir / "predictions.npz", target=arrays["test"]["label"], random=initial_prediction, simclr=prediction
        )
        result = {
            "seed": seed,
            "random_encoder": baseline,
            "simclr": trained,
            "history": history,
            "seconds": time.monotonic() - started,
        }
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(
            f"seed={seed} random={baseline['test']['accuracy']:.4f} simclr={trained['test']['accuracy']:.4f}",
            flush=True,
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
