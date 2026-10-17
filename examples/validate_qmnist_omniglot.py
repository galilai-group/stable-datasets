"""Reproducible CPU validation for QMNIST and Omniglot ingestion.

QMNIST: a small CNN, fixed training schedule, final-model test evaluation.
Omniglot: the official fixed one-shot tasks, using Modified Hausdorff Distance
or features from a CNN trained on background characters.
Neither validation tunes hyperparameters or selects checkpoints on test data.
"""

import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stable_datasets.images import QMNIST, Omniglot
from stable_datasets.schema import DownloadInfo
from stable_datasets.utils import download


def _dataset_kwargs(args):
    return {"download_dir": args.cache_dir / "downloads", "processed_cache_dir": args.cache_dir / "processed"}


def _tensors(dataset, resize=False):
    array = np.empty((len(dataset), 1, 28, 28), dtype=np.uint8)
    labels = np.empty(len(dataset), dtype=np.int64)
    for index, row in enumerate(dataset.with_format("numpy")):
        image = row["image"]
        if resize:
            image = np.asarray(Image.fromarray(image).resize((28, 28), Image.Resampling.BILINEAR))
        array[index, 0] = image
        labels[index] = row["label"]
    return torch.from_numpy(array).float().div_(255), torch.from_numpy(labels)


def _train_cnn(x, y, num_classes, args):
    # These are explicit validation choices, not hyperparameters from the paper.
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, generator=generator)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
        entry = {"epoch": epoch + 1, "train_loss": total_loss / len(y), "train_accuracy": correct / len(y)}
        history.append(entry)
        print(json.dumps(entry), flush=True)
    model.eval()
    return model, history


def validate_qmnist(args):
    torch.set_num_threads(args.threads)
    datasets = QMNIST(**_dataset_kwargs(args))
    x, y = _tensors(datasets["train"])
    model, history = _train_cnn(x, y, 10, args)
    results = {}
    with torch.no_grad():
        for split in ("test10k", "test50k"):
            test_x, test_y = _tensors(datasets[split])
            predictions = torch.cat([model(batch).argmax(1) for batch in test_x.split(args.batch_size)])
            correct = int((predictions == test_y).sum())
            results[split] = {"correct": correct, "count": len(test_y), "accuracy": correct / len(test_y)}
            print(json.dumps({split: results[split]}), flush=True)
    return {
        "dataset": "QMNIST",
        "source": QMNIST.SOURCE.homepage,
        "source_revision": "94061367ec2e099ea18c553c75727ecf849ddb9b",
        "method": "Small supervised CNN; independent loader-validation baseline, not paper reproduction",
        "architecture": str(model),
        "configuration": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "weight_decay": 0,
            "preprocessing": "Original 28x28 grayscale pixels divided by 255; no augmentation",
            "training_split": "train (all 60000)",
            "checkpoint_selection": "final epoch only",
            "threads": args.threads,
        },
        "training_history": history,
        "results": results,
    }


def _ink_points(image):
    # The authors' reference protocol centers coordinates of all black pixels.
    points = np.column_stack(np.nonzero(np.asarray(image) == 0)).astype(np.float64)
    if not len(points):
        raise ValueError("An Omniglot image has no ink pixels")
    return points - points.mean(axis=0, keepdims=True)


def _modified_hausdorff(first, second):
    distances = cdist(first, second)
    return max(distances.min(axis=1).mean(), distances.min(axis=0).mean())


def validate_omniglot(args):
    model, history = None, None
    if args.omniglot_method == "cnn":
        torch.set_num_threads(args.threads)
        background = Omniglot(split="background", **_dataset_kwargs(args))
        x, y = _tensors(background, resize=True)
        model, history = _train_cnn(x, y, 964, args)
    dataset = Omniglot(split="evaluation", **_dataset_kwargs(args))
    task_asset = DownloadInfo(
        url=(
            "https://raw.githubusercontent.com/brendenlake/omniglot/"
            "057f034baf2ecb8530bc5710e5a23584d2a519cc/python/one-shot-classification/all_runs.zip"
        ),
        checksum="sha256:437c1eeab0af8484112e1b4cc0f3d541ec727f7c47b9b202da738af9dfd6c325",
    )
    tasks_path = download(task_asset, dest_folder=args.cache_dir / "downloads")
    # Match the official task files to decoded builder pixels. Distances below
    # use builder output, ensuring this evaluates the ingestion path itself.
    task_hashes = {}
    with zipfile.ZipFile(tasks_path) as archive:
        for name in archive.namelist():
            if name.startswith("run") and name.endswith(".png"):
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    task_hashes[name] = hashlib.sha256(image.convert("L").tobytes()).hexdigest()
        required = set(task_hashes.values())
        points_by_hash, labels_by_hash = {}, {}
        for row in dataset:
            digest = hashlib.sha256(row["image"].tobytes()).hexdigest()
            if digest in required:
                if model is None:
                    points_by_hash[digest] = _ink_points(row["image"])
                else:
                    pixels = np.asarray(row["image"].resize((28, 28), Image.Resampling.BILINEAR))
                    points_by_hash[digest] = pixels
                labels_by_hash.setdefault(digest, set()).add(row["label"])
        missing = required - set(points_by_hash)
        if missing:
            raise ValueError(f"{len(missing)} official task images do not match builder evaluation pixels")
        if model is not None:
            keys = list(points_by_hash)
            images = torch.from_numpy(np.stack([points_by_hash[key] for key in keys])[:, None]).float() / 255
            encoder = model[:-1]
            with torch.no_grad():
                embeddings = torch.cat([encoder(batch) for batch in images.split(args.batch_size)]).numpy()
            points_by_hash = dict(zip(keys, embeddings))
        runs = []
        for number in range(1, 21):
            run = f"run{number:02d}"
            pairs = [line.split() for line in archive.read(f"{run}/class_labels.txt").decode().splitlines()]
            answers = dict(pairs)
            queries, supports = sorted(answers), sorted(set(answers.values()))
            if len(queries) != 20 or len(supports) != 20:
                raise ValueError(f"Expected 20-way one-shot task in {run}")
            correct = 0
            for query in queries:
                query_hash = task_hashes[query]
                target_hash = task_hashes[answers[query]]
                if not labels_by_hash[query_hash] & labels_by_hash[target_hash]:
                    raise ValueError(f"Official task labels disagree with builder character IDs: {query}")
                if model is None:
                    costs = [
                        _modified_hausdorff(points_by_hash[query_hash], points_by_hash[task_hashes[s]])
                        for s in supports
                    ]
                else:
                    costs = [
                        np.linalg.norm(points_by_hash[query_hash] - points_by_hash[task_hashes[s]]) for s in supports
                    ]
                prediction = supports[int(np.argmin(costs))]
                correct += int(prediction == answers[query])
            entry = {"run": run, "correct": correct, "count": 20, "accuracy": correct / 20}
            runs.append(entry)
            print(json.dumps(entry), flush=True)
    total_correct = sum(entry["correct"] for entry in runs)
    result = {
        "dataset": "Omniglot",
        "source": Omniglot.SOURCE.homepage,
        "source_revision": "057f034baf2ecb8530bc5710e5a23584d2a519cc",
        "method": "Modified Hausdorff Distance nearest neighbor, authors' reference baseline",
        "configuration": {
            "protocol": "20 official fixed within-alphabet 20-way 1-shot tasks; 400 queries",
            "preprocessing": "Original pixels; centered coordinates of black pixels; no resizing or rotation",
            "seed": None,
            "learned_parameters": False,
            "training": "No parametric training; one labeled support example per class in each official task",
            "model_selection": "None",
        },
        "verified_task_images": len(task_hashes),
        "runs": runs,
        "results": {
            "correct": total_correct,
            "count": 400,
            "accuracy": total_correct / 400,
            "error_percent": 100 * (1 - total_correct / 400),
        },
    }
    if model is not None:
        result["method"] = (
            "Supervised background CNN features + Euclidean 1-nearest-neighbor on official one-shot tasks"
        )
        result["architecture"] = str(model)
        result["training_history"] = history
        result["configuration"].update(
            {
                "preprocessing": "Bilinear resize to 28x28 and divide pixels by 255, both background and task images; no augmentation",
                "seed": args.seed,
                "learned_parameters": True,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "learning_rate": args.learning_rate,
                "adam_betas": [0.9, 0.999],
                "adam_eps": 1e-8,
                "weight_decay": 0,
                "training": "All 19280 background images, 964-way supervised classification; evaluation classes are unseen",
                "model_selection": "Final epoch only; no evaluation-based tuning",
                "embedding": "64-dimensional penultimate ReLU activations, no normalization",
                "threads": args.threads,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("qmnist", "omniglot"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".stable_datasets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--omniglot-method", choices=("mhd", "cnn"), default="mhd")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing result: {args.output}")
    if min(args.epochs, args.batch_size, args.learning_rate, args.threads) <= 0:
        parser.error("epochs, batch size, learning rate, and threads must be positive")
    start = time.perf_counter()
    result = validate_qmnist(args) if args.dataset == "qmnist" else validate_omniglot(args)
    result["elapsed_seconds"] = time.perf_counter() - start
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "CPU",
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
