"""CPU DINO transfer experiments for the audited Scene8/ISIC2016 protocols.

Run ``python -m benchmarks.transfer.scene8_isic2016 --help`` from a checkout.
The pilot, feature extraction, and classifier evaluation are separate stages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from threadpoolctl import threadpool_limits
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from benchmarks.protocols import ImageRecord, isic2016_protocol, scene8_protocol
from benchmarks.transfer.protocol_probe import (
    classification_metrics,
    content_hash,
    fit_select_refit,
    label_budget,
    normalize_features,
)
from stable_datasets.images import ISIC2016, Scene8


ASSET_LOCK = Path(__file__).with_name("dino_vits16_assets.json")
PREPROCESS = {
    "rgb": True,
    "shorter_edge": 256,
    "crop": 224,
    "interpolation": "PIL bicubic",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "feature": "last-block normalized CLS token, 384 dimensions",
}


def file_hash(path):
    """Hash files in bounded blocks, including large checkpoints and caches."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Commit a JSON marker only after its complete payload has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_arrays(path, **arrays):
    """Write numeric/string arrays without pickle or incomplete final files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def environment():
    """Record versions affecting decoding, inference, or classifier fitting."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "scipy",
                "scikit-learn",
                "pillow",
                "pyarrow",
                "stable-datasets",
            )
        },
    }


def source_hashes():
    """Identify executed project code even when changes are not committed."""
    paths = [
        Path(__file__),
        Path(__file__).with_name("protocol_probe.py"),
        ASSET_LOCK,
        Path("benchmarks/protocols.py"),
        Path("stable_datasets/images/scene8.py"),
        Path("stable_datasets/images/isic2016.py"),
        Path("benchmarks/SCENE8_ISIC2016_EXPERIMENT_PROTOCOL.md"),
    ]
    return {str(path.relative_to(Path.cwd()) if path.is_absolute() else path): file_hash(path) for path in paths}


def load_dino(asset_dir):
    """Load the pinned original implementation with strict official weights."""
    lock = json.loads(ASSET_LOCK.read_text())
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for name, info in lock["files"].items():
        path = asset_dir / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with requests.get(info["url"], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if file_hash(temporary) != info["sha256"]:
                raise ValueError(f"Downloaded asset checksum differs: {name}")
            temporary.replace(path)
        if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
            raise ValueError(f"Pinned asset differs: {name}")
    model = torch.hub.load(str(asset_dir.resolve()), "dino_vits16", source="local", pretrained=False)
    state = torch.load(asset_dir / "dino_deitsmall16_pretrain.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    # The selected output must be exactly the original final-block CLS path.
    fixed = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(fixed)
        reference = model.get_intermediate_layers(fixed, n=1)[0][:, 0]
    if output.shape != (1, 384) or not torch.equal(output, reference):
        raise ValueError("Official DINO forward differs from the declared final-block feature.")
    return model, lock


def eval_transform():
    """Deterministic official-style single-crop evaluation preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
        ]
    )


class SourceSamples:
    """Index original cache rows by ID without decoding Arrow record batches."""

    def __init__(self, datasets, records):
        self.records = records
        self.locations = {}
        expected = {r.image_id: r.label for r in records}
        for dataset in datasets:
            ids = dataset.table.column("image_id").to_pylist()
            labels = dataset.table.column("label").to_pylist()
            for index, (image_id, label) in enumerate(zip(ids, labels)):
                if image_id in self.locations or expected.get(image_id) != label:
                    raise ValueError(f"Source ID/label differs from audited records: {image_id}")
                self.locations[image_id] = (dataset, index)
        if self.locations.keys() != expected.keys():
            raise ValueError("Source dataset coverage differs from audited records.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        dataset, row = self.locations[self.records[index].image_id]
        return dataset[row]


class IndexedSamples:
    """Keep pilot decoding inside the timed, bounded extraction loop."""

    def __init__(self, source, indices):
        self.source, self.indices = source, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


def extract_features(model, samples, records, batch_size, transform=None, dimension=384, progress=False):
    """Extract all rows, including the tail; verify live pixels against the audit."""
    if batch_size < 1 or len(samples) != len(records) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Extraction requires a frozen model, positive batch size and complete source records.")
    transform = transform or eval_transform()
    model.eval()
    embeddings, ids, labels = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            tensors = []
            for index in range(start, min(start + batch_size, len(records))):
                sample, record = samples[index], records[index]
                image = sample["image"].convert("RGB")
                pixel_hash = hashlib.sha256(
                    f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()
                ).hexdigest()
                if (sample["image_id"], sample["label"], pixel_hash) != (
                    record.image_id,
                    record.label,
                    record.pixel_sha256,
                ):
                    raise ValueError(f"Live source sample differs from audit: {record.image_id}")
                tensors.append(transform(image))
                ids.append(record.image_id)
                labels.append(record.label)
            values = model(torch.stack(tensors)).cpu().numpy()
            if values.shape != (len(tensors), dimension):
                raise ValueError("Encoder returned an unexpected feature shape.")
            embeddings.append(values)
            if progress and (start == 0 or (start // batch_size + 1) % 25 == 0):
                print(
                    json.dumps(
                        {"extracted": len(ids), "total": len(records), "seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
    features = np.concatenate(embeddings)
    if (
        len(set(ids)) != len(records)
        or not np.isfinite(features).all()
        or np.any(np.linalg.norm(features, axis=1) == 0)
    ):
        raise ValueError("Incomplete, repeated or invalid extracted features.")
    return {"image_ids": np.asarray(ids), "labels": np.asarray(labels, dtype=np.int64), "features": features}


def feature_identity(records, lock, threads):
    """Full-source frozen features can safely serve different saved splits."""
    return {
        "schema": "dino-feature-cache-v1",
        "records_sha256": content_hash([asdict(r) for r in records]),
        "assets": lock,
        "preprocess": PREPROCESS,
        "environment": environment(),
        "source_sha256": source_hashes(),
        "cpu_threads": threads,
        "precision": "float32",
        "builders": {"scene8": str(Scene8.VERSION), "isic2016": str(ISIC2016.VERSION)},
    }


def feature_cache(directory, identity, records, compute=None):
    """Validate committed cache metadata and payload before accepting a hit."""
    key = content_hash(identity)
    directory = Path(directory) / key
    marker, payload = directory / "metadata.json", directory / "features.npz"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata["identity"] != identity or file_hash(payload) != metadata["payload_sha256"]:
            raise ValueError("Feature-cache identity or payload checksum differs.")
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ("image_ids", "labels", "features")}
    else:
        if compute is None:
            raise FileNotFoundError("No completed feature cache; run the extract stage first.")
        started = time.perf_counter()
        arrays = compute()
        _validate_feature_arrays(arrays, records)
        write_arrays(payload, **arrays)
        metadata = {
            "identity": identity,
            "payload_sha256": file_hash(payload),
            "seconds": time.perf_counter() - started,
        }
        write_json(marker, metadata)
    _validate_feature_arrays(arrays, records)
    return arrays, {"key": key, "payload_sha256": metadata["payload_sha256"], "seconds": metadata["seconds"]}


def _validate_feature_arrays(arrays, records):
    if arrays["image_ids"].tolist() != [r.image_id for r in records] or arrays["labels"].tolist() != [
        r.label for r in records
    ]:
        raise ValueError("Cached feature/label/ID alignment differs.")
    if arrays["features"].shape != (len(records), 384):
        raise ValueError("Cached feature dimensions differ.")
    normalize_features(arrays["features"])


def read_protocols(directory):
    """Require saved manifests to match the declared algorithm and source rows."""
    directory = Path(directory)
    records = {
        name: [ImageRecord(**r) for r in json.loads((directory / f"{name}.records.json").read_text())]
        for name in ("scene8", "isic2016_train", "isic2016_test")
    }
    manifests = {}
    for seed in (42, 43, 44):
        expected = [scene8_protocol(records["scene8"], seed)] + [
            isic2016_protocol(
                records["isic2016_train"], records["isic2016_test"], seed, exclude_test_duplicates=filtered
            )
            for filtered in (False, True)
        ]
        for value in expected:
            path = directory / f"{value.protocol}-seed{seed}.json"
            saved = json.loads(path.read_text())
            if content_hash(saved) != content_hash(asdict(value)):
                raise ValueError(f"Saved protocol differs from declared assignment: {path.name}")
            manifests[(value.protocol, seed)] = saved
    return {"scene8": records["scene8"], "isic2016": records["isic2016_train"] + records["isic2016_test"]}, manifests


def preprocessing_montage(samples, ids, output):
    """Save source thumbnails beside model crops, using fitting images only."""
    by_id = {r.image_id: i for i, r in enumerate(samples.records)}
    canvas = Image.new("RGB", (448, 250 * len(ids)), "white")
    draw = ImageDraw.Draw(canvas)
    crop = transforms.Compose(
        [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224)]
    )
    for row, image_id in enumerate(ids):
        image = samples[by_id[image_id]]["image"].convert("RGB")
        thumbnail = image.copy()
        thumbnail.thumbnail((224, 224))
        canvas.paste(thumbnail, (0, row * 250 + 24))
        canvas.paste(crop(image), (224, row * 250 + 24))
        draw.text((4, row * 250 + 4), f"{image_id}: source / model crop", fill="black")
    canvas.save(output)


def peak_memory_bytes():
    """Read process peak resident memory, including native tensor allocations."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)


def evaluate_matrix(records, manifests, feature_sets, feature_metadata, output_dir):
    """Run the locked matrix, keeping selection separate from test evaluation."""
    output_dir = Path(output_dir)
    results = []
    selected = {}
    for name in ("scene8", "isic2016"):
        arrays = feature_sets[name]
        embeddings = normalize_features(arrays["features"])
        lookup = {image_id: i for i, image_id in enumerate(arrays["image_ids"].tolist())}
        metric = "balanced_accuracy" if name == "scene8" else "average_precision"
        protocol = "scene8-pixel-groups-v1" if name == "scene8" else "isic2016-train-filtered-pixel-groups-v2"
        for seed in (42, 43, 44):
            for companion in (False, True) if name == "isic2016" else (False,):
                manifest = manifests[("isic2016-official-pixel-groups-v2" if companion else protocol, seed)]
                for fraction in (1.0,) if companion else (0.1, 0.25, 1.0):
                    budget = label_budget(manifest, records[name], fraction, seed + 1000)
                    roles = budget["roles"]
                    indices = {
                        role: np.asarray([lookup[i] for i in roles[role]]) for role in ("fit", "validation", "test")
                    }
                    fit_features = {role: embeddings[indices[role]] for role in ("fit", "validation")}
                    fit_labels = {role: arrays["labels"][indices[role]] for role in ("fit", "validation")}
                    for method in ("linear", "knn") if companion else ("prior", "linear", "knn"):
                        started = time.perf_counter()
                        parameter = selected[(seed, method)] if companion else None
                        probe, selection = fit_select_refit(method, fit_features, fit_labels, roles, metric, parameter)
                        if name == "isic2016" and not companion and fraction == 1:
                            selected[(seed, method)] = selection["selected_parameter"]
                        probabilities = probe.predict_proba(embeddings[indices["test"]])
                        labels = arrays["labels"][indices["test"]]
                        result = {
                            "dataset": name,
                            "protocol": manifest["protocol"],
                            "seed": seed,
                            "method": method,
                            "budget": budget,
                            "selection": selection,
                            "feature_cache": feature_metadata[name],
                            "manifest_sha256": content_hash(manifest),
                            "source_sha256": source_hashes(),
                            "metrics": classification_metrics(labels, probabilities),
                            "seconds": time.perf_counter() - started,
                        }
                        run_name = f"{manifest['protocol']}-seed{seed}-fraction{fraction:g}-{method}"
                        directory = output_dir / run_name
                        write_arrays(
                            directory / "predictions.npz",
                            image_ids=np.asarray(roles["test"]),
                            labels=labels,
                            probabilities=probabilities,
                        )
                        result["prediction_sha256"] = file_hash(directory / "predictions.npz")
                        write_json(directory / "result.json", result)
                        results.append(result)
                        print(
                            json.dumps(
                                {
                                    "run": run_name,
                                    "primary_score": result["metrics"][metric],
                                    "seconds": result["seconds"],
                                }
                            ),
                            flush=True,
                        )
    if len(results) != 60:
        raise ValueError("The declared experiment matrix is incomplete.")
    summary = {"complete": True, "environment": environment(), "source_sha256": source_hashes(), "results": results}
    try:
        summary["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        summary["git_head"] = None
    write_json(output_dir / "matrix.json", summary)
    return summary


def main():
    """Run a resource pilot, complete extraction, or the explicit-protocol probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "extract", "evaluate"))
    parser.add_argument("--raw-dir", type=Path, default=Path(".cache/project1/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/project1/processed"))
    parser.add_argument("--protocol-dir", type=Path, default=Path(".cache/project1/verified-v2"))
    parser.add_argument("--asset-dir", type=Path, default=Path(".cache/project1/assets/dino"))
    parser.add_argument("--feature-dir", type=Path, default=Path(".cache/project1/features"))
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/project1/experiments"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    records, manifests = read_protocols(args.protocol_dir)
    lock = json.loads(ASSET_LOCK.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate":
        sets, metadata = {}, {}
        for name, values in records.items():
            sets[name], metadata[name] = feature_cache(
                args.feature_dir, feature_identity(values, lock, args.threads), values
            )
        with threadpool_limits(limits=args.threads):
            evaluate_matrix(records, manifests, sets, metadata, args.output_dir)
        return
    scene = Scene8(split="train", archive_path=args.raw_dir / "scene8.zip", processed_cache_dir=args.cache_dir)
    isic = ISIC2016(data_dir=args.raw_dir, processed_cache_dir=args.cache_dir)
    samples = {
        "scene8": SourceSamples([scene], records["scene8"]),
        "isic2016": SourceSamples([isic["train"], isic["test"]], records["isic2016"]),
    }
    model, lock = load_dino(args.asset_dir)
    if args.stage == "pilot":
        report = {
            "environment": environment(),
            "threads": args.threads,
            "batch_size": args.batch_size,
            "assets": lock,
            "source_sha256": source_hashes(),
            "datasets": {},
        }
        for name, source in samples.items():
            protocol = "scene8-pixel-groups-v1" if name == "scene8" else "isic2016-train-filtered-pixel-groups-v2"
            fit_ids = manifests[(protocol, 42)]["roles"]["fit"]
            chosen_ids = [fit_ids[i] for i in np.linspace(0, len(fit_ids) - 1, 32, dtype=int)]
            preprocessing_montage(source, chosen_ids[::8], args.output_dir / f"{name}-preprocessing.png")
            by_id = {r.image_id: i for i, r in enumerate(source.records)}
            chosen_records = [source.records[by_id[i]] for i in chosen_ids]
            chosen_samples = IndexedSamples(source, [by_id[i] for i in chosen_ids])
            started = time.perf_counter()
            extract_features(model, chosen_samples, chosen_records, args.batch_size)
            seconds = time.perf_counter() - started
            report["datasets"][name] = {
                "images": 32,
                "seconds": seconds,
                "images_per_second": 32 / seconds,
                "estimated_full_seconds": seconds / 32 * len(source),
                "peak_process_memory_bytes": peak_memory_bytes(),
            }
        write_json(args.output_dir / "pilot.json", report)
        print(json.dumps(report, indent=2), flush=True)
    else:
        for name, source in samples.items():
            _, metadata = feature_cache(
                args.feature_dir,
                feature_identity(records[name], lock, args.threads),
                records[name],
                lambda source=source, name=name: extract_features(
                    model, source, records[name], args.batch_size, progress=True
                ),
            )
            print(json.dumps({"dataset": name, **metadata}), flush=True)


if __name__ == "__main__":
    main()
