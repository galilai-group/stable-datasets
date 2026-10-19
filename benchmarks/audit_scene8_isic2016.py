"""Audit original assets, verify native caches, and persist pre-model protocols.

Run from a source checkout, for example::

    python -m benchmarks.audit_scene8_isic2016 --raw-dir .cache/project1/raw \
        --cache-dir .cache/project1/processed --output-dir .cache/project1/verified

This command performs no model training and does not alter original data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from benchmarks.protocols import fingerprint_samples, isic2016_protocol, scene8_protocol
from stable_datasets.images import ISIC2016, Scene8


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_split(dataset, archive_path):
    """Compare every cached JPEG to its original ZIP member, then decode all."""
    original = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or "__MACOSX" in path.parts or path.suffix.lower() != ".jpg":
                continue
            if path.stem in original:
                raise ValueError(f"Duplicate source ID: {path.stem}")
            original[path.stem] = hashlib.sha256(archive.read(member)).hexdigest()
    seen = set()
    raw_dataset = dataset.with_format("raw")
    for index in range(len(raw_dataset)):
        row = raw_dataset[index]
        image_id = row["image_id"]
        if image_id in seen or hashlib.sha256(row["image"]).hexdigest() != original.get(image_id):
            raise ValueError(f"Cached image differs from original or is repeated: {image_id}")
        seen.add(image_id)
    if seen != original.keys():
        raise ValueError("Cached/source image coverage differs.")
    records = fingerprint_samples(dataset)
    return records, {
        "images": len(records),
        "class_counts": dict(Counter(r.label for r in records)),
        "original_bytes_preserved": True,
        "decoded_all": True,
    }


def run_audit(raw_dir, cache_dir, output_dir, seeds=(42, 43, 44)):
    """Build/reopen caches, audit all rows, and save seeded role manifests."""
    raw_dir, cache_dir, output_dir = Path(raw_dir), Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = raw_dir / "scene8.zip"
    # Validate even pre-existing local raw files; BaseDatasetBuilder otherwise
    # treats a completed raw download as a cache hit without rehashing it.
    expected = [(scene_path, Scene8.SOURCE.assets["train"])] + [
        (raw_dir / f"ISBI2016_ISIC_Part3_{key}", info) for key, info in ISIC2016.SOURCE.assets.items()
    ]
    hashes = {}
    for path, info in expected:
        digest = _file_hash(path)
        if f"sha256:{digest}" != info.checksum:
            raise ValueError(f"Raw asset differs from audited release: {path}")
        hashes[path.name] = digest
    scene = Scene8(split="train", archive_path=scene_path, processed_cache_dir=cache_dir)
    isic = ISIC2016(data_dir=raw_dir, processed_cache_dir=cache_dir)
    scene_records, scene_report = _audit_split(scene, scene_path)
    train_records, train_report = _audit_split(isic["train"], raw_dir / "ISBI2016_ISIC_Part3_Training_Data.zip")
    test_records, test_report = _audit_split(isic["test"], raw_dir / "ISBI2016_ISIC_Part3_Test_Data.zip")
    # Missing source paths can only succeed when the processed caches reopen.
    scene_warm = Scene8(split="train", archive_path=raw_dir / "absent.zip", processed_cache_dir=cache_dir)
    isic_warm = ISIC2016(data_dir=raw_dir / "absent", processed_cache_dir=cache_dir)
    for first, warm in ((scene, scene_warm), (isic["train"], isic_warm["train"]), (isic["test"], isic_warm["test"])):
        first_rows, warm_rows = first.with_format("raw"), warm.with_format("raw")
        if len(first_rows) != len(warm_rows):
            raise ValueError("Warm-cache row count changed.")
        for index in range(len(first_rows)):
            original_row, warm_row = first_rows[index], warm_rows[index]
            if original_row != warm_row:
                raise ValueError("Warm-cache sample differs.")
    for name, records in (
        ("scene8", scene_records),
        ("isic2016_train", train_records),
        ("isic2016_test", test_records),
    ):
        (output_dir / f"{name}.records.json").write_text(
            json.dumps([asdict(r) for r in records], sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    protocols = []
    for seed in seeds:
        for manifest in (
            scene8_protocol(scene_records, seed),
            isic2016_protocol(train_records, test_records, seed),
            isic2016_protocol(train_records, test_records, seed, exclude_test_duplicates=True),
        ):
            filename = f"{manifest.protocol}-seed{seed}.json"
            manifest.save(output_dir / filename)
            protocols.append(
                {
                    "manifest": filename,
                    "source_fingerprint": manifest.source_fingerprint,
                    "counts": {role: len(ids) for role, ids in manifest.roles.items()},
                    "excluded_train_ids": manifest.excluded_train_ids,
                }
            )
    report = {
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("stable-datasets", "pillow", "pyarrow", "scikit-learn")
            },
        },
        "source_sha256": hashes,
        "splits": {"scene8": scene_report, "isic2016_train": train_report, "isic2016_test": test_report},
        "warm_cache_all_rows_equal": True,
        "protocols": protocols,
        "limitations": [
            "Exact RGB pixel audit only; near duplicates and patient identity not established.",
            "No model training or evaluation has run in this audit.",
        ],
    }
    (output_dir / "audit.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    """Parse local asset locations and run the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    print(json.dumps(run_audit(args.raw_dir, args.cache_dir, args.output_dir, args.seeds), indent=2))


if __name__ == "__main__":
    main()
