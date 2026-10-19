"""Reproducible image-level protocols for Scene8 and ISIC2016.

These protocols are independent of model imports and preserve original builders.
Exact decoded-pixel groups stay together; this is not a near-duplicate or patient
independence guarantee. Persist the manifest before fitting any model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class ImageRecord:
    """Identity, label and exact RGB-pixel fingerprint for one source image."""

    image_id: str
    label: int
    pixel_sha256: str


def fingerprint_samples(samples) -> list[ImageRecord]:
    """Decode every sample once and fingerprint dimensions plus RGB pixels."""
    records = []
    # StableDataset.__iter__ decodes entire Arrow record batches. ISIC images
    # are high resolution; indexed iteration keeps only one decoded image live.
    iterator = (
        (samples[i] for i in range(len(samples)))
        if (hasattr(samples, "__len__") and hasattr(samples, "__getitem__"))
        else iter(samples)
    )
    for sample in iterator:
        image = sample["image"].convert("RGB")
        digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
        records.append(ImageRecord(str(sample["image_id"]), int(sample["label"]), digest))
    return records


@dataclass
class ProtocolManifest:
    """Explicit sample roles; ``trainval`` is the refit pool, not a fourth split."""

    protocol: str
    seed: int
    source_fingerprint: str
    roles: dict[str, list[str]]
    class_counts: dict[str, dict[int, int]]
    excluded_train_ids: list[str]
    duplicate_groups: list[list[str]]
    conflicting_label_groups: list[list[str]]

    def save(self, path):
        """Save the full assignment and its protocol/source identity as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _validate_records(records):
    records = sorted(records, key=lambda r: r.image_id)
    if not records:
        raise ValueError("Cannot partition an empty image collection.")
    if any(not r.image_id or not r.pixel_sha256 or r.label < 0 for r in records):
        raise ValueError("Each record needs a nonempty ID, pixel fingerprint and nonnegative label.")
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("Image IDs must be unique within a source collection.")
    return records


def _folds(records, seed):
    groups = defaultdict(set)
    for record in records:
        groups[record.label].add(record.pixel_sha256)
    if len(groups) < 2 or min(map(len, groups.values())) < 5:
        raise ValueError("Five-fold grouped stratification requires at least five distinct pixel groups per class.")
    labels = [r.label for r in records]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assignment = np.full(len(records), -1, dtype=int)
    for fold, (_, heldout) in enumerate(
        splitter.split(np.zeros(len(records)), labels, [r.pixel_sha256 for r in records])
    ):
        assignment[heldout] = fold
    for fold in range(5):
        if {r.label for r, f in zip(records, assignment) if f == fold} != set(labels):
            raise ValueError("Grouped stratification left a class absent; use a different declared protocol.")
    return assignment


def _manifest(protocol, seed, source, roles, excluded=()):
    # Include the original split membership, not just the image collection.
    payload = {name: [asdict(r) for r in records] for name, records in sorted(source.items())}
    source_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    records = [r for values in source.values() for r in values]
    pixels = defaultdict(list)
    for record in records:
        pixels[record.pixel_sha256].append(record)
    duplicates = sorted(sorted(r.image_id for r in group) for group in pixels.values() if len(group) > 1)
    conflicts = sorted(
        sorted(r.image_id for r in group) for group in pixels.values() if len({r.label for r in group}) > 1
    )
    return ProtocolManifest(
        protocol=protocol,
        seed=seed,
        source_fingerprint=source_hash,
        roles={name: sorted(r.image_id for r in values) for name, values in roles.items()},
        class_counts={name: dict(sorted(Counter(r.label for r in values).items())) for name, values in roles.items()},
        excluded_train_ids=sorted(excluded),
        duplicate_groups=duplicates,
        conflicting_label_groups=conflicts,
    )


def scene8_protocol(records, seed=42) -> ProtocolManifest:
    """Project protocol: approximately 60/20/20, grouped by identical RGB pixels.

    Fold 0 is test, fold 1 validation; the remaining folds fit the classifier.
    Labels and conflicting duplicate records are retained and disclosed.
    """
    records = _validate_records(records)
    folds = _folds(records, seed)
    roles = {
        "fit": [r for r, f in zip(records, folds) if f >= 2],
        "validation": [r for r, f in zip(records, folds) if f == 1],
        "trainval": [r for r, f in zip(records, folds) if f != 0],
        "test": [r for r, f in zip(records, folds) if f == 0],
    }
    return _manifest("scene8-pixel-groups-v1", seed, {"collection": records}, roles)


def isic2016_protocol(train, test, seed=42, *, exclude_test_duplicates=False) -> ProtocolManifest:
    """Preserve official test and share validation across the paired protocols.

    V2 forms validation from training images absent from test by exact pixels.
    Both variants use this same assignment. The official variant adds the
    cross-split duplicates to fit; the filtered variant excludes them entirely.
    Thus removing three images cannot reshuffle hundreds of retained samples.
    The primary analysis uses the filtered variant; the official variant is a
    full-budget sensitivity analysis with the primary hyperparameters frozen.
    Filtering uses test pixels, not diagnoses or scores, and must be disclosed.
    This does not assert patient independence.
    """
    train, test = _validate_records(train), _validate_records(test)
    if {r.image_id for r in train} & {r.image_id for r in test}:
        raise ValueError("Official train/test source IDs overlap.")
    if {r.label for r in train + test} != {0, 1}:
        raise ValueError("ISIC2016 expects labels 0 (benign) and 1 (malignant).")
    test_pixels = {r.pixel_sha256 for r in test}
    overlapping = [r for r in train if r.pixel_sha256 in test_pixels]
    pool = [r for r in train if r.pixel_sha256 not in test_pixels]
    folds = _folds(pool, seed)
    restored = [] if exclude_test_duplicates else overlapping
    roles = {
        "fit": [r for r, f in zip(pool, folds) if f != 0] + restored,
        "validation": [r for r, f in zip(pool, folds) if f == 0],
        "trainval": pool + restored,
        "test": test,
    }
    suffix = "train-filtered" if exclude_test_duplicates else "official"
    excluded = [r.image_id for r in overlapping] if exclude_test_duplicates else []
    return _manifest(f"isic2016-{suffix}-pixel-groups-v2", seed, {"train": train, "test": test}, roles, excluded)
