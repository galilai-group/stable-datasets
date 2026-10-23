"""Create documentation teasers from audited source images, without model inference.

Run after the data audit with::

    python -m benchmarks.transfer.generate_scene8_isic2016_teasers
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/project1/matplotlib").resolve()))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.protocols import fingerprint_samples
from stable_datasets.images import ISIC2016, Scene8


def make_teaser(dataset, records, names, per_class, columns, output):
    """Show the first audited IDs per class, preserving image aspect ratios."""
    selected = []
    for label in range(len(names)):
        indices = [i for i, record in enumerate(records) if record["label"] == label][:per_class]
        if len(indices) != per_class:
            raise ValueError(f"Not enough audited samples for class {names[label]}.")
        for index in indices:
            sample = dataset[index]
            if asdict(fingerprint_samples([sample])[0]) != records[index]:
                raise ValueError("Teaser sample differs from the audited source.")
            selected.append(sample)
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.8, rows * 3.25), squeeze=False)
    for axis, sample in zip(axes.flat, selected):
        axis.imshow(sample["image"])
        axis.set_title(f"{names[sample['label']]}\n{sample['image_id']}", fontsize=10)
        axis.axis("off")
    for axis in list(axes.flat)[len(selected) :]:
        axis.axis("off")
    fig.tight_layout(pad=1.3, h_pad=3.0, w_pad=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=90, facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(output), "image_ids": [s["image_id"] for s in selected]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(".cache/project1/raw"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/project1/processed"))
    parser.add_argument("--audit-dir", type=Path, default=Path(".cache/project1/verified-v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/source/datasets/teasers"))
    args = parser.parse_args()
    scene = Scene8(split="train", archive_path=args.raw_dir / "scene8.zip", processed_cache_dir=args.cache_dir)
    isic = ISIC2016(split="train", data_dir=args.raw_dir, processed_cache_dir=args.cache_dir)
    for dataset, records_name, names, per_class, columns, filename in (
        (scene, "scene8", Scene8.CLASS_NAMES, 1, 4, "scene8_teaser.png"),
        (isic, "isic2016_train", ("benign", "malignant"), 3, 3, "isic2016_teaser.png"),
    ):
        records = json.loads((args.audit_dir / f"{records_name}.records.json").read_text(encoding="utf-8"))
        make_teaser(dataset, records, names, per_class, columns, args.output_dir / filename)


if __name__ == "__main__":
    main()
