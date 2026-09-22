"""Render the IP102 teaser from original images and their official training labels.

Run from the repository root:
    PYTHONPATH=. python docs/source/datasets/teasers/generate_ip102_teaser.py \
        --archive /path/to/Images.tar --train-list /path/to/train.txt
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import matplotlib
from PIL import Image

from stable_datasets.images.ip102 import IP102


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SAMPLES = ["IP022001503.jpg", "IP057000345.jpg", "IP014000553.jpg", "IP019000054.jpg", "IP070001231.jpg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ip102_teaser.png"))
    args = parser.parse_args()
    labels = IP102._parse_labels(args.train_list)
    images = {}
    provenance = []
    with tarfile.open(args.archive, "r|*") as archive:
        for member in archive:
            filename = Path(member.name).name
            if not member.isfile() or filename not in SAMPLES:
                continue
            with archive.extractfile(member) as source:
                content = source.read()
            with Image.open(io.BytesIO(content)) as original:
                images[filename] = original.convert("RGB")
            record = {"member": member.name, "label": labels[filename], "sha256": hashlib.sha256(content).hexdigest()}
            provenance.append(record)
            print(json.dumps({**record, "size": images[filename].size}), flush=True)
            if len(images) == len(SAMPLES):
                break
    if len(images) != len(SAMPLES):
        raise ValueError(f"Missing teaser samples: {set(SAMPLES) - images.keys()}")

    figure, axes = plt.subplots(2, 5, figsize=(7.5, 1.9), gridspec_kw={"height_ratios": [0.2, 1]})
    for column, filename in enumerate(SAMPLES):
        title = IP102._labels()[labels[filename]]
        axes[0, column].text(0.5, 0.5, title, ha="center", va="center", fontsize=8, fontweight="bold")
        axes[0, column].axis("off")
        axes[1, column].imshow(images[filename])
        axes[1, column].set_anchor("N")
        axes[1, column].axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, hspace=0.02, wspace=0.05)
    figure.savefig(
        args.output,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={
            "Source": IP102.SOURCE["release"],
            "Images": json.dumps(provenance),
            "TrainManifestSHA256": hashlib.sha256(args.train_list.read_bytes()).hexdigest(),
        },
    )
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
