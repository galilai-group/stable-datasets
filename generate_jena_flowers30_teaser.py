"""Render original Jena Flowers 30 samples without downloading the full archive.

Usage: python generate_jena_flowers30_teaser.py [--archive /path/to/JF30-1.zip]
Remote generation additionally requires the optional ``remotezip`` package.
"""

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import matplotlib
from PIL import Image

from stable_datasets.images.jena_flowers30 import JenaFlowers30


matplotlib.use("Agg")


SAMPLES = ["JFT_00001.jpg", "JFT_00018.jpg", "JFT_00078.jpg", "JFT_00119.jpg", "JFT_00185.jpg"]


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local official JF30-1.zip instead of HTTP Range reads.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "docs/source/datasets/teasers/jena_flowers30_teaser.png",
    )
    args = parser.parse_args()
    if args.archive:
        archive = zipfile.ZipFile(args.archive)
    else:
        from remotezip import RemoteZip

        archive = RemoteZip(
            JenaFlowers30.SOURCE["assets"]["train"],
            headers={"User-Agent": "stable-datasets"},
            timeout=(15, 60),
        )

    samples = []
    with archive:
        labels = dict(csv.reader(io.StringIO(archive.read("classlabels.txt").decode("utf-8")), skipinitialspace=True))
        for filename in SAMPLES:
            data = archive.read(f"jpg/{filename}")
            with Image.open(io.BytesIO(data)) as original:
                image = original.convert("RGB")
            class_id = int(labels[filename])
            samples.append((class_id, image))
            print(
                f"{filename}: source_class_id={class_id}, label={class_id - 1}, "
                f"size={image.size}, sha256={hashlib.sha256(data).hexdigest()}",
                flush=True,
            )

    figure = plt.figure(figsize=(7.5, 1.65))
    grid = figure.add_gridspec(
        2,
        len(samples),
        height_ratios=[0.1, 1],
        hspace=0,
        wspace=0.05,
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.01,
    )
    for index, (class_id, image) in enumerate(samples):
        label_axis = figure.add_subplot(grid[0, index])
        label_axis.set_axis_off()
        label_axis.text(0.5, 0.5, f"Class {class_id}", ha="center", va="center", fontsize=9, fontweight="bold")
        axis = figure.add_subplot(grid[1, index])
        axis.imshow(image)
        axis.set_axis_off()
    figure.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
