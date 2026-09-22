"""Reproduce the Flavia teaser using unmodified photographs from the raw archive.

Run from the repository root:
    PYTHONPATH=. MPLBACKEND=Agg python docs/source/datasets/teasers/generate_flavia_teaser.py --archive /path/to/Leaves.tar.bz2

Source: https://flavia.sourceforge.net/ (Wu et al., 2007).
Members: Leaves/3529.jpg, Leaves/3411.jpg, Leaves/2339.jpg,
         Leaves/1389.jpg, Leaves/2423.jpg.
"""

import argparse
import textwrap
from contextlib import closing
from pathlib import Path

import matplotlib.pyplot as plt

from stable_datasets.images.flavia import Flavia


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("flavia_teaser.png"))
    args = parser.parse_args()

    members = [f"Leaves/{image_id}.jpg" for image_id in (3529, 3411, 2339, 1389, 2423)]
    builder = object.__new__(Flavia)
    builder.__init__()
    samples = {}
    with closing(builder._generate_examples(args.archive, split="train")) as examples:
        for key, sample in examples:
            if key in members:
                samples[key] = sample
            if len(samples) == len(members):
                break
    missing = set(members) - set(samples)
    if missing:
        raise ValueError(f"Missing original Flavia teaser images: {sorted(missing)}")

    figure, axes = plt.subplots(1, len(members), figsize=(10, 2.2), layout="constrained")
    for axis, member in zip(axes, members):
        sample = samples[member]
        name = builder.info.features["label"].int2str(sample["label"]).replace("_", " ")
        axis.imshow(sample["image"])
        axis.set_title(textwrap.fill(name.title(), width=20), fontsize=10, fontweight="bold")
        axis.axis("off")
    figure.savefig(
        args.output,
        dpi=180,
        facecolor="white",
        metadata={
            "Description": (
                "Flavia dataset photographs, Wu et al. (2007); https://flavia.sourceforge.net/; "
                f"source archive: {Flavia.SOURCE['assets']['train']}; members: {', '.join(members)}"
            )
        },
    )
    plt.close(figure)
    print(f"Saved {args.output} from original Flavia archive members: {', '.join(members)}")


if __name__ == "__main__":
    main()
