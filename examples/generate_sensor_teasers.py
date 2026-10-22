"""Plot actual cached samples for the two sensor dataset documentation pages.

Install matplotlib, then run from the repository root:
python examples/generate_sensor_teasers.py --output-dir docs/source/datasets/teasers
Only a subset of channels is displayed; source signals are not normalized.
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_datasets.timeseries import DailyAndSportsActivities, RobotExecutionFailures


COLORS = ("#0072B2", "#D55E00", "#009E73")


def activities(output, cache, downloads):
    """Show five activities from subject 1, segment 1, using torso acceleration."""
    dataset = DailyAndSportsActivities(split="train", processed_cache_dir=cache, download_dir=downloads)
    labels = (0, 1, 4, 8, 11)
    selected = {}
    for row in dataset:
        if row["subject_id"] == 1 and row["segment_id"] == 1 and row["label"] in labels:
            selected[row["label"]] = row
        if len(selected) == len(labels):
            break
    if set(selected) != set(labels):
        raise ValueError("Missing the documented activity samples")
    fig, axes = plt.subplots(1, len(labels), figsize=(16, 4), layout="constrained")
    for ax, label in zip(axes, labels):
        x = np.asarray(selected[label]["series"], dtype=np.float32)
        for channel, color in enumerate(COLORS):
            ax.plot(np.arange(len(x)) / DailyAndSportsActivities.SAMPLE_RATE, x[:, channel], color=color, lw=1.2)
        ax.set_title(DailyAndSportsActivities.ACTIVITY_NAMES[label].replace("_", " "), fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Torso acceleration (source units)")
    fig.suptitle("Daily and Sports Activities | subject 1, segment 1 | 125 samples at 25 Hz", fontsize=14)
    fig.legend(axes[0].lines, ("x", "y", "z"), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "daily_and_sports_activities_teaser.png", dpi=140, facecolor="white")
    plt.close(fig)


def robot(output, cache, downloads):
    """Show the first non-normal record in each independent learning problem."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), layout="constrained")
    for column, config in enumerate(RobotExecutionFailures.CLASS_NAMES):
        dataset = RobotExecutionFailures(
            config_name=config, split="all", processed_cache_dir=cache, download_dir=downloads
        )
        row = next(sample for sample in dataset if sample["label"] != 0)
        x = np.asarray(row["series"], dtype=np.float32)
        label = RobotExecutionFailures.CLASS_NAMES[config][row["label"]]
        axes[0, column].set_title(f"{config.upper()}: {label}\nsample_id={row['sample_id']}", fontsize=11)
        for group in range(2):
            ax = axes[group, column]
            for channel, color in enumerate(COLORS):
                ax.plot(np.arange(1, 16), x[:, group * 3 + channel], color=color, lw=1.4)
            ax.set_xticks([1, 5, 10, 15])
            ax.grid(alpha=0.2)
            if group == 1:
                ax.set_xlabel("Sample index")
    axes[0, 0].set_ylabel("Force (source units)")
    axes[1, 0].set_ylabel("Torque (source units)")
    fig.suptitle("Robot Execution Failures | original 15-by-6 windows | one record per task", fontsize=14)
    fig.legend(axes[0, 0].lines, ("x", "y", "z"), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "robot_execution_failures_teaser.png", dpi=140, facecolor="white")
    plt.close(fig)


def main():
    """Write deterministic figures using the public dataset interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-cache-dir", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    activities(args.output_dir, args.processed_cache_dir, args.download_dir)
    robot(args.output_dir, args.processed_cache_dir, args.download_dir)


if __name__ == "__main__":
    main()
