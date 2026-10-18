"""Subject-disjoint windows from UCI's Daily and Sports Activities dataset."""

import re
import zipfile

import numpy as np

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder


class DailyAndSportsActivities(BaseDatasetBuilder):
    """Load 5-second, 45-channel windows without mixing subjects across splits.

    The source provides no official train/test partition. This builder defines
    a reproducible cross-subject split: subjects 1-6 train, subjects 7-8 test.
    ``series`` is time-major (125, 45). Original 1-based subject and segment
    identifiers are retained; activity labels are zero-based.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities",
        assets={
            split: DownloadInfo(
                url="https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
                filename="daily_and_sports_activities.zip",
            )
            for split in ("train", "test")
        },
        citation=(
            "Barshan, B. & Altun, K. (2010). Daily and Sports Activities [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5C59F."
        ),
        license="CC BY 4.0",
    )
    SAMPLE_RATE = 25
    SEQUENCE_LENGTH = 125
    CLASS_NAMES = (
        "sitting",
        "standing",
        "lying_on_back",
        "lying_on_right_side",
        "ascending_stairs",
        "descending_stairs",
        "standing_in_elevator",
        "moving_in_elevator",
        "walking_in_parking_lot",
        "walking_on_flat_treadmill",
        "walking_on_inclined_treadmill",
        "running_on_treadmill",
        "stepper",
        "cross_trainer",
        "cycling_horizontal",
        "cycling_vertical",
        "rowing",
        "jumping",
        "basketball",
    )
    ACTIVITY_NAMES = CLASS_NAMES
    CHANNEL_NAMES = tuple(
        f"{unit}_{sensor}_{axis}"
        for unit in ("torso", "right_arm", "left_arm", "right_leg", "left_leg")
        for sensor in ("acc", "gyro", "mag")
        for axis in ("x", "y", "z")
    )
    SPLIT_SUBJECTS = {"train": frozenset(range(1, 7)), "test": frozenset((7, 8))}

    def _info(self):
        return DatasetInfo(
            description=(
                "19 activities performed by 8 subjects using five body-worn IMUs at 25 Hz. "
                "Each example is an unnormalized, time-major (125, 45) window. "
                "Builder-defined split: subjects 1-6 train and 7-8 test; no official split exists."
            ),
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=list(self.CLASS_NAMES)),
                    "subject_id": Value("int32"),
                    "segment_id": Value("int32"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        if split not in self.SPLIT_SUBJECTS:
            raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'.")
        subjects = self.SPLIT_SUBJECTS[split]
        pattern = re.compile(r"(?:^|/)a(\d{2})/p(\d+)/s(\d{2})\.txt$")
        with zipfile.ZipFile(data_path) as archive:
            members = []
            seen = set()
            for name in archive.namelist():
                match = pattern.search(name)
                if match is None:
                    continue
                activity, subject, segment = map(int, match.groups())
                if not (1 <= activity <= 19 and 1 <= subject <= 8 and 1 <= segment <= 60):
                    raise ValueError(f"Invalid activity/subject/segment identifier in {name!r}.")
                key = (activity, subject, segment)
                if key in seen:
                    raise ValueError(f"Duplicate activity/subject/segment window: {key}.")
                seen.add(key)
                if subject in subjects:
                    members.append((key, name))
            if not members:
                raise ValueError(f"No Daily and Sports Activities windows found for {split!r}.")
            for (activity, subject, segment), name in sorted(members):
                with archive.open(name) as stream:
                    series = np.loadtxt(stream, delimiter=",", dtype=np.float32, ndmin=2)
                if series.shape != (self.SEQUENCE_LENGTH, len(self.CHANNEL_NAMES)):
                    raise ValueError(f"Expected (125, 45) in {name!r}, got {series.shape}.")
                if not np.isfinite(series).all():
                    raise ValueError(f"Non-finite sensor values in {name!r}.")
                yield (
                    f"a{activity:02d}/p{subject}/s{segment:02d}",
                    {
                        "series": series,
                        "label": activity - 1,
                        "subject_id": subject,
                        "segment_id": segment,
                    },
                )
