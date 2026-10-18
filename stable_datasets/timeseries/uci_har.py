"""Official UCI HAR inertial-signal windows and subject-disjoint splits."""

import io
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


class UCIHAR(BaseDatasetBuilder):
    """UCI HAR windows in (time, channel) order; labels are zero-based."""

    VERSION = Version("1.0.0")
    SAMPLE_RATE = 50
    CHANNEL_NAMES = tuple(f"{signal}_{axis}" for signal in ("body_acc", "body_gyro", "total_acc") for axis in "xyz")
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        assets={
            split: DownloadInfo(
                url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
                filename="UCI_HAR.zip",
                checksum="sha256:c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
            )
            for split in ("train", "test")
        },
        citation=(
            "Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). "
            "A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN 2013. "
            "Dataset DOI: https://doi.org/10.24432/C54S4K."
        ),
        license="Unresolved: UCI page lists CC BY 4.0; bundled README restricts commercial use.",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "UCI HAR: 128-sample windows with nine inertial channels at 50 Hz. "
                "Preserves official train/test splits and original subject IDs; "
                "maps activity labels 1-6 to 0-5. Signals retain the source preprocessing and units."
            ),
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(
                        names=["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
                    ),
                    "subject_id": Value("int32"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        # The UCI download wraps the original dataset ZIP in another ZIP.
        with zipfile.ZipFile(data_path) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as archive:
                prefix = f"UCI HAR Dataset/{split}/"
                labels = np.array([int(v) for v in archive.read(f"{prefix}y_{split}.txt").split()])
                subjects = np.array([int(v) for v in archive.read(f"{prefix}subject_{split}.txt").split()])
                if labels.size == 0 or np.any((labels < 1) | (labels > 6)):
                    raise ValueError("UCI HAR activity labels must be integers in 1-6.")
                if subjects.shape != labels.shape or np.any((subjects < 1) | (subjects > 30)):
                    raise ValueError("UCI HAR subject IDs must match the window count and be in 1-30.")
                channels = []
                for name in self.CHANNEL_NAMES:
                    with archive.open(f"{prefix}Inertial Signals/{name}_{split}.txt") as stream:
                        values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
                    if values.shape != (labels.size, 128):
                        raise ValueError(f"UCI HAR {name} has shape {values.shape}; expected ({labels.size}, 128).")
                    if not np.isfinite(values).all():
                        raise ValueError(f"UCI HAR {name} must contain finite signal values.")
                    channels.append(values)
                series = np.stack(channels, axis=-1)

        for index in range(labels.size):
            yield (
                index,
                {
                    "series": series[index],
                    "label": int(labels[index]) - 1,
                    "subject_id": int(subjects[index]),
                },
            )
