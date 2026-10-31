"""Original force/torque windows from UCI Robot Execution Failures."""

import io
import zipfile

import numpy as np

from stable_datasets.schema import (
    BuilderConfig,
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


class RobotExecutionFailures(BaseDatasetBuilder):
    """Keep the five learning problems separate and preserve all source records.

    Each configuration exposes an unsplit ``train`` collection: the source has no
    official evaluation partition or recording/session identifiers. Evaluation
    splitting belongs to the experiment, not this loader. Packaged executables
    are never extracted or run.
    """

    VERSION = Version("1.0.1")
    BUILDER_CONFIGS = [BuilderConfig(name=f"lp{i}", version=Version("1.0.1")) for i in range(1, 6)]
    DEFAULT_CONFIG_NAME = "lp1"
    SEQUENCE_LENGTH = 15
    CHANNEL_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    CLASS_NAMES = {
        "lp1": ("normal", "collision", "obstruction", "fr_collision"),
        "lp2": ("normal", "back_col", "front_col", "right_col", "left_col"),
        "lp3": ("ok", "moved", "slightly_moved", "lost"),
        "lp4": ("normal", "collision", "obstruction"),
        "lp5": ("normal", "collision_in_tool", "collision_in_part", "bottom_obstruction", "bottom_collision"),
    }
    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/138/robot+execution+failures",
        assets={
            "train": DownloadInfo(
                url="https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip",
                filename="robot_execution_failures.zip",
                checksum="sha256:cdfbe0b260f4a21c5249fe0da4b1baf4041959663a9182253885ee1f41a4568e",
            )
        },
        citation="""@misc{robot_execution_failures_138,
  author = {Lopes, Luis and Camarinha-Matos, Luis},
  title = {Robot Execution Failures},
  year = {1998},
  publisher = {UCI Machine Learning Repository},
  doi = {10.24432/C5M89N},
  url = {https://doi.org/10.24432/C5M89N}
}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                f"Robot Execution Failures {self.config.name}: original 15-by-6 force/torque windows; "
                "no normalization, resampling or official evaluation split. sample_id is source record order."
            ),
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=list(self.CLASS_NAMES[self.config.name])),
                    "sample_id": Value("int32"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        if split != "train":
            raise ValueError("Robot Execution Failures exposes only the unsplit 'train' collection")
        filename = f"{self.config.name}.data"
        with zipfile.ZipFile(data_path) as archive:
            members = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == filename]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one {filename!r}, found {len(members)}")
            text = archive.read(members[0]).decode("ascii")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or len(lines) % 16:
            raise ValueError(f"Expected label plus 15 measurement rows per record in {filename}")
        names = self.CLASS_NAMES[self.config.name]
        for index, offset in enumerate(range(0, len(lines), 16)):
            label = lines[offset]
            if label not in names:
                raise ValueError(f"Unknown {self.config.name} label {label!r} in record {index}")
            series = np.loadtxt(io.StringIO("\n".join(lines[offset + 1 : offset + 16])), dtype=np.float64, ndmin=2)
            if (
                series.shape != (15, 6)
                or not np.isfinite(series).all()
                or not np.equal(series, np.floor(series)).all()
            ):
                raise ValueError(f"Expected finite integer measurements with shape (15, 6) in record {index}")
            converted = series.astype(np.float32)
            if not np.array_equal(converted.astype(np.float64), series):
                raise ValueError(f"Measurement is not losslessly representable as float32 in record {index}")
            yield index, {"series": converted, "label": names.index(label), "sample_id": index}
