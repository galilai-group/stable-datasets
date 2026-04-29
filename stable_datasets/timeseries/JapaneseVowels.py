import tempfile
import zipfile
from pathlib import Path

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
from stable_datasets.utils import BaseDatasetBuilder, load_from_tsfile_to_dataframe


class JapaneseVowels(BaseDatasetBuilder):
    """JapaneseVowels multivariate timeseries classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels",
        assets={
            "train": DownloadInfo(url="http://www.timeseriesclassification.com/Downloads/JapaneseVowels.zip"),
            "test": DownloadInfo(url="http://www.timeseriesclassification.com/Downloads/JapaneseVowels.zip"),
        },
        citation="See dataset homepage.",
    )
    SEQUENCE_LENGTH = 29

    def _info(self):
        return DatasetInfo(
            description="JapaneseVowels multivariate timeseries classification dataset.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=[f"speaker_{idx}" for idx in range(9)]),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            member = _find_member(archive, f"JapaneseVowels_{split.upper()}.ts")
            with tempfile.TemporaryDirectory() as tmp:
                archive.extract(member, tmp)
                ts_path = Path(tmp) / member
                x_df, labels = load_from_tsfile_to_dataframe(ts_path)

        dims = []
        for col in x_df.columns:
            dims.append(np.stack(list(x_df[col].map(lambda x: x.reindex(range(self.SEQUENCE_LENGTH))))))
        series = np.nan_to_num(np.stack(dims, axis=-1).astype("float32"))
        label_to_id = {name: idx for idx, name in enumerate(sorted({_label_name(v) for v in labels}))}

        for idx, (x, y) in enumerate(zip(series, labels)):
            yield idx, {"series": x, "label": label_to_id[_label_name(y)]}


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    suffix = suffix.lower()
    for name in archive.namelist():
        if name.lower().endswith(suffix):
            return name
    raise FileNotFoundError(f"Could not find {suffix!r} in {archive.filename}")


def _label_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    return str(value)
