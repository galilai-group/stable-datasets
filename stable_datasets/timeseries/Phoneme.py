import zipfile

import numpy as np
from scipy.io import arff

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


class Phoneme(BaseDatasetBuilder):
    """Phoneme timeseries classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="http://www.timeseriesclassification.com/description.php?Dataset=Phoneme",
        assets={
            "train": DownloadInfo(url="http://www.timeseriesclassification.com/Downloads/Phoneme.zip"),
            "test": DownloadInfo(url="http://www.timeseriesclassification.com/Downloads/Phoneme.zip"),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Phoneme audio-derived timeseries classification dataset.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(num_classes=39),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            member = _find_member(archive, f"Phoneme_{split.upper()}.arff")
            with archive.open(member) as fh:
                records, meta = arff.loadarff(fh)

        names = meta.names()
        columns = np.asarray([records[name] for name in names], dtype=object)
        series = columns[:-1].T.astype("float32")
        labels = columns[-1]
        label_to_id = _label_to_id(labels, names[-1], meta)

        for idx, (x, y) in enumerate(zip(series, labels)):
            yield idx, {"series": x[:, None], "label": label_to_id[_label_name(y)]}


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


def _label_to_id(labels, label_attr: str, meta) -> dict[str, int]:
    try:
        declared = [_label_name(v) for v in meta[label_attr][1]]
    except Exception:
        declared = sorted({_label_name(v) for v in labels})
    return {name: idx for idx, name in enumerate(declared)}
