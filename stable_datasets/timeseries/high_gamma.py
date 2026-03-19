"""High-Gamma dataset (stub).

Moved under `stable_datasets.timeseries` as it is an EEG/time-series dataset.

Reference:
- https://github.com/robintibor/high-gamma-dataset

TODO: Implement using `BaseDatasetBuilder` and the local download helpers in `stable_datasets.utils`.
"""

from stable_datasets.schema import DatasetInfo, Version
from stable_datasets.utils import BaseDatasetBuilder


class HighGamma(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "https://github.com/robintibor/high-gamma-dataset",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")
