"""High-Gamma dataset (stub).

Moved under `stable_datasets.timeseries` as it is an EEG/time-series dataset.

Reference:
- https://github.com/robintibor/high-gamma-dataset

TODO: Implement as a HuggingFace-compatible builder using `BaseDatasetBuilder`
and the local download helpers in `stable_datasets.utils`.
"""

import datasets

from stable_datasets.utils import BaseDatasetBuilder


class HighGamma(BaseDatasetBuilder):
    VERSION = datasets.Version("0.0.0")
    SOURCE = {
        "homepage": "https://github.com/robintibor/high-gamma-dataset",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> datasets.DatasetInfo:  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _split_generators(self, dl_manager):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")
