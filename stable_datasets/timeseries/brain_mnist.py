"""BrainMNIST dataset (stub).

Moved under `stable_datasets.timeseries` per project convention.

Reference:
- http://mindbigdata.com/opendb/index.html

TODO: Implement using `BaseDatasetBuilder` and the local download helpers in `stable_datasets.utils`.
"""

from stable_datasets.schema import DatasetInfo, Version
from stable_datasets.utils import BaseDatasetBuilder


class BrainMNIST(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "http://mindbigdata.com/opendb/index.html",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")
