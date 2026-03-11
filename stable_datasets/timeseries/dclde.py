"""DCLDE dataset (stub).

This file was previously a legacy imperative loader at the top-level package. It was
moved under `stable_datasets.timeseries` to match the repository layout.

TODO: Implement using `BaseDatasetBuilder` and the local download helpers in `stable_datasets.utils`.
"""

from stable_datasets.schema import DatasetInfo, Version
from stable_datasets.utils import BaseDatasetBuilder


class DCLDE(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "TBD",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")
