"""BrainMNIST dataset (stub).

Moved under `stable_datasets.timeseries` per project convention.

Reference:
- http://mindbigdata.com/opendb/index.html

TODO: Implement as a HuggingFace-compatible builder using `StableDatasetBuilder`
and the local download helpers in `stable_datasets.utils`.
"""

import datasets

from stable_datasets.utils import StableDatasetBuilder


class BrainMNIST(StableDatasetBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _split_generators(self, dl_manager):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")


