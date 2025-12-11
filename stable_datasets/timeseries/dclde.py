"""DCLDE dataset (stub).

This file was previously a legacy imperative loader at the top-level package. It was
moved under `stable_datasets.timeseries` to match the repository layout.

TODO: Implement as a HuggingFace-compatible builder using `StableDatasetBuilder`
and the local download helpers in `stable_datasets.utils`.
"""

import datasets

from stable_datasets.utils import StableDatasetBuilder


class DCLDE(StableDatasetBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _split_generators(self, dl_manager):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")


