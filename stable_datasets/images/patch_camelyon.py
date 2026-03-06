"""PatchCamelyon dataset (stub).

This file was previously a broken legacy loader at the top-level package. It was moved
under `stable_datasets.images` to match the repository layout.

TODO: Implement as a HuggingFace-compatible builder using `BaseDatasetBuilder`
and the local download helpers in `stable_datasets.utils`.
"""


from stable_datasets.utils import BaseDatasetBuilder
from stable_datasets.schema import DatasetInfo, Version


class PatchCamelyon(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "https://github.com/basveeling/pcam",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> datasets.DatasetInfo:  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")

    def _split_generators(self, dl_manager):  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")
