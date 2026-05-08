"""PatchCamelyon dataset (stub).

TODO: Implement using ``BaseDatasetBuilder`` and the download helpers
in ``stable_datasets.utils``.
"""

from stable_datasets.schema import DatasetInfo, Version
from stable_datasets.utils import BaseDatasetBuilder


class PatchCamelyon(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "https://github.com/basveeling/pcam",
        "citation": "TBD",
        "assets": {},
    }

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("PatchCamelyon builder not implemented yet.")
