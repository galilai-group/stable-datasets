"""BrainMNIST dataset builder stub."""

from stable_datasets.schema import DatasetInfo, DatasetSource, Version
from stable_datasets.utils import BaseDatasetBuilder


class BrainMNIST(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = DatasetSource(
        homepage="http://mindbigdata.com/opendb/index.html",
        citation="TBD",
        assets={},
    )

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("BrainMNIST builder not implemented yet.")
