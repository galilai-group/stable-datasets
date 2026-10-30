"""High-Gamma dataset builder stub."""

from stable_datasets.schema import DatasetInfo, DatasetSource, Version
from stable_datasets.utils import BaseDatasetBuilder


class HighGamma(BaseDatasetBuilder):
    """High-Gamma motor-movement EEG dataset (Schirrmeister et al., 2017). Not implemented yet."""

    VERSION = Version("0.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/robintibor/high-gamma-dataset",
        citation="TBD",
        assets={},
    )

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("HighGamma builder not implemented yet.")
