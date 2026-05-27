"""MovingMNIST: the canonical 10K-sequence test set from Srivastava et al. (2015)."""

from pathlib import Path

import numpy as np

from stable_datasets.schema import Array4D, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


class MovingMNIST(BaseDatasetBuilder):
    """Moving MNIST (test split only).

    10,000 sequences of 20 grayscale frames at 64x64. The canonical artifact has
    no class labels; training data is conventionally generated procedurally from
    MNIST digits, which is out of scope here.
    """

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="http://www.cs.toronto.edu/~nitish/unsupervised_video/",
        assets={
            "test": DownloadInfo(url="http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"),
        },
        citation="""@inproceedings{srivastava2015unsupervised,
            title={Unsupervised Learning of Video Representations using LSTMs},
            author={Srivastava, Nitish and Mansimov, Elman and Salakhutdinov, Ruslan},
            booktitle={ICML},
            year={2015}
        }""",
    )

    def _info(self):
        return DatasetInfo(
            description="MovingMNIST test split: 10,000 sequences of 20 grayscale 64x64 frames.",
            features=Features({"video": Array4D(shape=(20, 64, 64, 1), dtype="uint8")}),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        data_path = download(self.SOURCE["assets"]["test"], dest_folder=self._raw_download_dir)
        return [SplitGenerator(name=Split.TEST, gen_kwargs={"data_path": data_path})]

    def _generate_examples(self, data_path):
        arr = np.load(Path(data_path))
        # Source layout: (T=20, N, H=64, W=64). Reshape to (N, T, H, W, 1).
        arr = np.transpose(arr, (1, 0, 2, 3))[..., None]
        for idx in range(arr.shape[0]):
            yield idx, {"video": arr[idx]}
