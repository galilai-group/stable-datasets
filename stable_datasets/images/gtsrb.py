"""German Traffic Sign Recognition Benchmark (GTSRB)."""

import io

import pyarrow.parquet as pq
from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import BaseDatasetBuilder


_DATA_URL = "https://huggingface.co/datasets/ilee0022/GTSRB/resolve/5acb9804b1afaa577104ab9b38e87200b24b4454/data"


class GTSRB(BaseDatasetBuilder):
    """GTSRB classification with the official final train and labeled test splits."""

    VERSION = Version("1.1.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/houbensebastian/GermanTrafficSignBenchmarks",
        assets={
            "train": DownloadInfo(
                url=f"{_DATA_URL}/train-00000-of-00001.parquet",
                checksum="sha256:9d126ccbb1f7069d0bf32b7bc00d588921e4a096096f28bbf5778ab648130901",
            ),
            "test": DownloadInfo(
                url=f"{_DATA_URL}/test-00000-of-00001.parquet",
                checksum="sha256:2377157b1a6641797e7ef171c047765c001ecfa2e94f1085c8d937f4fbfc3fb0",
            ),
        },
        citation=r"""@inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    pages = {1453--1460},
    year = {2011}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description="Official GTSRB final training and test images with 43 traffic sign classes.",
            features=Features({"image": Image(), "label": ClassLabel(num_classes=43)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _generate_examples(self, data_path, split):
        rows = pq.read_table(data_path, columns=["image", "label"]).to_pylist()
        # Keep the same sample order as the official ZIPs.
        rows.sort(key=lambda row: row["image"]["path"])
        for row in rows:
            with PILImage.open(io.BytesIO(row["image"]["bytes"])) as source_image:
                image = source_image.convert("RGB")
            yield row["image"]["path"], {"image": image, "label": int(row["label"])}
