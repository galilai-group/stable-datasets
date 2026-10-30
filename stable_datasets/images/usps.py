import bz2

import numpy as np

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.utils import BaseDatasetBuilder


class USPS(BaseDatasetBuilder):
    """USPS handwritten digits from the LIBSVM distribution."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps",
        citation="""@article{hull1994database,
            author={Jonathan J. Hull},
            title={A Database for Handwritten Text Recognition Research},
            journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
            volume={16},
            number={5},
            pages={550--554},
            year={1994},
            doi={10.1109/34.291440}
        }""",
        assets={
            "train": DownloadInfo(url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"),
            "test": DownloadInfo(url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"),
        },
        license="Unknown",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "USPS contains 7,291 training and 2,007 test images of handwritten digits "
                "in 10 classes. Images are 16x16 grayscale uint8 arrays. "
                "LIBSVM labels 1 through 10 are mapped to class indices 0 through 9, "
                "following torchvision.datasets.USPS."
            ),
            features=Features({"image": Image(), "label": ClassLabel(num_classes=10)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with bz2.open(data_path, "rt", encoding="ascii") as stream:
            for index, line in enumerate(stream):
                fields = line.split()
                try:
                    if not fields:
                        raise ValueError("empty sample")
                    label = int(fields[0])
                    if not 1 <= label <= 10:
                        raise ValueError("label must be between 1 and 10")

                    # LIBSVM omits zero-valued features in sparse representations.
                    pixels = np.zeros(256, dtype=np.float32)
                    seen = set()
                    for field in fields[1:]:
                        feature, value = field.split(":")
                        feature = int(feature)
                        value = float(value)
                        if not 1 <= feature <= 256 or feature in seen:
                            raise ValueError("invalid or duplicate pixel index")
                        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                            raise ValueError("pixel value must be finite and in [-1, 1]")
                        pixels[feature - 1] = value
                        seen.add(feature)
                except (ValueError, OverflowError) as error:
                    raise ValueError(f"Invalid USPS sample at line {index + 1}: {error}") from error

                # Preserve the torchvision USPS conversion, including uint8 truncation.
                image = ((pixels.reshape(16, 16) + 1) / 2 * 255).astype(np.uint8)
                yield index, {"image": image, "label": label - 1}
