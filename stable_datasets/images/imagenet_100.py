from stable_datasets.schema import DatasetSource, DownloadInfo, Version

from ._imagenet_wnids import IN1K_CLASSES
from .imagenet_1k import ImageNet1K


IN100_CLASSES: list[str] = IN1K_CLASSES[:100]


class ImageNet100(ImageNet1K):
    """ImageNet-100: the first 100 wnids of canonical sorted-alphabetical ImageNet-1K.

    Labels are integers in [0, 100), where label i corresponds to ``IN100_CLASSES[i]``.
    """

    VERSION = Version("3.0.0")
    SOURCE = DatasetSource(
        homepage="https://www.image-net.org/challenges/LSVRC/2012/",
        assets={
            "train": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"),
            "val": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"),
            "devkit": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"),
        },
        citation=ImageNet1K.SOURCE["citation"],
    )

    _ALLOWED_WNIDS: set[str] | None = set(IN100_CLASSES)
