import io
import tarfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# One archive holds all three splits. The DataShare REST endpoint sends Content-Length, which the
# downloader uses to check completeness; the classic bitstream URL is the fallback.
_ARCHIVE = DownloadInfo(
    url="https://datashare.ed.ac.uk/server/api/core/bitstreams/e8e186cc-2688-48f1-aa5a-3fda1a43f8b6/content",
    fallbacks=["https://datashare.ed.ac.uk/bitstreams/e8e186cc-2688-48f1-aa5a-3fda1a43f8b6/download"],
    checksum="md5:6ee4d0c996905fe93221de577967a372",
    filename="CINIC-10.tar.gz",
)


class CINIC10(BaseDatasetBuilder):
    """CINIC-10: CINIC-10 Is Not ImageNet or CIFAR-10.

    CINIC-10 (Darlow et al., 2018) extends CIFAR-10 to 270,000 32x32 RGB images over the same
    ten classes: all 60,000 CIFAR-10 images plus 210,000 down-sampled ImageNet images. Each split
    (``train``, ``validation``, ``test``) has 90,000 class-balanced images, 20,000 of them from
    CIFAR-10. ``origin`` records which source each image comes from.
    """

    VERSION = Version("1.0.0")

    # Split name in the library -> directory name inside the archive.
    _SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}

    SOURCE = DatasetSource(
        homepage="https://datashare.ed.ac.uk/handle/10283/3192",
        assets={"train": _ARCHIVE, "validation": _ARCHIVE, "test": _ARCHIVE},
        citation="""@article{darlow2018cinic,
                        title={CINIC-10 is not ImageNet or CIFAR-10},
                        author={Darlow, Luke N and Crowley, Elliot J and Antoniou, Antreas and Storkey, Amos J},
                        journal={arXiv preprint arXiv:1810.03505},
                        year={2018}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "CINIC-10: 270,000 32x32 RGB images over the ten CIFAR-10 classes, made of all 60,000 CIFAR-10 "
                "images plus 210,000 down-sampled ImageNet images."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                    "origin": ClassLabel(names=["cifar10", "imagenet"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Stream the archive, whose members are ``<train|valid|test>/<class>/<file>.png``.

        CIFAR-10 images are named ``cifar10-<train|test>-<idx>.png``, ImageNet images ``<wnid>_<idx>.png``.
        """
        prefix = self._SPLIT_DIRS[split] + "/"
        label_to_idx = {name: idx for idx, name in enumerate(self._labels())}

        # Streaming mode ("r|gz") reads the archive sequentially without building a member index.
        with tarfile.open(data_path, "r|gz") as tar:
            for member in tar:
                if not member.name.startswith(prefix) or not member.name.endswith(".png"):
                    continue
                _, class_name, file_name = member.name.split("/")
                image = self._to_rgb(tar.extractfile(member).read())
                origin = 0 if file_name.startswith("cifar10-") else 1
                yield member.name, {"image": image, "label": label_to_idx[class_name], "origin": origin}

    @staticmethod
    def _to_rgb(raw: bytes):
        """Keep RGB PNGs as stored; convert the few grayscale ones in the release to RGB."""
        with PILImage.open(io.BytesIO(raw)) as img:
            if img.mode == "RGB":
                return raw
            return img.convert("RGB")

    @staticmethod
    def _labels():
        # Same ten classes, in the same order, as CIFAR-10.
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
