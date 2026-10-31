import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC"


class BelgiumTSC(BaseDatasetBuilder):
    """BelgiumTSC: the Belgium Traffic Sign Classification benchmark.

    Cropped traffic signs from the BelgiumTS database in 62 classes, following the GTSRB layout. Each image
    keeps a border of 10% of the sign size around the sign, and images vary in size. The official
    Training/Testing partitions are kept as released.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://btsd.ethz.ch/shareddata/",
        assets={
            "train": DownloadInfo(
                url=f"{_URL}/BelgiumTSC_Training.zip",
                checksum="sha256:e5f6a33c18716917d69989488b9f6af573821a7126ca1390af7e75403af1ee61",
            ),
            "test": DownloadInfo(
                url=f"{_URL}/BelgiumTSC_Testing.zip",
                checksum="sha256:1811447452f02ede83c0237f2852fa8bd278f39e40ddcb1ba01c2ed212e735fa",
            ),
        },
        citation="""@inproceedings{timofte2011sparse,
                title     = {Sparse representation based projections},
                author    = {Timofte, Radu and Van Gool, Luc},
                booktitle = {British Machine Vision Conference},
                year      = {2011} }
                @article{timofte2014multi,
                title     = {Multi-view traffic sign detection, recognition, and 3D localisation},
                author    = {Timofte, Radu and Zimmermann, Karel and Van Gool, Luc},
                journal   = {Machine Vision and Applications},
                volume    = {25},
                number    = {3},
                pages     = {633--647},
                year      = {2014} }
        """,
    )

    _SPLIT_DIRS = {"train": "Training", "test": "Testing"}

    def _info(self):
        return DatasetInfo(
            description="""BelgiumTSC contains cropped RGB images of Belgian traffic signs in 62 classes, taken
                from the BelgiumTS street-view database. Images vary in size and include a border around the sign.""",
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield the PPMs under ``<Training|Testing>/<class id>/`` in the zip archive, re-encoded as PNG.

        The label is the class folder. The per-class ``GT-*.csv`` files list a few images that are not in
        the archive, so the images themselves are enumerated instead.
        """
        split_dir = self._SPLIT_DIRS[split]
        with zipfile.ZipFile(data_path) as archive:
            for member in sorted(archive.infolist(), key=lambda m: m.filename):
                parts = member.filename.split("/")
                if len(parts) != 3 or parts[0] != split_dir or not parts[2].endswith(".ppm"):
                    continue
                with archive.open(member) as f, PILImage.open(f) as image:
                    image = image.convert("RGB")
                yield member.filename, {"image": image, "label": int(parts[1])}

    @staticmethod
    def _labels():
        return [f"{i:05d}" for i in range(62)]
