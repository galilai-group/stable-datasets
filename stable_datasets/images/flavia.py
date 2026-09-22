import io
import re
import tarfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Inclusive filename ranges in the official class-table order (original labels skip 13).
_CLASS_RANGES = (
    (1001, 1059, "pubescent_bamboo"),
    (1060, 1122, "chinese_horse_chestnut"),
    (1552, 1616, "anhui_barberry"),
    (1123, 1194, "chinese_redbud"),
    (1195, 1267, "true_indigo"),
    (1268, 1323, "japanese_maple"),
    (1324, 1385, "nanmu"),
    (1386, 1437, "castor_aralia"),
    (1497, 1551, "chinese_cinnamon"),
    (1438, 1496, "goldenrain_tree"),
    (2001, 2050, "big-fruited_holly"),
    (2051, 2113, "japanese_cheesewood"),
    (2114, 2165, "wintersweet"),
    (2166, 2230, "camphortree"),
    (2231, 2290, "japan_arrowwood"),
    (2291, 2346, "sweet_osmanthus"),
    (2347, 2423, "deodar"),
    (2424, 2485, "ginkgo"),
    (2486, 2546, "crape_myrtle"),
    (2547, 2612, "oleander"),
    (2616, 2675, "yew_plum_pine"),
    (3001, 3055, "japanese_flowering_cherry"),
    (3056, 3110, "glossy_privet"),
    (3111, 3175, "chinese_toon"),
    (3176, 3229, "peach"),
    (3230, 3281, "ford_woodlotus"),
    (3282, 3334, "trident_maple"),
    (3335, 3389, "beale's_barberry"),
    (3390, 3446, "southern_magnolia"),
    (3447, 3510, "canadian_poplar"),
    (3511, 3563, "chinese_tulip_tree"),
    (3566, 3621, "tangerine"),
)


class Flavia(BaseDatasetBuilder):
    """Flavia: 1,907 leaf images from 32 plant species.

    The complete raw collection is exposed as ``train`` because no fixed train/test
    partition is distributed. Labels 0-31 follow the published class-table order,
    not the order of filenames or archive entries. Original image sizes are retained.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://flavia.sourceforge.net/",
        "assets": {
            "train": "https://downloads.sourceforge.net/project/flavia/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2",
        },
        "citation": """@inproceedings{wu2007flavia,
            author    = {Stephen Gang Wu and Forrest Sheng Bao and Eric You Xu and
                         Yu-Xuan Wang and Yi-Fan Chang and Qiao-Liang Xiang},
            title     = {A Leaf Recognition Algorithm for Plant Classification Using Probabilistic Neural Network},
            booktitle = {2007 IEEE International Symposium on Signal Processing and Information Technology},
            year      = {2007},
            doi       = {10.1109/ISSPIT.2007.4458016}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "Flavia contains 1,907 leaf images from 32 plant species. The complete "
                "raw collection is available as train, without an official fixed train/test split."
            ),
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Stream Leaves/NNNN.jpg and assign classes using the published filename ranges."""
        with tarfile.open(data_path, "r|bz2") as archive:
            for entry in archive:
                if not entry.isfile():
                    continue
                match = re.fullmatch(r"([0-9]{4})\.jpg", PurePosixPath(entry.name).name, flags=re.IGNORECASE)
                if match is None:
                    continue
                image_id = int(match.group(1))
                label = next(
                    (label for label, (first, last, _) in enumerate(_CLASS_RANGES) if first <= image_id <= last),
                    None,
                )
                if label is None:
                    raise ValueError(f"Unknown Flavia image ID {image_id} in {entry.name!r}.")
                with archive.extractfile(entry) as image_file:
                    image = PILImage.open(io.BytesIO(image_file.read())).convert("RGB")
                yield entry.name, {"image": image, "label": label}

    @staticmethod
    def _labels():
        """Common names in official class order, mapped to contiguous labels 0-31."""
        return [name for _, _, name in _CLASS_RANGES]
