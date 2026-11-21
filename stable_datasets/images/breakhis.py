import io
import re
import tarfile

from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


# Benign subtypes first, then malignant, so ``label < 4`` means benign.
TUMOR_TYPES = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]
FOLDS = [f"fold{k}" for k in range(1, 6)]

# e.g. SOB_B_A-14-22549AB-40-001.png: biopsy procedure, tumour class and type, patient ID (14-22549AB),
# magnification and image number, as documented on the homepage.
_IMAGE_RE = re.compile(
    r"^BreaKHis_v1/histology_slides/breast/(?:benign|malignant)/SOB/(?P<tumor_type>[a-z_]+)/[^/]+/(?P<magnification>\d+X)/"
    r"(?P<file_name>SOB_[BM]_[A-Z]+-(?P<patient_id>\d+-\d+[A-Z]*)-\d+-\d+\.png)$"
)


def _read_fold(mkfold_path, fold):
    """Map each image file name to "train" or "test" in one of the official folds of mkfold.tar.gz."""
    with tarfile.open(mkfold_path, "r:gz") as tar:
        lines = tar.extractfile(f"ds{fold}.txt").read().decode().splitlines()
    # Each line is "<file name>|<magnification>|<fold>|<train or test>".
    return {name: group.strip() for name, _, _, group in (line.split("|") for line in lines if line.strip())}


class BreakHis(BaseDatasetBuilder):
    """Breast Cancer Histopathological Image Classification (BreakHis).

    BreakHis contains 7,909 microscopic images of breast tumour tissue from surgical biopsy slides of 82 patients,
    stained with hematoxylin and eosin and imaged at four magnifications (40X, 100X, 200X, 400X). The images are
    RGB PNGs of 700x460 pixels, except 74 that are 700x456.
    Each image is labelled with one of eight tumour types, four benign (adenosis, fibroadenoma, phyllodes
    tumour, tubular adenoma) and four malignant (ductal, lobular, mucinous and papillary carcinoma).

    The authors publish the five train/test folds used in the paper, each of which keeps all images of a
    slide (one patient ID and tumour type) in the same split. Select one with ``config_name="fold1"``
    (the default) to ``"fold5"``. Fold 1 has 5,005 train and 2,904 test images.
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = [BuilderConfig(name=fold, description=f"Official fold {fold[-1]}") for fold in FOLDS]
    DEFAULT_CONFIG_NAME = "fold1"

    SOURCE = DatasetSource(
        homepage="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
        assets={
            "images": DownloadInfo(
                url="https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
                checksum="sha256:d118014cf8a58f6d1fc4571e55efb523c932c85cfba2fb05627e455fc40bd640",
            ),
            # The authors' fold lists (dsfold1.txt, ..., dsfold5.txt), linked from the homepage.
            "folds": DownloadInfo(
                url="https://www.inf.ufpr.br/lesoliveira/download/mkfold.tar.gz",
                checksum="sha256:db7e2f16c398fbc42286b4f7d65e5f55c8f94d95123a46dfd052d8117d68ae27",
            ),
        },
        citation="""@article{spanhol2016dataset,
            title={A Dataset for Breast Cancer Histopathological Image Classification},
            author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
            journal={IEEE Transactions on Biomedical Engineering},
            volume={63},
            number={7},
            pages={1455--1462},
            year={2016},
            publisher={IEEE}
        }""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "BreakHis: 7,909 breast tumour histopathology images (700x460 RGB, 74 at 700x456) from 82 patients "
                "at 40X, 100X, 200X and 400X magnification, labelled with 8 tumour types "
                "(4 benign, 4 malignant), with the authors' five train/test folds."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=TUMOR_TYPES),
                    "malignant": ClassLabel(names=["benign", "malignant"]),
                    "magnification": ClassLabel(names=MAGNIFICATIONS),
                    "patient_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source()["assets"]
        images_path, folds_path = bulk_download(
            [assets["images"], assets["folds"]], dest_folder=self._raw_download_dir
        )
        return [
            SplitGenerator(
                name=split, gen_kwargs={"images_path": images_path, "folds_path": folds_path, "split": split}
            )
            for split in ("train", "test")
        ]

    def _generate_examples(self, images_path, folds_path, split):
        """Stream the PNGs of one split of the selected fold out of the archive, in archive order."""
        groups = _read_fold(folds_path, self.config.name)
        idx = 0
        with tarfile.open(images_path, "r:gz") as tar:
            for member in tqdm(tar, desc=f"Processing {split} split"):
                match = _IMAGE_RE.match(member.name) if member.isfile() else None
                if match is None or groups[match["file_name"]] != split:
                    continue

                raw_bytes = tar.extractfile(member).read()
                with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
                    if pil_img.mode != "RGB":
                        buf = io.BytesIO()
                        pil_img.convert("RGB").save(buf, format="PNG")
                        raw_bytes = buf.getvalue()

                label = TUMOR_TYPES.index(match["tumor_type"])
                yield (
                    idx,
                    {
                        "image": raw_bytes,
                        "label": label,
                        "malignant": int(label >= 4),
                        "magnification": MAGNIFICATIONS.index(match["magnification"]),
                        "patient_id": match["patient_id"],
                    },
                )
                idx += 1
