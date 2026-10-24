import io
import zipfile

from PIL import Image as PILImage
from PIL import ImageOps

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Pinned to a commit so the split folders and images can't change under us.
_ARCHIVE = DownloadInfo(
    url="https://github.com/pratikkayal/PlantDoc-Dataset/archive/5467f6012d78d1c446145d5f582da6096f852ae8.zip",
    filename="PlantDoc-Dataset.zip",
)

# EXIF tag that stores how a photo should be rotated for display.
_EXIF_ORIENTATION = 0x0112

# Class folder names of the dataset repository, in sorted order.
_CLASSES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato two spotted spider mites leaf",
    "grape leaf",
    "grape leaf black rot",
]


class PlantDoc(BaseDatasetBuilder):
    """PlantDoc: plant disease classification from leaf images taken in the wild.

    The classification data of the PlantDoc repository: 2,578 images of 13 plant species,
    one label per image, in 28 classes that are either a healthy leaf of a species or one
    of its diseases or pests. Unlike lab datasets such as PlantVillage, the images were
    collected from the internet and mostly show leaves in natural conditions, with cluttered
    backgrounds. The repository README calls this data Cropped-PlantDoc, but the images are
    the whole photos, not the 9,216 bounding-box crops that the paper calls Cropped-PlantDoc.

    The official split has 2,342 train and 236 test images. "Tomato two spotted spider mites
    leaf" only has 2 train images and does not appear in the test split.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/pratikkayal/PlantDoc-Dataset",
        assets={
            "train": _ARCHIVE,
            "test": _ARCHIVE,
        },
        citation="""@inproceedings{singh2020plantdoc,
                         title={PlantDoc: A Dataset for Visual Plant Disease Detection},
                         author={Singh, Davinder and Jain, Naman and Jain, Pranjali and Kayal, Pratik and
                                 Kumawat, Sudhakar and Batra, Nipun},
                         booktitle={Proceedings of the 7th ACM IKDD CoDS and 25th COMAD},
                         pages={249--253},
                         year={2020},
                         doi={10.1145/3371158.3371196}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""PlantDoc contains 2,578 images of plant leaves from 13 species collected from the
                internet, labeled with 28 classes covering healthy leaves and plant diseases.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        label_feature = self.info.features["label"]
        with zipfile.ZipFile(data_path, "r") as archive:
            for name in sorted(archive.namelist()):
                # Images live at "<repo>-<commit>/<split>/<class>/<file>".
                parts = name.split("/")
                if len(parts) != 4 or parts[1] != split or not parts[3]:
                    continue
                _, _, class_name, file_name = parts

                image = archive.read(name)
                with PILImage.open(io.BytesIO(image)) as img:
                    if img.format != "JPEG" or img.mode != "RGB":
                        # A few files are PNG, MPO, CMYK or grayscale even when named .jpg.
                        image = ImageOps.exif_transpose(img).convert("RGB")
                    elif img.getexif().get(_EXIF_ORIENTATION, 1) > 1:
                        # Some photos only store their rotation in EXIF, which decoding ignores,
                        # so save them upright. They are large photos, so keep them as JPEG.
                        buff = io.BytesIO()
                        ImageOps.exif_transpose(img).save(buff, format="JPEG", quality=95)
                        image = buff.getvalue()

                yield (
                    f"{split}/{class_name}/{file_name}",
                    {"image": image, "label": label_feature.str2int(class_name)},
                )
