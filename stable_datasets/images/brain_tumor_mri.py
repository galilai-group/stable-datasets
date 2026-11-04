import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=2"
)

# The archive stores each split in its own top level folder.
SPLIT_FOLDERS = {"train": "Training", "test": "Testing"}


class BrainTumorMRI(BaseDatasetBuilder):
    """Brain Tumor MRI Dataset

    This dataset was put together by Masoud Nickparvar by combining three older
    datasets (Figshare, SARTAJ and Br35H). It has 7,200 brain MRI images, and each image
    belongs to one of four classes: glioma, meningioma, pituitary tumor or no tumor.

    The training set has 1,400 images per class and the testing set has 400 per class.
    The images come in many different sizes and are a mix of grayscale and color, and
    they are kept exactly as published.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        assets={
            "train": DownloadInfo(url=KAGGLE_URL, filename="brain_tumor_mri.zip"),
            "test": DownloadInfo(url=KAGGLE_URL, filename="brain_tumor_mri.zip"),
        },
        citation="""@misc{nickparvar2021braintumor,
                      title={Brain Tumor MRI Dataset},
                      author={Nickparvar, Masoud},
                      year={2021},
                      publisher={Kaggle},
                      howpublished={\\url{https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="Brain MRI images labeled as glioma, meningioma, pituitary tumor or no tumor.",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read the images of one split from the zip. The folder name is the label."""
        split_folder = SPLIT_FOLDERS[split]

        with zipfile.ZipFile(data_path) as archive:
            for file_name in sorted(archive.namelist()):
                if not file_name.lower().endswith(".jpg"):
                    continue

                folder, label_name, _ = file_name.split("/")
                if folder != split_folder:
                    continue

                yield file_name, {"image": archive.read(file_name), "label": label_name}

    @staticmethod
    def _labels():
        return ["glioma", "meningioma", "notumor", "pituitary"]
