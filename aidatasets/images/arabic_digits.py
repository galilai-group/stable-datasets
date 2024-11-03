# import numpy as np
import io
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile
# from ..utils import Dataset
import datasets


# class ArabicDigits(Dataset):
#     """Arabic Handwritten Characters Dataset
#
#     Astract
#     Handwritten Arabic character recognition systems face several challenges, including the unlimited variation in human handwriting and large public databases. In this work, we model a deep learning architecture that can be effectively apply to recognizing Arabic handwritten characters. A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in supervised mode. The CNN trained and tested our database that contain 16800 of handwritten Arabic characters. In this paper, the optimization methods implemented to increase the performance of CNN. Common machine learning methods usually apply a combination of feature extractor and trainable classifier. The use of CNN leads to significant improvements across different machine-learning classification algorithms. Our proposed CNN is giving an average 5.1% misclassification error on testing data.
#
#     Context
#     The motivation of this study is to use cross knowledge learned from multiple works to enhancement the performance of Arabic handwritten character recognition. In recent years, Arabic handwritten characters recognition with different handwriting styles as well, making it important to find and work on a new and advanced solution for handwriting recognition. A deep learning systems needs a huge number of data (images) to be able to make a good decisions.
#
#     Content
#     The data-set is composed of 16,800 characters written by 60 participants, the age range is between 19 to 40 years, and 90% of participants are right-hand. Each participant wrote each character (from ’alef’ to ’yeh’) ten times on two forms as shown in Fig. 7(a) & 7(b). The forms were scanned at the resolution of 300 dpi. Each block is segmented automatically using Matlab 2016a to determining the coordinates for each block. The database is partitioned into two sets: a training set (13,440 characters to 480 images per class) and a test set (3,360 characters to 120 images per class). Writers of training set and test set are exclusive. Ordering of including writers to test set are randomized to make sure that writers of test set are not from a single institution (to ensure variability of the test set).
#
#
#     Parameters
#     ----------
#
#         path: str (optional)
#             default ($DATASET_PATH), the path to look for the data and
#             where the data will be downloaded if not present
#
#     Returns
#     -------
#
#         train_images: array
#
#         train_labels: array
#
#         valid_images: array
#
#         valid_labels: array
#
#         test_images: array
#
#         test_labels: array
#
#     """
#
#     cite = """
#     @inproceedings{el2016cnn,
#     title={CNN for handwritten arabic digits recognition based on LeNet-5},
#     author={El-Sawy, Ahmed and Hazem, EL-Bakry and Loey, Mohamed},
#     booktitle={International conference on advanced intelligent systems and informatics},
#     pages={566--575},
#     year={2016},
#     organization={Springer}
#     }"""
#
#     _name = "arabic_digits"
#     _source = "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset"
#
#     @property
#     def md5(self):
#         return {
#             "TestImages.zip": "258d12dff10a8fdaf17be73e257e0f25",
#             "TrainImages.zip": "3be5796e39c1497c3a61bc4be77514b1",
#         }
#
#     @property
#     def urls(self):
#         return {
#             "TestImages.zip": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Test%20Images.zip",
#             "TrainImages.zip": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Train%20Images.zip",
#         }
#
#     @property
#     def image_shape(self):
#         return (28, 28, 1)
#
#     @property
#     def modalities(self):
#         return dict(train_X="image", test_X="image", train_y=int, test_y=int)
#
#     def load(self):
#
#         train_images = []
#         test_images = []
#         train_labels = []
#         test_labels = []
#         with ZipFile(self.path / self.name / "TestImages.zip") as archive:
#             for entry in tqdm(archive.infolist()):
#                 if ".png" not in entry.filename:
#                     continue
#                 content = archive.read(entry)
#                 test_images.append(Image.open(io.BytesIO(content)))
#                 test_labels.append(int(entry.filename.split("_")[-1][:-4]))
#
#         with ZipFile(self.path / self.name / "TrainImages.zip") as archive:
#             for entry in tqdm(archive.infolist(), ascii=True):
#                 if ".png" not in entry.filename:
#                     continue
#                 content = archive.read(entry)
#                 train_images.append(Image.open(io.BytesIO(content)))
#                 train_labels.append(int(entry.filename.split("_")[-1][:-4]))
#         self["train_X"] = np.array(train_images)
#         self["train_y"] = np.array(train_labels)
#         self["test_X"] = np.array(test_images)
#         self["test_y"] = np.array(test_labels)
#         return self


class ArabicDigits(datasets.GeneratorBasedBuilder):
    """Arabic Handwritten Digits Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""Arabic Handwritten Digits Dataset, composed of images of Arabic digits handwritten 
                           by participants. This dataset is structured for use in machine learning tasks such 
                           as digit classification.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[str(i) for i in range(10)])
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/mloey/Arabic-Handwritten-Digits-Dataset",
            citation="""@inproceedings{el2016cnn,
                        title={CNN for handwritten arabic digits recognition based on LeNet-5},
                        author={El-Sawy, Ahmed and Hazem, EL-Bakry and Loey, Mohamed},
                        booktitle={International conference on advanced intelligent systems and informatics},
                        pages={566--575},
                        year={2016},
                        organization={Springer}
                        }"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Train%20Images.zip",
            "test": "https://github.com/mloey/Arabic-Handwritten-Digits-Dataset/raw/master/Test%20Images.zip"
        }
        downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": downloaded_files["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": downloaded_files["test"], "split": "test"},
            )
        ]

    def _generate_examples(self, archive_path, split):
        """Generate examples from the ZIP archives of images and labels."""
        with ZipFile(archive_path, "r") as archive:
            for entry in tqdm(archive.infolist(), desc=f"Processing {split} set"):
                if entry.filename.endswith(".png"):
                    content = archive.read(entry)
                    image = Image.open(io.BytesIO(content))
                    label = int(entry.filename.split("_")[-1][:-4])  # Extract label from filename

                    yield entry.filename, {
                        "image": image,
                        "label": label
                    }
