import io
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile
import datasets


class ArabicCharacters(datasets.GeneratorBasedBuilder):
    """Arabic Handwritten Characters Dataset

    Abstract
    Handwritten Arabic character recognition systems face several challenges, including the unlimited variation in human handwriting and large public databases. In this work, we model a deep learning architecture that can be effectively apply to recognizing Arabic handwritten characters. A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in supervised mode. The CNN trained and tested our database that contain 16800 of handwritten Arabic characters. In this paper, the optimization methods implemented to increase the performance of CNN. Common machine learning methods usually apply a combination of feature extractor and trainable classifier. The use of CNN leads to significant improvements across different machine-learning classification algorithms. Our proposed CNN is giving an average 5.1% misclassification error on testing data.

    Context
    The motivation of this study is to use cross knowledge learned from multiple works to enhancement the performance of Arabic handwritten character recognition. In recent years, Arabic handwritten characters recognition with different handwriting styles as well, making it important to find and work on a new and advanced solution for handwriting recognition. A deep learning systems needs a huge number of data (images) to be able to make a good decisions.

    Content
    The data-set is composed of 16,800 characters written by 60 participants, the age range is between 19 to 40 years, and 90% of participants are right-hand. Each participant wrote each character (from ’alef’ to ’yeh’) ten times on two forms as shown in Fig. 7(a) & 7(b). The forms were scanned at the resolution of 300 dpi. Each block is segmented automatically using Matlab 2016a to determining the coordinates for each block. The database is partitioned into two sets: a training set (13,440 characters to 480 images per class) and a test set (3,360 characters to 120 images per class). Writers of training set and test set are exclusive. Ordering of including writers to test set are randomized to make sure that writers of test set are not from a single institution (to ensure variability of the test set).
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""Arabic Handwritten Characters Dataset, consisting of 16,800 characters 
                           written by 60 participants. The dataset is split into training and test 
                           sets, with a balanced distribution across all classes.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[str(i) for i in range(28)])
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/mloey/Arabic-Handwritten-Characters-Dataset",
            citation="""@article{el2017arabic,
                        title={Arabic handwritten characters recognition using convolutional neural network},
                        author={El-Sawy, Ahmed and Loey, Mohamed and El-Bakry, Hazem},
                        journal={WSEAS Transactions on Computer Research},
                        volume={5},
                        pages={11--19},
                        year={2017}}"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Train%20Images%2013440x32x32.zip",
            "test": "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Test%20Images%203360x32x32.zip"
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
                    label = int(entry.filename.split("_")[-1][:-4]) - 1  # Extract label from filename

                    yield entry.filename, {
                        "image": image,
                        "label": label
                    }
