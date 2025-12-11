import numpy as np
import scipy.io as sio
import datasets

class SVHN(datasets.GeneratorBasedBuilder):
    """SVHN (Street View House Numbers) Dataset for image classification."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="The SVHN dataset contains images of digits obtained from house numbers in Google Street View "
                        "images. It has over 600,000 labeled digit images.",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[str(i) for i in range(10)]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://ufldl.stanford.edu/housenumbers/",
            citation="""@inproceedings{netzer2011reading,
                          title={Reading digits in natural images with unsupervised feature learning},
                          author={Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Baolin and Ng, Andrew Y and others},
                          booktitle={NIPS workshop on deep learning and unsupervised feature learning},
                          volume={2011},
                          number={2},
                          pages={4},
                          year={2011},
                          organization={Granada}
                        }"""
        )

    def _split_generators(self, dl_manager):
        urls = {
            "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
        }
        downloaded_files = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, file_path):
        data = sio.loadmat(file_path)
        images = data["X"].transpose([3, 0, 1, 2])
        labels = np.squeeze(data["y"])

        # Convert '0' label from 10 to 0
        labels[labels == 10] = 0

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": label}

