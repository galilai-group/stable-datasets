import numpy as np
import datasets
import scipy.io as sio
import os


class EMNISTConfig(datasets.BuilderConfig):
    def __init__(self, variant, **kwargs):
        super(EMNISTConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.variant = variant


class EMNIST(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        EMNISTConfig(name="byclass", variant="byclass"),
        EMNISTConfig(name="bymerge", variant="bymerge"),
        EMNISTConfig(name="balanced", variant="balanced"),
        EMNISTConfig(name="letters", variant="letters"),
        EMNISTConfig(name="digits", variant="digits"),
        EMNISTConfig(name="mnist", variant="mnist"),
    ]

    def _info(self):
        variant = self.config.variant
        if variant == "byclass":
            num_classes = 62
        elif variant == "bymerge":
            num_classes = 47
        elif variant == "balanced":
            num_classes = 47
        elif variant == "letters":
            num_classes = 26
        elif variant == "digits":
            num_classes = 10
        elif variant == "mnist":
            num_classes = 10

        return datasets.DatasetInfo(
            description="EMNIST dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(num_classes=num_classes)
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://www.nist.gov/itl/iad/image-group/emnist-dataset",
            citation="""@misc{cohen2017emnistextensionmnisthandwritten,
                        title={EMNIST: an extension of MNIST to handwritten letters}, 
                        author={Gregory Cohen and Saeed Afshar and Jonathan Tapson and André van Schaik},
                        year={2017},
                        eprint={1702.05373},
                        archivePrefix={arXiv},
                        primaryClass={cs.CV},
                        url={https://arxiv.org/abs/1702.05373}, 
            }"""
        )

    def _split_generators(self, dl_manager):
        variant = self.config.variant
        # Download and extract the matlab.zip file
        extracted_path = dl_manager.download_and_extract("https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip")

        # The extracted_path now points to the directory containing "matlab" folder
        # The .mat files are likely in extracted_path/matlab/
        mat_dir = os.path.join(extracted_path, "matlab")
        mat_file = f"emnist-{variant}.mat"
        mat_path = os.path.join(mat_dir, mat_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"mat_path": mat_path, "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"mat_path": mat_path, "split": "test"}
            ),
        ]

    def _generate_examples(self, mat_path, split):
        data = sio.loadmat(mat_path)
        dataset = data['dataset'][0, 0]
        subset = dataset[split][0, 0]

        images = subset['images']
        labels = subset['labels']

        images = np.array(images, dtype=np.uint8).reshape(-1, 28, 28)
        labels = np.array(labels, dtype=np.int64).flatten()

        for idx, (img, lbl) in enumerate(zip(images, labels)):
            yield idx, {
                "image": img,
                "label": int(lbl)
            }
