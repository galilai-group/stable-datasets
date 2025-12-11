import numpy as np
import datasets
import tarfile
from io import BytesIO
from PIL import Image


class CIFAR10C(datasets.GeneratorBasedBuilder):
    """CIFAR-10-C dataset with corrupted CIFAR-10 images."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""CIFAR-10-C is a corrupted version of the CIFAR-10 dataset, with 19 different types of 
                           corruptions applied to the images. The dataset consists of 10 classes and 5 levels 
                           of severity per corruption type.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=[
                        "airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"
                    ]),
                    "corruption_name": datasets.Value("string"),
                    "corruption_level": datasets.Value("int32")
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://zenodo.org/records/2535967",
            license="CC BY 4.0",
            citation="""@article{hendrycks2019robustness,
                        title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
                        author={Dan Hendrycks and Thomas Dietterich},
                        journal={Proceedings of the International Conference on Learning Representations},
                        year={2019}}"""
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(
            "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": archive_path, "corruptions": None},
            ),
        ]

    def _generate_examples(self, archive_path, corruptions=None):
        """Generates corrupted CIFAR-10-C examples."""
        with tarfile.open(archive_path, "r") as tar:
            # Load labels
            array_file = BytesIO()
            array_file.write(tar.extractfile("CIFAR-10-C/labels.npy").read())
            array_file.seek(0)
            labels = np.load(array_file)

            # Determine corruptions to load
            if isinstance(corruptions, str):
                corruptions = [corruptions]
            elif corruptions is None:
                corruptions = self._corruptions()
            images, labels_list, corruption_names, corruption_levels = [], [], [], []

            # Load corrupted images for each specified corruption type
            for corruption in corruptions:
                assert corruption in self._corruptions(), f"Unknown corruption type: {corruption}"
                print(f"Loading corruption: {corruption}")

                array_file = BytesIO()
                array_file.write(tar.extractfile(f"CIFAR-10-C/{corruption}.npy").read())
                array_file.seek(0)
                corrupted_images = np.load(array_file)

                for level in range(1, 6):  # Each corruption has 5 levels
                    start_idx = (level - 1) * 10000
                    end_idx = level * 10000
                    images.extend(corrupted_images[start_idx:end_idx])
                    labels_list.extend(labels)  # Extend labels accordingly
                    corruption_names.extend([corruption] * 10000)
                    corruption_levels.extend([level] * 10000)

            # Generate examples
            for idx, (image, label, corruption_name, corruption_level) in enumerate(
                    zip(images, labels_list, corruption_names, corruption_levels)
            ):
                yield idx, {
                    "image": Image.fromarray(image),
                    "label": int(label),
                    "corruption_name": corruption_name,
                    "corruption_level": corruption_level,
                }

    @staticmethod
    def _corruptions():
        """Returns the list of available corruption types for CIFAR-10-C."""
        return [
            "zoom_blur", "speckle_noise", "spatter", "snow", "shot_noise", "saturate",
            "pixelate", "motion_blur", "jpeg_compression", "impulse_noise", "glass_blur",
            "gaussian_noise", "gaussian_blur", "frost", "fog", "elastic_transform",
            "defocus_blur", "contrast", "brightness"
        ]
