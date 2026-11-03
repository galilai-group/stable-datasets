import io
import zipfile

import numpy as np

from stable_datasets.schema import (
    BuilderConfig,
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_MNIST_C_URL = "https://zenodo.org/records/3239543/files/mnist_c.zip?download=1"


class MNISTC(BaseDatasetBuilder):
    """MNIST-C: a corrupted MNIST benchmark for out-of-distribution robustness.

    MNIST-C applies 15 corruptions (noise, blur, affine transforms, digital and
    weather-like artifacts) to the MNIST train and test sets at a single severity
    per corruption. Each corruption is exposed as a builder config; the
    ``identity`` config is the uncorrupted MNIST reference shipped in the same
    archive.

    Usage:
        dataset = MNISTC(config_name="shot_noise", split="test")
        dataset = MNISTC(config_name="identity", split="train")
    """

    VERSION = Version("1.0.0")

    # Corruption order follows Mu & Gilmer (2019); ``identity`` is the clean reference.
    BUILDER_CONFIGS = [
        BuilderConfig(name="shot_noise", description="Poisson (shot) noise"),
        BuilderConfig(name="impulse_noise", description="Salt-and-pepper (impulse) noise"),
        BuilderConfig(name="glass_blur", description="Glass blur"),
        BuilderConfig(name="motion_blur", description="Motion blur"),
        BuilderConfig(name="shear", description="Affine shear"),
        BuilderConfig(name="scale", description="Affine scale"),
        BuilderConfig(name="rotate", description="Affine rotation"),
        BuilderConfig(name="brightness", description="Brightness increase"),
        BuilderConfig(name="translate", description="Affine translation"),
        BuilderConfig(name="stripe", description="Vertical stripe inversion"),
        BuilderConfig(name="fog", description="Fog"),
        BuilderConfig(name="spatter", description="Spatter"),
        BuilderConfig(name="dotted_line", description="Overlaid dotted line"),
        BuilderConfig(name="zigzag", description="Overlaid zigzag line"),
        BuilderConfig(name="canny_edges", description="Canny edge map"),
        BuilderConfig(name="identity", description="Uncorrupted MNIST reference"),
    ]
    DEFAULT_CONFIG_NAME = "shot_noise"

    # Single archive holds every corruption and both splits; downloaded once.
    SOURCE = DatasetSource(
        homepage="https://github.com/google-research/mnist-c",
        assets={
            "train": DownloadInfo(url=_MNIST_C_URL),
            "test": DownloadInfo(url=_MNIST_C_URL),
        },
        citation="""@article{mu2019mnist,
                        title={MNIST-C: A Robustness Benchmark for Computer Vision},
                        author={Mu, Norman and Gilmer, Justin},
                        journal={arXiv preprint arXiv:1906.02337},
                        year={2019}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=f"""MNIST-C ({self.config.name}): {self.config.description}. 28x28 grayscale handwritten
                digits (10 classes) from MNIST with a fixed corruption applied, for benchmarking robustness to
                distribution shift. See https://arxiv.org/abs/1906.02337 for more information.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(num_classes=10),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        prefix = f"mnist_c/{self.config.name}/{split}"
        with zipfile.ZipFile(data_path, "r") as archive:
            images = np.load(io.BytesIO(archive.read(f"{prefix}_images.npy")))
            labels = np.load(io.BytesIO(archive.read(f"{prefix}_labels.npy")))

        # Stored as (N, 28, 28, 1); drop the channel axis so images decode as mode "L".
        images = images.reshape(-1, 28, 28)

        for idx, (image, label) in enumerate(zip(images, labels)):
            yield idx, {"image": image, "label": int(label)}
