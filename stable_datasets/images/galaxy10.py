import datasets
import h5py
import numpy as np
from PIL import Image

from stable_datasets.utils import BaseDatasetBuilder


class Galaxy10Decal(BaseDatasetBuilder):
    """Galaxy10 DECaLS Dataset: Galaxy morphology classification with DECaLS images.

    Galaxy10 DECaLS is a much improved version of the original Galaxy10 dataset. It contains
    17,736 256x256 pixel colored galaxy images (g, r and z band) separated into 10 classes.
    The images come from DESI Legacy Imaging Surveys (DECaLS) and labels come from Galaxy Zoo.

    The original Galaxy10 dataset was created with Galaxy Zoo (GZ) Data Release 2 where volunteers
    classify ~270k of SDSS galaxy images. GZ later utilized images from DESI Legacy Imaging Surveys
    (DECaLS) with much better resolution and image quality. Galaxy10 DECaLS has combined all three
    (GZ DR2 with DECaLS images instead of SDSS images and DECaLS campaign a/b, c) resulting in
    ~441k of unique galaxies covered by DECaLS where ~18k of those images were selected in 10 broad
    classes using volunteer votes with more rigorous filtering.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://astronn.readthedocs.io/en/latest/galaxy10.html",
        "assets": {
            "train": "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5",
        },
        "citation": """@article{walmsley2020galaxy,
                        title={Galaxy Zoo: probabilistic morphology through Bayesian CNNs and active learning},
                        author={Walmsley, Mike and Smith, Lewis and Lintott, Chris and Gal, Yarin and Bamford, Steven and Dickinson, Hugh and Fortson, Lucy and Kruk, Sandor and Masters, Karen and Scarlata, Claudia and others},
                        journal={Monthly Notices of the Royal Astronomical Society},
                        volume={491},
                        number={2},
                        pages={1554--1574},
                        year={2020},
                        publisher={Oxford University Press}
                    }""",
        "license": "MIT",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "Galaxy10 DECaLS dataset: 17,736 256x256 colored galaxy images (g, r and z band) "
                "separated into 10 classes. Images come from DESI Legacy Imaging Surveys (DECaLS) "
                "and labels come from Galaxy Zoo volunteer classifications. This dataset is commonly "
                "used for galaxy morphology classification tasks."
            ),
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                    "ra": datasets.Value("float64"),
                    "dec": datasets.Value("float64"),
                    "redshift": datasets.Value("float64"),
                    "pxscale": datasets.Value("float64"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the HDF5 file.

        The h5 file contains:
        - images: (17736, 256, 256, 3) uint8 array
        - ans: (17736,) int array with class labels (0-9)
        - ra: (17736,) float array with right ascension
        - dec: (17736,) float array with declination
        - redshift: (17736,) float array
        - pxscale: (17736,) float array (arcsecond per pixel)
        """
        with h5py.File(data_path, "r") as f:
            images = np.array(f["images"])
            labels = np.array(f["ans"])
            ra = np.array(f["ra"])
            dec = np.array(f["dec"])
            redshift = np.array(f["redshift"])
            pxscale = np.array(f["pxscale"])

        for idx in range(len(images)):
            img_pil = Image.fromarray(images[idx], mode="RGB")

            yield (
                idx,
                {
                    "image": img_pil,
                    "label": int(labels[idx]),
                    "ra": float(ra[idx]),
                    "dec": float(dec[idx]),
                    "redshift": float(redshift[idx]),
                    "pxscale": float(pxscale[idx]),
                },
            )

    @staticmethod
    def _labels():
        """Returns the list of Galaxy10 class names."""
        return [
            "Disturbed Galaxies",
            "Merging Galaxies",
            "Round Smooth Galaxies",
            "In-between Round Smooth Galaxies",
            "Cigar Shaped Smooth Galaxies",
            "Barred Spiral Galaxies",
            "Unbarred Tight Spiral Galaxies",
            "Unbarred Loose Spiral Galaxies",
            "Edge-on Galaxies without Bulge",
            "Edge-on Galaxies with Bulge",
        ]
