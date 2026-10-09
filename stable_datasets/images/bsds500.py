import io
import tarfile
from pathlib import PurePosixPath

import numpy as np
import scipy.io as sio
from PIL import Image as PILImage

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Sequence, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


_URL = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
_ROOT = PurePosixPath("BSR/BSDS500/data")
_SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}  # split name -> folder in archive


def encode_mask(mask: np.ndarray) -> bytes:
    """Encode a 2-D integer map as lossless 16-bit PNG bytes."""
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(mask, dtype=np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def decode_mask(data: bytes) -> np.ndarray:
    """Decode PNG bytes produced by `encode_mask` back to a 2-D uint16 array."""
    return np.array(PILImage.open(io.BytesIO(data)), dtype=np.uint16)


class BSDS500(BaseDatasetBuilder):
    """Unsupervised image segmentation and boundary detection.
    The `Berkeley Segmentation Dataset 500 <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html>`_
    contains 500 natural images (481x321 or 321x481), each with 4-9 human
    segmentations given as region-label maps and boundary maps. The official
    split is 200 train / 100 validation / 200 test images. No class labels.

    Segmentations and boundaries are stored as lists of PNG bytes (one per
    annotator); use `decode_mask` to turn each into a 2-D integer array.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
        assets={split: DownloadInfo(url=_URL) for split in _SPLIT_DIRS},  # one archive holds all splits
        citation="""@article{arbelaez2011contour,
                         title={Contour Detection and Hierarchical Image Segmentation},
                         author={Arbel{\\'a}ez, Pablo and Maire, Michael and Fowlkes, Charless and Malik, Jitendra},
                         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
                         volume={33}, number={5}, pages={898--916}, year={2011}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""BSDS500: 500 natural images with multiple human segmentations (region labels and boundary maps) per image, split 200/100/200 into train/validation/test. See https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html for more information.""",
            features=Features(
                {
                    "image": Image(),
                    "image_id": Value("string"),
                    "segmentations": Sequence(Value("binary")),  # PNG bytes, one per annotator
                    "boundaries": Sequence(Value("binary")),  # PNG bytes, one per annotator
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield one example per image, with all of its human annotations."""
        image_dir = _ROOT / "images" / _SPLIT_DIRS[split]
        truth_dir = _ROOT / "groundTruth" / _SPLIT_DIRS[split]

        # Single pass over the archive: collect raw bytes for this split's files.
        images, truths = {}, {}
        with tarfile.open(data_path, "r:*") as tar:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.parent == image_dir and path.suffix == ".jpg":
                    images[path.stem] = tar.extractfile(member).read()
                elif path.parent == truth_dir and path.suffix == ".mat":
                    truths[path.stem] = tar.extractfile(member).read()

        for idx, image_id in enumerate(sorted(images)):
            # MATLAB cell array of structs, shape (1, n_annotators).
            annotations = sio.loadmat(io.BytesIO(truths[image_id]))["groundTruth"][0]
            yield (
                idx,
                {
                    "image": images[image_id],  # original JPEG bytes, stored as-is
                    "image_id": image_id,
                    "segmentations": [encode_mask(a[0, 0]["Segmentation"]) for a in annotations],
                    "boundaries": [encode_mask(a[0, 0]["Boundaries"]) for a in annotations],
                },
            )
