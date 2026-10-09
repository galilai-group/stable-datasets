import io
import zipfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image as PILImage

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Sequence, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


_BASE_URL = "https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB"
_DOWNLOAD_URL = f"{_BASE_URL}/download"
_COLOR_MARGIN = 100  # how much the marking color must exceed the other channels to count as marked


def encode_mask(mask: np.ndarray) -> bytes:
    """Encode a 2-D label map (0 = background, 1/2 = objects) as lossless PNG bytes."""
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(mask, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def decode_mask(data: bytes) -> np.ndarray:
    """Decode PNG bytes produced by `encode_mask` back to a 2-D uint8 label map."""
    return np.array(PILImage.open(io.BytesIO(data)), dtype=np.uint8)


def _marks_to_mask(marked: np.ndarray) -> np.ndarray:
    """Turn a human segmentation (image with objects painted red / blue) into a label map."""
    rgb = marked.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = np.zeros(r.shape, dtype=np.uint8)
    mask[r - np.maximum(g, b) > _COLOR_MARGIN] = 1  # red: first object
    mask[b - np.maximum(r, g) > _COLOR_MARGIN] = 2  # blue: second object
    return mask


class WeizmannSegmentation(BaseDatasetBuilder):
    """Unsupervised figure-ground segmentation (evaluation only).
    The `Weizmann Segmentation Evaluation Database <https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/>`_
    contains 200 images: 100 showing one salient object (split ``one_object``)
    and 100 showing two (split ``two_objects``). Each image has 1-3 human
    segmentations (always 3 for ``one_object``). There is no training split and no class labels.

    Segmentations are stored as a list of PNG bytes (one per annotator); use
    `decode_mask` to get a 2-D map with 0 = background, 1 = first object, 2 = second object.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage=f"{_BASE_URL}/",
        assets={
            "one_object": DownloadInfo(url=f"{_DOWNLOAD_URL}/Weizmann_Seg_DB_1obj.ZIP"),
            "two_objects": DownloadInfo(url=f"{_DOWNLOAD_URL}/Weizmann_Seg_DB_2obj.zip"),
        },
        citation="""@inproceedings{alpert2007image,
                         title={Image Segmentation by Probabilistic Bottom-Up Aggregation and Cue Integration},
                         author={Alpert, Sharon and Galun, Meirav and Basri, Ronen and Brandt, Achi},
                         booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
                         year={2007}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""Weizmann Segmentation Evaluation Database: 200 images (100 with one object, 100 with two objects), each with 1-3 human figure-ground segmentations. Evaluation only; no training split and no class labels. See https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/ for more information.""",
            features=Features(
                {
                    "image": Image(),
                    "image_bw": Image(),
                    "image_id": Value("string"),
                    "segmentations": Sequence(Value("binary")),  # PNG bytes, one per annotator
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Yield one example per image folder, with all of its human segmentations."""
        # Archive layout: <image_id>/{src_color,src_bw,human_seg}/*.png (plus a few unrelated files).
        files = {}  # image_id -> {"src_color": {stem: bytes}, "src_bw": {...}, "human_seg": {...}}
        with zipfile.ZipFile(data_path) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.suffix.lower() != ".png" or "__MACOSX" in path.parts or path.name.startswith("._"):
                    continue
                if len(path.parts) < 3 or path.parts[-2] not in {"src_color", "src_bw", "human_seg"}:
                    continue
                image_id, kind = path.parts[-3], path.parts[-2]
                files.setdefault(image_id, {}).setdefault(kind, {})[path.stem] = archive.read(name)

        for idx, image_id in enumerate(sorted(files)):
            entry = files[image_id]
            # Pick the file named after the folder: one folder also contains an unrelated stray image.
            image_bw = PILImage.open(io.BytesIO(entry["src_bw"][image_id])).convert("L")
            segmentations = [
                encode_mask(_marks_to_mask(np.array(PILImage.open(io.BytesIO(data)).convert("RGB"))))
                for _, data in sorted(entry["human_seg"].items())
            ]
            yield (
                idx,
                {
                    "image": entry["src_color"][image_id],  # original PNG bytes, stored as-is
                    "image_bw": np.array(image_bw),
                    "image_id": image_id,
                    "segmentations": segmentations,
                },
            )
