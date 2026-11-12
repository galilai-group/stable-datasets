import contextlib
import gzip
import io
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_ZENODO = "https://zenodo.org/records/2546921/files/camelyonpatch_level_2_split_"


class PatchCamelyon(BaseDatasetBuilder):
    """PatchCamelyon (PCam): metastasis detection in lymph node histopathology patches.

    PCam (Veeling et al., 2018) contains 327,680 96x96 RGB patches extracted from the Camelyon16
    whole-slide images. A patch is labeled ``positive`` if its center 32x32 region contains
    metastatic tumor tissue. The official splits come from disjoint sets of slides:

    - ``train``: 262,144 patches
    - ``validation``: 32,768 patches (``valid`` in the release)
    - ``test``: 32,768 patches

    Rows also keep the per-patch metadata of the release: source slide, coordinates and
    whether the patch contains tumor outside the center region.
    """

    VERSION = Version("1.0.0")

    # Split name in the library -> split tag in the release file names.
    _SPLIT_TAGS = {Split.TRAIN: "train", Split.VALIDATION: "valid", Split.TEST: "test"}

    # Patches read from the HDF5 file at once (4096 * 96 * 96 * 3 bytes = 113 MB).
    _READ_ROWS = 4096

    # MD5s from the Zenodo record. The PCam README lists the validation and test MD5s swapped.
    SOURCE = DatasetSource(
        homepage="https://github.com/basveeling/pcam",
        assets={
            "train_x": DownloadInfo(
                url=_ZENODO + "train_x.h5.gz",
                checksum="md5:1571f514728f59376b705fc836ff4b63",
            ),
            "train_y": DownloadInfo(
                url=_ZENODO + "train_y.h5.gz",
                checksum="md5:35c2d7259d906cfc8143347bb8e05be7",
            ),
            "train_meta": DownloadInfo(
                url=_ZENODO + "train_meta.csv",
                checksum="md5:5a3dd671e465cfd74b5b822125e65b0a",
            ),
            "validation_x": DownloadInfo(
                url=_ZENODO + "valid_x.h5.gz",
                checksum="md5:d5b63470df7cfa627aeec8b9dc0c066e",
            ),
            "validation_y": DownloadInfo(
                url=_ZENODO + "valid_y.h5.gz",
                checksum="md5:2b85f58b927af9964a4c15b8f7e8f179",
            ),
            "validation_meta": DownloadInfo(
                url=_ZENODO + "valid_meta.csv",
                checksum="md5:67589e00a4a37ec317f2d1932c7502ca",
            ),
            "test_x": DownloadInfo(
                url=_ZENODO + "test_x.h5.gz",
                checksum="md5:d8c2d60d490dbd479f8199bdfa0cf6ec",
            ),
            "test_y": DownloadInfo(
                url=_ZENODO + "test_y.h5.gz",
                checksum="md5:60a7035772fbdb7f34eb86d4420cf66a",
            ),
            "test_meta": DownloadInfo(
                url=_ZENODO + "test_meta.csv",
                checksum="md5:3455fd69135b66734e1008f3af684566",
            ),
        },
        citation="""@article{veeling2018rotation,
                        title={Rotation Equivariant {CNNs} for Digital Pathology},
                        author={Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and Cohen, Taco and Welling, Max},
                        journal={arXiv preprint arXiv:1806.03962},
                        year={2018}}
                    @article{bejnordi2017diagnostic,
                        title={Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer},
                        author={Ehteshami Bejnordi, Babak and others},
                        journal={JAMA},
                        volume={318},
                        number={22},
                        pages={2199--2210},
                        year={2017},
                        doi={10.1001/jama.2017.14585}}""",
        license="CC0 1.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "PatchCamelyon (PCam): 327,680 96x96 RGB patches from Camelyon16 lymph node whole-slide images, "
                "labeled positive if the center 32x32 region contains metastatic tumor tissue."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["negative", "positive"]),
                    "wsi": Value("string"),
                    "coord_x": Value("int32"),
                    "coord_y": Value("int32"),
                    "tumor_patch": Value("bool"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        # Asset keys are per file, not per split.
        return list(self._SPLIT_TAGS)

    def _split_generators(self):
        assets = self._source()["assets"]
        keys = [f"{split}_{part}" for split in self._SPLIT_TAGS for part in ("x", "y", "meta")]
        paths = dict(zip(keys, bulk_download([assets[key] for key in keys], dest_folder=self._raw_download_dir)))
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={
                    "x_path": paths[f"{split}_x"],
                    "y_path": paths[f"{split}_y"],
                    "meta_path": paths[f"{split}_meta"],
                    "split": split,
                },
            )
            for split in self._SPLIT_TAGS
        ]

    def _generate_examples(self, x_path, y_path, meta_path, split):
        with gzip.open(y_path, "rb") as f, h5py.File(io.BytesIO(f.read()), "r") as h5:
            labels = h5["y"][:].reshape(-1)
        meta = pd.read_csv(meta_path, index_col=0)
        # Guards against mixing up the image, label and metadata files of different splits.
        if not np.array_equal(meta["center_tumor_patch"].to_numpy(), labels.astype(bool)):
            raise ValueError(f"PCam {split}: labels do not match 'center_tumor_patch' in the metadata.")
        wsi = meta["wsi"].tolist()
        coord_x = meta["coord_x"].tolist()
        coord_y = meta["coord_y"].tolist()
        tumor_patch = meta["tumor_patch"].tolist()

        with self._decompressed(x_path) as h5_path, h5py.File(h5_path, "r") as f:
            images = f["x"]
            for start in range(0, len(labels), self._READ_ROWS):
                for idx, image in enumerate(images[start : start + self._READ_ROWS], start):
                    yield (
                        idx,
                        {
                            "image": image,
                            "label": int(labels[idx]),
                            "wsi": wsi[idx],
                            "coord_x": coord_x[idx],
                            "coord_y": coord_y[idx],
                            "tumor_patch": tumor_patch[idx],
                        },
                    )

    @staticmethod
    @contextlib.contextmanager
    def _decompressed(gz_path):
        """Decompress ``gz_path`` next to itself (HDF5 needs random access) and remove it on exit."""
        gz_path = Path(gz_path)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{gz_path.name}.", suffix=".tmp", dir=gz_path.parent)
        try:
            with os.fdopen(fd, "wb") as out, gzip.open(gz_path, "rb") as src:
                shutil.copyfileobj(src, out)
            yield tmp_name
        finally:
            Path(tmp_name).unlink(missing_ok=True)
