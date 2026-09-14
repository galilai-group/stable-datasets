import io
import tarfile
from pathlib import Path

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download

from ._imagenet_wnids import IN1K_CLASSES


class _ImageNetArchiveMixin:
    def _iter_inner_images(self, class_tar_bytes: bytes, class_name: str, label: int):
        with tarfile.open(fileobj=io.BytesIO(class_tar_bytes), mode="r:*") as inner:
            for image_member in inner:
                if not image_member.isfile():
                    continue
                if not image_member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_file = inner.extractfile(image_member)
                if image_file is None:
                    continue

                image_bytes = image_file.read()
                yield f"{class_name}/{image_member.name}", {"image": image_bytes, "label": label}

    def _iter_train_examples(self, archive_path: Path, label_map: dict[str, int]):
        outer_mode = "r|*" if self.streaming else "r:*"
        with tarfile.open(archive_path, outer_mode) as outer:
            for member in outer:
                if not member.isfile() or not member.name.endswith(".tar"):
                    continue
                wnid = Path(member.name).stem
                label = label_map.get(wnid)
                if label is None:
                    continue

                class_file = outer.extractfile(member)
                if class_file is None:
                    continue
                yield from self._iter_inner_images(class_file.read(), wnid, label)

    def _iter_val_examples(self, val_tar_path: Path, devkit_tar_gz_path: Path, label_map: dict[str, int]):
        ilsvrc_id_to_wnid = _load_devkit_id_to_wnid(devkit_tar_gz_path)
        gt_labels = _load_devkit_val_ground_truth(devkit_tar_gz_path)

        outer_mode = "r|*" if self.streaming else "r:*"
        with tarfile.open(val_tar_path, outer_mode) as outer:
            for member in outer:
                if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Filenames are ILSVRC2012_val_NNNNNNNN.JPEG; the 1-indexed integer is
                # the line in the ground-truth list.
                stem = Path(member.name).stem
                try:
                    val_idx = int(stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                if val_idx < 1 or val_idx > len(gt_labels):
                    continue
                ilsvrc_id = gt_labels[val_idx - 1]
                wnid = ilsvrc_id_to_wnid.get(ilsvrc_id)
                if wnid is None:
                    continue
                label = label_map.get(wnid)
                if label is None:
                    continue

                image_file = outer.extractfile(member)
                if image_file is None:
                    continue
                yield f"val/{member.name}", {"image": image_file.read(), "label": label}


def _load_devkit_id_to_wnid(devkit_tar_gz_path: Path) -> dict[int, str]:
    """Return ILSVRC2012_ID → wnid mapping parsed from the devkit's meta.mat."""
    from scipy.io import loadmat

    with tarfile.open(devkit_tar_gz_path, "r:*") as tf:
        member = _find_devkit_member(tf, "data/meta.mat")
        fh = tf.extractfile(member)
        assert fh is not None
        mat = loadmat(io.BytesIO(fh.read()), squeeze_me=True, struct_as_record=False)

    synsets = mat["synsets"]
    # synsets may be a 0-d ndarray when squeezed; normalize to iterable.
    if not hasattr(synsets, "__iter__"):
        synsets = [synsets]
    return {int(s.ILSVRC2012_ID): str(s.WNID) for s in synsets}


def _load_devkit_val_ground_truth(devkit_tar_gz_path: Path) -> list[int]:
    with tarfile.open(devkit_tar_gz_path, "r:*") as tf:
        member = _find_devkit_member(tf, "data/ILSVRC2012_validation_ground_truth.txt")
        fh = tf.extractfile(member)
        assert fh is not None
        return [int(line) for line in fh.read().decode().splitlines() if line.strip()]


def _find_devkit_member(tf: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    for member in tf.getmembers():
        if member.isfile() and member.name.endswith(suffix):
            return member
    raise FileNotFoundError(f"Devkit archive does not contain a file ending in {suffix!r}.")


class ImageNet1K(_ImageNetArchiveMixin, BaseDatasetBuilder):
    VERSION = Version("3.0.0")
    SOURCE = DatasetSource(
        homepage="https://www.image-net.org/challenges/LSVRC/2012/",
        assets={
            "train": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"),
            "val": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"),
            "devkit": DownloadInfo(url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"),
        },
        citation="""@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and others},
        journal={CVPR},
        year={2009}
    }""",
    )

    _ALLOWED_WNIDS: set[str] | None = None  # None = use all 1000 IN1K classes.

    def __init__(self, streaming: bool = True, **kwargs):
        self.streaming = streaming
        super().__init__(**kwargs)

    @classmethod
    def _class_names(cls) -> list[str]:
        if cls._ALLOWED_WNIDS is None:
            return IN1K_CLASSES
        return [w for w in IN1K_CLASSES if w in cls._ALLOWED_WNIDS]

    @classmethod
    def _label_map(cls) -> dict[str, int]:
        return {wnid: idx for idx, wnid in enumerate(cls._class_names())}

    def _info(self):
        return DatasetInfo(
            description="ImageNet-1K (ILSVRC2012) train and validation splits.",
            features=Features({"image": Image(encode_format="JPEG"), "label": ClassLabel(names=self._class_names())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        train_path = download(self.SOURCE["assets"]["train"], dest_folder=self._raw_download_dir)
        val_path = download(self.SOURCE["assets"]["val"], dest_folder=self._raw_download_dir)
        devkit_path = download(self.SOURCE["assets"]["devkit"], dest_folder=self._raw_download_dir)
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"split": "train", "data_path": train_path},
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"split": "val", "val_tar": val_path, "devkit_tar_gz": devkit_path},
            ),
        ]

    def _generate_examples(self, split, data_path=None, val_tar=None, devkit_tar_gz=None):
        label_map = self._label_map()
        if split == "train":
            yield from self._iter_train_examples(Path(data_path), label_map=label_map)
        else:
            yield from self._iter_val_examples(Path(val_tar), Path(devkit_tar_gz), label_map=label_map)
