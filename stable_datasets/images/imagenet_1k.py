import io
import tarfile
from pathlib import Path

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


def _default_class_names(count: int) -> list[str]:
    width = max(2, len(str(count - 1)))
    return [f"class_{idx:0{width}d}" for idx in range(count)]


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

                image = PILImage.open(io.BytesIO(image_file.read())).convert("RGB")
                yield f"{class_name}/{image_member.name}", {"image": image, "label": label}

    def _iter_train_examples(self, archive_path: Path, class_limit: int | None):
        if self.streaming:
            with tarfile.open(archive_path, "r|*") as outer:
                class_count = 0
                for member in outer:
                    if not member.isfile() or not member.name.endswith(".tar"):
                        continue
                    if class_limit is not None and class_count >= class_limit:
                        break

                    class_file = outer.extractfile(member)
                    if class_file is None:
                        continue

                    yield from self._iter_inner_images(class_file.read(), Path(member.name).stem, class_count)
                    class_count += 1
            return

        with tarfile.open(archive_path, "r:*") as outer:
            class_count = 0
            for member in outer:
                if not member.isfile() or not member.name.endswith(".tar"):
                    continue
                if class_limit is not None and class_count >= class_limit:
                    break

                class_file = outer.extractfile(member)
                if class_file is None:
                    continue
                yield from self._iter_inner_images(class_file.read(), Path(member.name).stem, class_count)
                class_count += 1


class ImageNet1K(_ImageNetArchiveMixin, BaseDatasetBuilder):
    VERSION = Version("2.0.0")
    SOURCE = {
        "homepage": "https://www.image-net.org/challenges/LSVRC/2012/",
        "assets": {"train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"},
        "citation": """@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and others},
        journal={CVPR},
        year={2009}
    }""",
    }

    def __init__(self, streaming: bool = True, **kwargs):
        self.streaming = streaming
        super().__init__(**kwargs)

    def _info(self):
        return DatasetInfo(
            description="ImageNet-1K training split in TAR format with optional streaming.",
            features=Features({"image": Image(), "label": ClassLabel(names=_default_class_names(1000))}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        train_path = download(self.SOURCE["assets"]["train"], dest_folder=self._raw_download_dir)
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": train_path})]

    def _generate_examples(self, data_path, split=None):
        yield from self._iter_train_examples(Path(data_path), class_limit=None)
