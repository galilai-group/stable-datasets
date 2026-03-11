import zipfile
from pathlib import Path

import gdown
import pandas as pd
from PIL import Image
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Sequence, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


class CelebA(BaseDatasetBuilder):
    """
    The CelebA dataset is a large-scale face attributes dataset with more than 200K celebrity images,
    each with 40 attribute annotations.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        "citation": """@inproceedings{liu2015faceattributes,
                         title = {Deep Learning Face Attributes in the Wild},
                         author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
                         booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
                         month = {December},
                         year = {2015}}""",
        "assets": {
            "archive": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            "attributes": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "partition": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        },
    }

    def _info(self):
        return DatasetInfo(
            description="""CelebA is a large-scale face attributes dataset with 200K images and 40 attribute annotations per image,
                           useful for face attribute recognition, detection, and landmark localization tasks.""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "attributes": Sequence(ClassLabel(names=["-1", "1"])),
                }
            ),
            supervised_keys=("image", "attributes"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        assets = source["assets"]
        cache_dir = Path(self._raw_download_dir) / "celebA"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # File names for each asset
        asset_filenames = {
            "archive": "img_align_celeba.zip",
            "attributes": "list_attr_celeba.txt",
            "partition": "list_eval_partition.txt",
        }

        paths = {}
        for key, filename in asset_filenames.items():
            dest = cache_dir / filename
            if not dest.exists():
                gdown.download(assets[key], str(dest), quiet=False)
            paths[key] = dest

        archive_path = paths["archive"]
        attr_path = paths["attributes"]
        partition_path = paths["partition"]

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "archive_path": str(archive_path),
                    "attr_path": str(attr_path),
                    "partition_path": str(partition_path),
                    "split": 0,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "archive_path": str(archive_path),
                    "attr_path": str(attr_path),
                    "partition_path": str(partition_path),
                    "split": 1,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "archive_path": str(archive_path),
                    "attr_path": str(attr_path),
                    "partition_path": str(partition_path),
                    "split": 2,
                },
            ),
        ]

    def _generate_examples(self, archive_path, attr_path, partition_path, split):
        with open(attr_path) as f:
            lines = f.readlines()
            parsed = [line.split() for line in lines[2:]]
            image_ids = [p[0] for p in parsed]
            attributes = [p[1:] for p in parsed]

        partition_df = pd.read_csv(partition_path, sep=r"\s+", header=None, names=["image_id", "split"])
        split_indices = partition_df[partition_df["split"] == split].index
        start_idx, end_idx = split_indices[0], split_indices[-1] + 1

        split_image_ids = image_ids[start_idx:end_idx]
        split_attributes = attributes[start_idx:end_idx]

        with zipfile.ZipFile(archive_path, "r") as z:
            for idx, image_name in enumerate(tqdm(split_image_ids, desc=f"Processing split {split}")):
                with z.open(f"img_align_celeba/{image_name}") as img_file:
                    image = Image.open(img_file).convert("RGB")
                    attrs = [int(attr) for attr in split_attributes[idx]]
                    yield (
                        idx,
                        {
                            "image": image,
                            "attributes": attrs,
                        },
                    )
