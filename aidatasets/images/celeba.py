from tqdm import tqdm
import zipfile
import datasets
import pandas as pd
from PIL import Image
from pathlib import Path
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown


class CelebA(datasets.GeneratorBasedBuilder):
    """
    The CelebA dataset is a large-scale face attributes dataset with more than 200K celebrity images,
    each with 40 attribute annotations.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""CelebA is a large-scale face attributes dataset with 200K images and 40 attribute annotations per image, 
                           useful for face attribute recognition, detection, and landmark localization tasks.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "attributes": datasets.Sequence(datasets.ClassLabel(names=["-1", "1"])),  # Binary attributes
                }
            ),
            supervised_keys=("image", "attributes"),
            homepage="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
            citation="""@inproceedings{liu2015faceattributes,
                         title = {Deep Learning Face Attributes in the Wild},
                         author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
                         booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
                         month = {December},
                         year = {2015}}""",
        )

    def _split_generators(self, dl_manager):
        # Define a manual cache directory path
        cache_dir = Path.home() / ".cache/huggingface/datasets/celebA"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Google Drive file IDs
        archive_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        attr_id = "0B7EVK8r0v71pblRyaVFSWGxPY0U"
        partition_id = "0B7EVK8r0v71pY0NSMzRuSXJEVkk"

        # Define file paths in the cache directory
        archive_path = cache_dir / "img_align_celeba.zip"
        attr_path = cache_dir / "list_attr_celeba.txt"
        partition_path = cache_dir / "list_eval_partition.txt"

        # Download files using gdown to the cache directory
        if not archive_path.exists():
            gdown.download(f"https://drive.google.com/uc?export=download&id={archive_id}", str(archive_path), quiet=False)
        if not attr_path.exists():
            gdown.download(f"https://drive.google.com/uc?export=download&id={attr_id}", str(attr_path), quiet=False)
        if not partition_path.exists():
            gdown.download(f"https://drive.google.com/uc?export=download&id={partition_id}", str(partition_path), quiet=False)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": str(archive_path), "attr_path": str(attr_path), "partition_path": str(partition_path), "split": 0},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"archive_path": str(archive_path), "attr_path": str(attr_path), "partition_path": str(partition_path), "split": 1},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": str(archive_path), "attr_path": str(attr_path), "partition_path": str(partition_path), "split": 2},
            ),
        ]

    def _generate_examples(self, archive_path, attr_path, partition_path, split):
        # Load attribute data
        with open(attr_path, "r") as f:
            lines = f.readlines()
            attributes = [line.split()[1:] for line in lines[2:]]  # Skip header lines
            image_ids = [line.split()[0] for line in lines[2:]]

        # Load partition data
        partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None, names=["image_id", "split"])
        split_indices = partition_df[partition_df["split"] == split].index
        start_idx, end_idx = split_indices[0], split_indices[-1] + 1  # end_idx is non-inclusive

        # Slice attributes and image IDs for the split range
        split_image_ids = image_ids[start_idx:end_idx]
        split_attributes = attributes[start_idx:end_idx]

        # Open the zip file and process each image
        with zipfile.ZipFile(archive_path, "r") as z:
            for idx, image_name in enumerate(tqdm(split_image_ids, desc=f"Processing split {split}")):
                with z.open(f"img_align_celeba/{image_name}") as img_file:
                    image = Image.open(img_file).convert("RGB")

                    # Get attributes for this image and convert them to integers (-1 or 1)
                    attributes = [int(attr) for attr in split_attributes[idx]]

                    yield idx, {
                        "image": image,
                        "attributes": attributes,
                    }
