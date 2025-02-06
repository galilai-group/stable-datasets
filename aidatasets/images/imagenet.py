import os
import json
import tarfile
import urllib.request
import datasets
from PIL import Image
import io


class ImageNet(datasets.GeneratorBasedBuilder):
    """ImageNet (ILSVRC2012) dataset.

    The training archive contains one tar file per class (named e.g. "n01440764.tar"),
    and the mapping from WordNet ID (wnid) to label is derived from a JSON file.

    The validation archive contains all images (with filenames such as "ILSVRC2012_val_00000001.JPEG").
    The ground truth labels are read from a file contained in the devkit archive.
    """

    VERSION = datasets.Version("1.0.0")

    # URLs for automatic download
    _TRAIN_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
    _VAL_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    _DEVKIT_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
    _CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

    def _info(self):
        # Download the mapping file directly (this file is very small).
        with urllib.request.urlopen(self._CLASS_INDEX_URL) as response:
            mapping = json.load(response)
        # mapping is a dict with keys "0", "1", …, "999"
        # Each entry is a list: [wnid, class_name].
        class_names = [f"{mapping[str(i)][0]}: {mapping[str(i)][1]}" for i in range(1000)]
        return datasets.DatasetInfo(
            description="ImageNet Large Scale Visual Recognition Challenge 2012 dataset.",
            features=datasets.Features({
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=class_names),
            }),
            supervised_keys=("image", "label"),
            homepage="http://www.image-net.org/challenges/LSVRC/2012/",
            citation="""@article{ILSVRC15,
                        Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                        Title = {{ImageNet Large Scale Visual Recognition Challenge}},
                        Year = {2015},
                        journal   = {International Journal of Computer Vision (IJCV)},
                        doi = {10.1007/s11263-015-0816-y},
                        volume={115},
                        number={3},
                        pages={211-252}}"""
        )

    def _split_generators(self, dl_manager):
        # Automatically download the archives using the DownloadManager.
        # train_archive = dl_manager.download(self._TRAIN_URL)
        # val_archive = dl_manager.download(self._VAL_URL)
        train_archive = "/gpfs/data/shared/imagenet/ILSVRC2012/ILSVRC2012_img_train.tar"
        val_archive = "/gpfs/data/shared/imagenet/ILSVRC2012/ILSVRC2012_img_val.tar"
        devkit_archive = dl_manager.download(self._DEVKIT_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": train_archive, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archive_path": val_archive,
                    "split": "validation",
                    "devkit_archive": devkit_archive,
                },
            ),
        ]

    def _generate_examples(self, archive_path, split, devkit_archive=None):
        if split == "train":
            # Open the training tar archive.
            with tarfile.open(archive_path, "r:*") as train_tar:
                # The training archive contains many tar files (one per class).
                sub_tar_members = [
                    m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")
                ]
                # Download and cache the class mapping (if not already done).
                if not hasattr(self, "_mapping"):
                    with urllib.request.urlopen(self._CLASS_INDEX_URL) as response:
                        self._mapping = json.load(response)
                mapping = self._mapping
                # Build a mapping from WordNet ID (wnid) to integer label.
                wnid_to_label = {mapping[str(i)][0]: i for i in range(1000)}

                example_idx = 0
                for m in sub_tar_members:
                    # Each member’s name is e.g. "n01440764.tar"; extract the wnid.
                    wnid = os.path.splitext(m.name)[0]
                    if wnid not in wnid_to_label:
                        # Skip any unexpected files.
                        continue
                    label = wnid_to_label[wnid]
                    sub_tar_file = train_tar.extractfile(m)
                    # Open the inner tar file containing images for this class.
                    with tarfile.open(fileobj=sub_tar_file, mode="r:*") as sub_tar:
                        for sub_m in sub_tar.getmembers():
                            if sub_m.isfile():
                                img_f = sub_tar.extractfile(sub_m)
                                image_bytes = img_f.read()
                                img = Image.open(io.BytesIO(image_bytes))
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                yield example_idx, {
                                    "image": img,
                                    "label": label,
                                }
                                example_idx += 1

        elif split == "validation":
            # For validation, open the validation tar archive.
            with tarfile.open(archive_path, "r:*") as val_tar:
                # Get all file members (each a validation image).
                members = [m for m in val_tar.getmembers() if m.isfile()]
                # Sort by filename to ensure the same order as in the ground truth file.
                members = sorted(members, key=lambda m: m.name)
                # Open the devkit archive to extract the ground truth file.
                with tarfile.open(devkit_archive, "r:*") as devkit_tar:
                    gt_member = None
                    for m in devkit_tar.getmembers():
                        if m.name.endswith("ILSVRC2012_validation_ground_truth.txt"):
                            gt_member = m
                            break
                    if gt_member is None:
                        raise ValueError("Could not find the ground truth file in the devkit archive.")
                    gt_file = devkit_tar.extractfile(gt_member)
                    gt_lines = gt_file.read().decode("utf-8").strip().splitlines()
                if len(gt_lines) != len(members):
                    raise ValueError("Mismatch between the number of validation images and ground truth labels.")
                for example_idx, (m, gt) in enumerate(zip(members, gt_lines)):
                    # Convert the ground truth label from 1-indexed to 0-indexed.
                    label = int(gt) - 1
                    img_f = val_tar.extractfile(m)
                    image_bytes = img_f.read()
                    img = Image.open(io.BytesIO(image_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    yield example_idx, {
                        "image": img,
                        "label": label,
                    }
