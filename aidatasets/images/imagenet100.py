import os
import json
import tarfile
import urllib.request

import datasets


class ImageNet100(datasets.GeneratorBasedBuilder):
    """ImageNet100: A subset of ImageNet (ILSVRC2012) containing only 100 classes.

    The training archive contains one tar file per class (named, e.g., "n01440764.tar"),
    and this builder filters out all classes except for those in self._class_names().

    For the validation split the ground-truth file is used to determine the original label,
    which is then converted to a wnid (via a full JSON mapping) and filtered accordingly.
    """

    VERSION = datasets.Version("1.0.0")

    # URLs for automatic download (same as the full ImageNet builder)
    _TRAIN_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
    _VAL_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
    _DEVKIT_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
    _CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

    def _info(self):
        # Download the full class index mapping.
        with urllib.request.urlopen(self._CLASS_INDEX_URL) as response:
            full_mapping = json.load(response)
        # Build a mapping from wnid to human-readable class name.
        wnid_to_name = {value[0]: value[1] for key, value in full_mapping.items()}

        # Construct the list of class names for our subset (in the order of self._class_names())
        subset_class_names = [f"{wnid}: {wnid_to_name.get(wnid, wnid)}" for wnid in self._class_names()]

        return datasets.DatasetInfo(
            description="ImageNet100: A subset of ImageNet (ILSVRC2012) containing only 100 classes.",
            features=datasets.Features({
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=subset_class_names),
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        # Download archives automatically using the DownloadManager.
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
        # Build a mapping from the selected wnids to new label indices (0 to 99)
        wnid_to_new_label = {wnid: idx for idx, wnid in enumerate(self._class_names())}

        if split == "train":
            with tarfile.open(archive_path, "r:*") as train_tar:
                # The training tar archive contains many tar files (one per class)
                sub_tar_members = [
                    m for m in train_tar.getmembers() if m.isfile() and m.name.endswith(".tar")
                ]
                example_idx = 0
                for m in sub_tar_members:
                    # Each member’s name is, e.g., "n01440764.tar"; extract the wnid.
                    wnid = os.path.splitext(m.name)[0]
                    # Only process if the wnid is in our selected subset.
                    if wnid not in wnid_to_new_label:
                        continue
                    new_label = wnid_to_new_label[wnid]
                    sub_tar_file = train_tar.extractfile(m)
                    # Open the inner tar file containing images for this class.
                    with tarfile.open(fileobj=sub_tar_file, mode="r:*") as sub_tar:
                        for sub_m in sub_tar.getmembers():
                            if sub_m.isfile():
                                img_f = sub_tar.extractfile(sub_m)
                                image_bytes = img_f.read()
                                yield example_idx, {
                                    "image": {"bytes": image_bytes, "filename": sub_m.name},
                                    "label": new_label,
                                }
                                example_idx += 1

        elif split == "validation":
            # For validation, filter examples based on their wnid.
            with tarfile.open(archive_path, "r:*") as val_tar:
                members = [m for m in val_tar.getmembers() if m.isfile()]
                # Sort by filename to match the ground truth order.
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

                # Download the full class mapping to convert the original label to wnid.
                with urllib.request.urlopen(self._CLASS_INDEX_URL) as response:
                    full_mapping = json.load(response)
                # Build a mapping from original index (0-indexed) to wnid.
                idx_to_wnid = {int(key): value[0] for key, value in full_mapping.items()}

                example_idx = 0
                # Iterate over the validation tar members paired with the ground truth lines.
                for m, gt in zip(members, gt_lines):
                    original_label = int(gt) - 1  # original labels are 1-indexed
                    wnid = idx_to_wnid.get(original_label)
                    # Only include the example if its wnid is in our selected subset.
                    if wnid not in wnid_to_new_label:
                        continue
                    new_label = wnid_to_new_label[wnid]
                    img_f = val_tar.extractfile(m)
                    image_bytes = img_f.read()
                    yield example_idx, {
                        "image": {"bytes": image_bytes, "filename": m.name},
                        "label": new_label,
                    }
                    example_idx += 1

    @staticmethod
    def _class_names():
        return [
            "n02869837",
            "n01749939",
            "n02488291",
            "n02107142",
            "n13037406",
            "n02091831",
            "n04517823",
            "n04589890",
            "n03062245",
            "n01773797",
            "n01735189",
            "n07831146",
            "n07753275",
            "n03085013",
            "n04485082",
            "n02105505",
            "n01983481",
            "n02788148",
            "n03530642",
            "n04435653",
            "n02086910",
            "n02859443",
            "n13040303",
            "n03594734",
            "n02085620",
            "n02099849",
            "n01558993",
            "n04493381",
            "n02109047",
            "n04111531",
            "n02877765",
            "n04429376",
            "n02009229",
            "n01978455",
            "n02106550",
            "n01820546",
            "n01692333",
            "n07714571",
            "n02974003",
            "n02114855",
            "n03785016",
            "n03764736",
            "n03775546",
            "n02087046",
            "n07836838",
            "n04099969",
            "n04592741",
            "n03891251",
            "n02701002",
            "n03379051",
            "n02259212",
            "n07715103",
            "n03947888",
            "n04026417",
            "n02326432",
            "n03637318",
            "n01980166",
            "n02113799",
            "n02086240",
            "n03903868",
            "n02483362",
            "n04127249",
            "n02089973",
            "n03017168",
            "n02093428",
            "n02804414",
            "n02396427",
            "n04418357",
            "n02172182",
            "n01729322",
            "n02113978",
            "n03787032",
            "n02089867",
            "n02119022",
            "n03777754",
            "n04238763",
            "n02231487",
            "n03032252",
            "n02138441",
            "n02104029",
            "n03837869",
            "n03494278",
            "n04136333",
            "n03794056",
            "n03492542",
            "n02018207",
            "n04067472",
            "n03930630",
            "n03584829",
            "n02123045",
            "n04229816",
            "n02100583",
            "n03642806",
            "n04336792",
            "n03259280",
            "n02116738",
            "n02108089",
            "n03424325",
            "n01855672",
            "n02090622",
        ]
