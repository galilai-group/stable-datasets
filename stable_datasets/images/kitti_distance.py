import collections
import os
import zipfile

import numpy as np
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


# KITTI object types. Car, Van and Truck count as "vehicles" for this task (VTAB: type index < 3).
_VEHICLE_TYPES = frozenset({"Car", "Van", "Truck"})

# Distance bins on the camera-frame z coordinate (metres), identical to VTAB's
# `_closest_vehicle_distance_pp`: label = number of thresholds strictly below the distance, minus one.
_THRESHOLDS = np.array([-100.0, 8.0, 20.0, 999.0])
_NO_VEHICLE_DISTANCE = 1000.0

# Video-level split proportions used by TFDS `kitti` (and therefore by VTAB / CLIP).
_VALIDATION_SPLIT_PERCENT_VIDEOS = 10
_TEST_SPLIT_PERCENT_VIDEOS = 10
_SPLIT_SEED = 123


class KITTIDistance(BaseDatasetBuilder):
    """KITTI Distance (closest-vehicle distance classification).

    Abstract
    A 4-way image classification task built from the KITTI 2D/3D object detection benchmark: given a single
    front-camera frame from a car driving around Karlsruhe, predict how far away the closest vehicle (car, van or
    truck) is. Classes are ``<= 8m``, ``8-20m``, ``> 20m`` and ``no vehicle``. It is the ``kitti/closest_vehicle_distance``
    task of the Visual Task Adaptation Benchmark (VTAB-1k) and the "KITTI Distance" dataset of CLIP's linear-probe
    and zero-shot evaluation suites.

    Context
    Unlike most classification benchmarks, which test object identity, this task probes spatial / depth
    understanding on real driving footage, so it is a useful check of whether representations encode geometry.

    Content

    - Images: the 7,481 labelled frames of KITTI's object ``training`` set (left colour camera, ~1242x375 PNG).
      KITTI's official ``testing`` set has no public labels and is not used.
    - Labels: derived from the 3D object annotations (``label_2``): the minimum camera-frame ``z`` over all Car / Van /
      Truck objects, binned with VTAB's thresholds.
    - Splits: frames are split by source drive (video), so no drive appears in more than one split, reproducing
      TFDS ``kitti`` (v3.x) exactly: 10% of drives for test, 10% for validation (seed 123), the rest for train.
      This gives 6,347 / 423 / 711 frames for train / validation / test.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d",
        assets={
            "images": DownloadInfo(
                url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
                checksum="sha256:351c5a2aa0cd9238b50174a3a62b846bc5855da256b82a196431d60ff8d43617",
            ),
            "labels": DownloadInfo(
                url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
                checksum="sha256:4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880",
            ),
            "devkit": DownloadInfo(
                url="https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip",
                checksum="sha256:ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1",
            ),
        },
        license="CC BY-NC-SA 3.0",
        citation="""@inproceedings{geiger2012kitti,
                    title={Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
                    author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
                    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
                    year={2012}}
                    @article{zhai2019vtab,
                    title={A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark},
                    author={Zhai, Xiaohua and Puigcerver, Joan and Kolesnikov, Alexander and Ruyssen, Pierre and
                            Riquelme, Carlos and Lucic, Mario and Djolonga, Josip and Pinto, Andre Susano and
                            Neumann, Maxim and Dosovitskiy, Alexey and others},
                    journal={arXiv preprint arXiv:1910.04867},
                    year={2019}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="""KITTI Distance: classify the distance to the closest vehicle (car, van or truck) in a
                KITTI driving frame into one of four bins: <= 8m, 8-20m, > 20m, or no vehicle (VTAB
                kitti/closest_vehicle_distance). `distance` holds the raw closest-vehicle depth in metres
                (1000.0 when there is no vehicle).""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                    "distance": Value("float32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        source = self._source()
        asset_keys = ["images", "labels", "devkit"]
        local_paths = bulk_download([source["assets"][k] for k in asset_keys], dest_folder=self._raw_download_dir)
        path_map = dict(zip(asset_keys, local_paths))

        split_ids = dict(zip(("train", "val", "test"), _build_splits(path_map["devkit"])))
        return [
            SplitGenerator(name=name, gen_kwargs={"path_map": path_map, "image_ids": split_ids[key], "split": key})
            for name, key in ((Split.TRAIN, "train"), (Split.VALIDATION, "val"), (Split.TEST, "test"))
        ]

    def _generate_examples(self, path_map, image_ids, split):
        distances = _read_closest_vehicle_distances(path_map["labels"])

        with zipfile.ZipFile(path_map["images"], "r") as archive:
            for image_id in tqdm(sorted(image_ids), desc=f"Processing KITTI Distance {split} split"):
                distance = distances[image_id]
                yield (
                    f"{image_id:06d}",
                    {
                        "image": archive.read(f"training/image_2/{image_id:06d}.png"),
                        "label": distance_to_label(distance),
                        "distance": distance,
                    },
                )

    @staticmethod
    def _labels():
        return ["<=8m", "8-20m", ">20m", "no_vehicle"]


def distance_to_label(distance: float) -> int:
    """Bin a closest-vehicle distance (metres) into the 4 VTAB classes."""
    return int(np.sum(_THRESHOLDS < distance)) - 1


def _closest_vehicle_distance(label_lines) -> float:
    """Min camera-frame z over Car/Van/Truck objects in one KITTI label file (DontCare rows are skipped)."""
    zs = [_NO_VEHICLE_DISTANCE]
    for line in label_lines:
        fields = line.split()
        if not fields or fields[0] not in _VEHICLE_TYPES:
            continue
        zs.append(float(fields[13]))  # type trunc occ alpha l t r b h w l x y z ry -> z is index 13
    return float(min(zs))


def _read_closest_vehicle_distances(labels_zip) -> dict[int, float]:
    out = {}
    with zipfile.ZipFile(labels_zip, "r") as archive:
        for name in archive.namelist():
            prefix, ext = os.path.splitext(name)
            if ext != ".txt" or not name.startswith("training/"):
                continue
            lines = archive.read(name).decode("utf-8").splitlines()
            out[int(prefix[-6:])] = _closest_vehicle_distance(lines)
    return out


def _build_splits(devkit_zip):
    """Split KITTI training frames into train/val/test by drive, exactly as TFDS `kitti` does.

    `mapping/train_rand.txt` gives, for frame i, a (1-based) line in `mapping/train_mapping.txt`, whose second
    field is the source drive. 10% of drives go to test and 10% to validation, sampled with a fixed seed.
    """
    rand_ids = mapping_lines = None
    with zipfile.ZipFile(devkit_zip, "r") as archive:
        for name in archive.namelist():
            if name.endswith("mapping/train_rand.txt"):
                text = archive.read(name).decode("utf-8")
                rand_ids = [int(x.strip()) - 1 for x in text.split(",") if x.strip()]
            elif name.endswith("mapping/train_mapping.txt"):
                mapping_lines = archive.read(name).decode("utf-8").splitlines()
    if not rand_ids or not mapping_lines:
        raise RuntimeError("KITTI devkit is missing mapping/train_rand.txt or mapping/train_mapping.txt")

    video_to_images = collections.defaultdict(list)
    for image_id, line_id in enumerate(rand_ids):
        video_to_images[mapping_lines[line_id].split(" ")[1]].append(image_id)

    # Legacy RandomState(seed) draws the same stream TFDS gets from np.random.seed(123).
    rng = np.random.RandomState(_SPLIT_SEED)
    videos = sorted(video_to_images)
    n_test = max(1, _TEST_SPLIT_PERCENT_VIDEOS * len(videos) // 100)
    n_val = max(1, _VALIDATION_SPLIT_PERCENT_VIDEOS * len(videos) // 100)
    test_videos = set(rng.choice(videos, n_test, replace=False))
    val_videos = set(rng.choice(sorted(set(videos) - test_videos), n_val, replace=False))

    train, val, test = [], [], []
    for video, ids in video_to_images.items():
        (test if video in test_videos else val if video in val_videos else train).extend(ids)
    return train, val, test
