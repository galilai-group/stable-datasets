import zipfile
from pathlib import Path

from stable_datasets.schema import BuilderConfig, DatasetInfo, DatasetSource, DownloadInfo, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download

from ._multi_domain import (
    ensure_rgb,
    google_drive_url,
    image_members,
    leave_one_domain_out_configs,
    multi_domain_features,
    normalize_name,
    stratified_test_keys,
)


_DOMAINS = ("art", "clipart", "product", "real_world")
_RANDOM = "random"
_TEST_FRACTION = 0.2


class OfficeHome(BaseDatasetBuilder):
    """Office-Home.

    A domain adaptation / generalization benchmark with 15,588 images of 65 everyday
    office and home objects in 4 domains: art (2,427), clipart (4,365), product (4,439)
    and real world (4,357). The class set is shared across domains.

    The release has no official split, so the configs define one:

    ``"random"`` (default)
        Stratified 80/20 train/test split inside every (domain, class) pair. Images are
        assigned by a SHA-256 of their bytes, so the split is reproducible.
    ``"art"``, ``"clipart"``, ``"product"``, ``"real_world"``
        Leave-one-domain-out: ``train`` is every image of the other three domains and
        ``test`` is every image of the named domain.

    Every example has ``image`` (RGB, variable size), ``label`` (65 classes) and ``domain`` (4 domains).

    Usage:
        train = OfficeHome(split="train")
        target_test = OfficeHome(config_name="clipart", split="test")
    """

    VERSION = Version("1.0.0")

    DOMAINS = _DOMAINS
    CLASS_NAMES = (
        "alarm_clock", "backpack", "batteries", "bed", "bike", "bottle", "bucket", "calculator",
        "calendar", "candles", "chair", "clipboards", "computer", "couch", "curtains", "desk_lamp",
        "drill", "eraser", "exit_sign", "fan", "file_cabinet", "flipflops", "flowers", "folder",
        "fork", "glasses", "hammer", "helmet", "kettle", "keyboard", "knives", "lamp_shade",
        "laptop", "marker", "monitor", "mop", "mouse", "mug", "notebook", "oven",
        "pan", "paper_clip", "pen", "pencil", "postit_notes", "printer", "push_pin", "radio",
        "refrigerator", "ruler", "scissors", "screwdriver", "shelf", "sink", "sneakers", "soda",
        "speaker", "spoon", "tv", "table", "telephone", "toothbrush", "toys", "trash_can",
        "webcam",
    )  # fmt: skip

    BUILDER_CONFIGS = [
        BuilderConfig(
            name=_RANDOM, description="Stratified 80/20 train/test split within every (domain, class) pair."
        ),
        *leave_one_domain_out_configs(
            "Office-Home",
            _DOMAINS,
            "Leave-one-domain-out: train on the other domains, test on every image of '{domain}'.",
        ),
    ]
    DEFAULT_CONFIG_NAME = _RANDOM

    SOURCE = DatasetSource(
        homepage="https://www.hemanthdv.org/officeHomeDataset.html",
        assets={
            "archive": DownloadInfo(
                url=google_drive_url("0B81rNlvomiwed0V1YUxQdC1uOTg", "0-2SNWq0CDAuWOBRRBL7ZZsw"),
                filename="OfficeHomeDataset_10072016.zip",
            ),
        },
        license="Custom fair-use notice, non-commercial research and educational use only",
        citation="""@inproceedings{venkateswara2017deep,
            title={Deep Hashing Network for Unsupervised Domain Adaptation},
            author={Venkateswara, Hemanth and Eusebio, Jose and Chakraborty, Shayok and Panchanathan, Sethuraman},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            pages={5018--5027},
            year={2017}}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Office-Home contains 15,588 images of 65 office and home object classes in 4 domains: "
                f"art, clipart, product and real world. Config '{self.config.name}': {self.config.description}"
            ),
            features=multi_domain_features(self.CLASS_NAMES, self.DOMAINS),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list[str]:
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        info = self._normalize_download_info(self._source()["assets"]["archive"], asset_name="archive")
        download_dir = getattr(self, "_raw_download_dir", None) or _default_dest_folder()
        archive_path = Path(bulk_download([info], dest_folder=download_dir)[0])
        return [
            SplitGenerator(name=split, gen_kwargs={"archive_path": archive_path, "split": split})
            for split in self._candidate_splits()
        ]

    def _iter_images(self, archive):
        """Yield ``(key, member, label, domain)`` for every image, in archive order."""
        label_feature = self.info.features["label"]
        domain_feature = self.info.features["domain"]
        for key, member in image_members(archive, self.DOMAINS).items():
            domain, class_name = key.split("/")[:2]
            class_name = normalize_name(class_name)
            if class_name not in label_feature.names:
                raise ValueError(f"OfficeHome: unexpected class folder {class_name!r} in {member}.")
            yield key, member, label_feature.str2int(class_name), domain_feature.str2int(domain)

    def _generate_examples(self, archive_path, split):
        with zipfile.ZipFile(archive_path) as archive:
            if self.config.name == _RANDOM:
                # Both splits need the same test set, so hash the images once per builder.
                if getattr(self, "_test_keys", None) is None:
                    self._test_keys = stratified_test_keys(
                        (
                            (key, archive.read(member), (domain, label))
                            for key, member, label, domain in self._iter_images(archive)
                        ),
                        _TEST_FRACTION,
                    )
                test_keys = self._test_keys
                in_test = lambda key, domain: key in test_keys  # noqa: E731
            else:
                held_out = self.info.features["domain"].str2int(self.config.name)
                in_test = lambda key, domain: domain == held_out  # noqa: E731

            for key, member, label, domain in self._iter_images(archive):
                if in_test(key, domain) == (split == Split.TEST):
                    yield key, {"image": ensure_rgb(archive.read(member)), "label": label, "domain": domain}
