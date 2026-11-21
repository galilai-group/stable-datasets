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
)


_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
_ALL_DOMAINS = "all"

# Google Drive IDs of the original release (https://dali-dl.github.io/project_iccv2017.html).
_IMAGES_ID = ("0B6x7gtvErXgfbF9CSk53UkRxVzg", "0-aQEyqs-mKSsoWHvclYvgjw")  # Raw images/PACS.zip
_SPLIT_IDS = {
    ("art_painting", "train"): "1cSo_5_cQLKoURsWwsKqFmw9neWFvDIeV",
    ("art_painting", "crossval"): "1g7haGuvxmZA1SxPyGFZneDVi9wOm3nK4",
    ("art_painting", "test"): "1KVe_S73jECKG9MsIujF-yqZAjPrf0y_O",
    ("cartoon", "train"): "1EhlIyRoeZ5UPiUJs_UXGH-Ql1PS5qtYW",
    ("cartoon", "crossval"): "1suUUrxyPOmN_dAVdhVSaG-NZDMOsl8JR",
    ("cartoon", "test"): "14fpMKzcherr1onzp8jcyhmjJMjZSZ72h",
    ("photo", "train"): "1RyCVv67zVITU4JYYr1XxLnG5q165yu8s",
    ("photo", "crossval"): "1lXtOcPL-yWli_MuDZ0mQwLfq0XbzDu--",
    ("photo", "test"): "1opW_IbwDQQmuylquX5UvRFxbj3muP5sz",
    ("sketch", "train"): "13rWm64rW_dcdcyTFugu4fJcSCTkCB8YA",
    ("sketch", "crossval"): "1Rs0dUTduQ320bXwHBLQT1s82gyr92J9J",
    ("sketch", "test"): "1dFnnsWxYHvdCXmWdhPVwuJ8zgkb_wtbQ",
}


def _split_file(domain: str, part: str) -> str:
    return f"{domain}_{part}_kfold"


class PACS(BaseDatasetBuilder):
    """PACS: Photo, Art painting, Cartoon and Sketch.

    A domain generalization benchmark with 9,991 images of 7 object classes in 4 visual
    domains: photo (1,670), art painting (2,048), cartoon (2,344) and sketch (3,929).

    The original release comes with an official train/validation split for every domain
    (``<domain>_train_kfold.txt`` / ``<domain>_crossval_kfold.txt``), which the authors ask
    to be used for fair comparisons. The configs follow it:

    ``"all"`` (default)
        ``train`` and ``validation`` are the official splits of all four domains.
    ``"photo"``, ``"art_painting"``, ``"cartoon"``, ``"sketch"``
        Leave-one-domain-out: ``train`` and ``validation`` are the official splits of the
        other three domains, and ``test`` is every image of the named domain
        (``<domain>_test_kfold.txt``).

    Every example has ``image`` (227x227 RGB), ``label`` (7 classes) and ``domain`` (4 domains).

    Usage:
        train = PACS(split="train")
        source_train = PACS(config_name="sketch", split="train")
        target_test = PACS(config_name="sketch", split="test")
    """

    VERSION = Version("1.0.0")

    DOMAINS = _DOMAINS
    CLASS_NAMES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")

    BUILDER_CONFIGS = [
        BuilderConfig(name=_ALL_DOMAINS, description="Official train/validation splits of all four domains."),
        *leave_one_domain_out_configs(
            "PACS",
            _DOMAINS,
            "Leave-one-domain-out: train/validation are the official splits of the other domains, "
            "test is every image of '{domain}'.",
        ),
    ]
    DEFAULT_CONFIG_NAME = _ALL_DOMAINS

    SOURCE = DatasetSource(
        homepage="https://dali-dl.github.io/project_iccv2017.html",
        assets={
            "images": DownloadInfo(url=google_drive_url(*_IMAGES_ID), filename="PACS.zip"),
            **{
                _split_file(domain, part): DownloadInfo(
                    url=google_drive_url(file_id), filename=f"pacs_{_split_file(domain, part)}.txt"
                )
                for (domain, part), file_id in _SPLIT_IDS.items()
            },
        },
        license="Unknown (research use; see the original release)",
        citation="""@inproceedings{li2017deeper,
            title={Deeper, Broader and Artier Domain Generalization},
            author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy M.},
            booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
            pages={5542--5550},
            year={2017}}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "PACS contains 9,991 images of 7 classes (dog, elephant, giraffe, guitar, horse, house, person) "
                "in 4 domains (photo, art painting, cartoon, sketch), with the official per-domain "
                f"train/validation splits. Config '{self.config.name}': {self.config.description}"
            ),
            features=multi_domain_features(self.CLASS_NAMES, self.DOMAINS),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _split_files(self) -> dict[str, list[str]]:
        """Asset names of the official split files that make up each split of this config."""
        held_out = self.config.name
        sources = [d for d in self.DOMAINS if d != held_out]
        splits = {
            Split.TRAIN: [_split_file(d, "train") for d in sources],
            Split.VALIDATION: [_split_file(d, "crossval") for d in sources],
        }
        if held_out != _ALL_DOMAINS:
            splits[Split.TEST] = [_split_file(held_out, "test")]
        return splits

    def _candidate_splits(self) -> list[str]:
        return list(self._split_files())

    def _split_generators(self):
        source = self._source()
        names = list(source["assets"])
        infos = [self._normalize_download_info(source["assets"][n], asset_name=n) for n in names]
        download_dir = getattr(self, "_raw_download_dir", None) or _default_dest_folder()
        paths = dict(zip(names, (Path(p) for p in bulk_download(infos, dest_folder=download_dir))))
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={"archive_path": paths["images"], "split_paths": [paths[f] for f in files]},
            )
            for split, files in self._split_files().items()
        ]

    def _generate_examples(self, archive_path, split_paths):
        label_feature = self.info.features["label"]
        domain_feature = self.info.features["domain"]
        with zipfile.ZipFile(archive_path) as archive:
            members = image_members(archive, self.DOMAINS)
            for split_path in split_paths:
                for line in Path(split_path).read_text().splitlines():
                    if not line.strip():
                        continue
                    rel_path, label = line.rsplit(maxsplit=1)
                    domain, class_name = rel_path.split("/")[:2]
                    label = int(label) - 1  # the split files number classes from 1
                    if label_feature.int2str(label) != class_name:
                        raise ValueError(f"PACS: {split_path.name} labels {rel_path} as class {label + 1}.")
                    if rel_path not in members:
                        raise ValueError(
                            f"PACS: {rel_path} from {split_path.name} is missing from {archive_path.name}."
                        )
                    yield (
                        rel_path,
                        {
                            "image": ensure_rgb(archive.read(members[rel_path])),
                            "label": label,
                            "domain": domain_feature.str2int(domain),
                        },
                    )
