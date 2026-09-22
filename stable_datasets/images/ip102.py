import tarfile
from pathlib import Path, PurePosixPath

import gdown
from filelock import FileLock
from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


class IP102(BaseDatasetBuilder):
    """Original IP102 classification images with the published train/validation/test lists.

    The original image resolutions and zero-based split labels are retained.
    Class names follow the authors' 1--102 classes.txt order.
    """

    VERSION = Version("1.0.0")
    SOURCE = {
        "homepage": "https://github.com/xpwu95/IP102",
        "assets": {
            "train": "https://drive.google.com/uc?export=download&id=1OVXLA4O_RRQvFU3-6uPpM9q5cQ75GoE2",
            "val": "https://drive.google.com/uc?export=download&id=1QqyduB17VAcCUEhYCN1EkYHKgl16xZ8F",
            "test": "https://drive.google.com/uc?export=download&id=1hO8gBoQjdpm9j7lBM2424MSdFUKvcKyj",
            "archive": "https://drive.google.com/uc?export=download&id=1rtFkWBj08vvgevCw3bMU5b9xc6LK9lSP",
        },
        "release": "https://drive.google.com/drive/folders/1dnw8Z4XIADWBdLX0ecokRwqVmRKJa-QP",
        "class_names": "https://raw.githubusercontent.com/xpwu95/IP102/abd7defa50dd7268eaa3e7db5bca4ad29ccd17f4/classes.txt",
        "citation": """@inproceedings{Wu2019Insect,
            title = {IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition},
            author = {Xiaoping Wu and Chi Zhan and Yukun Lai and Ming-Ming Cheng and Jufeng Yang},
            booktitle = {IEEE CVPR},
            pages = {8787--8796},
            year = {2019}
        }""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "IP102: 75,222 images of 102 insect pest classes with the official "
                "45,095 / 7,508 / 22,619 train/validation/test split."
            ),
            features=Features({"image": ImageFeature(), "label": ClassLabel(names=self._labels())}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license="Free for academic usage; contact the authors for other purposes.",
        )

    def _split_generators(self):
        cache_dir = Path(self._raw_download_dir) / "ip102"
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        # Use one release's image names and matching manifests, not the v1.1 renames.
        for key, url in self._source()["assets"].items():
            destination = cache_dir / ("Images.tar" if key == "archive" else f"{key}.txt")
            with FileLock(str(destination) + ".lock"):
                if not destination.exists():
                    temporary = destination.with_name(destination.name + ".part")
                    result = gdown.download(url, str(temporary), quiet=False, resume=True)
                    if result is None or not temporary.is_file():
                        raise RuntimeError(
                            f"IP102 {key} download failed. Download the classification files from "
                            f"{self.SOURCE['release']} and place them in {cache_dir}."
                        )
                    if key == "archive":
                        with tarfile.open(temporary, "r:*") as archive:
                            archive.getmembers()
                    else:
                        self._parse_labels(temporary)
                    temporary.replace(destination)
            paths[key] = destination
        self._read_split_labels(paths)
        return [
            SplitGenerator(name=name, gen_kwargs={"path_map": paths, "split": source_split})
            for name, source_split in [(Split.TRAIN, "train"), (Split.VALIDATION, "val"), (Split.TEST, "test")]
        ]

    @staticmethod
    def _parse_labels(path):
        labels = {}
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"Invalid IP102 row in {path}: {line.strip()!r}.")
                filename, raw_label = fields
                if PurePosixPath(filename).name != filename or not filename.endswith(".jpg"):
                    raise ValueError(f"Invalid IP102 filename: {filename!r}.")
                label = int(raw_label)
                if not 0 <= label < 102:
                    raise ValueError(f"Invalid IP102 label {raw_label!r} for {filename}.")
                if filename in labels:
                    raise ValueError(f"Duplicate IP102 filename in {path}: {filename}.")
                labels[filename] = label
        if not labels:
            raise ValueError(f"Empty IP102 split list: {path}.")
        return labels

    @classmethod
    def _read_split_labels(cls, paths):
        labels_by_split = {}
        seen = set()
        for split in ("train", "val", "test"):
            labels = cls._parse_labels(paths[split])
            overlap = seen.intersection(labels)
            if overlap:
                raise ValueError(f"IP102 filenames appear in multiple splits: {sorted(overlap)[:3]}.")
            seen.update(labels)
            labels_by_split[split] = labels
        return labels_by_split

    def _generate_examples(self, path_map, split):
        labels = self._read_split_labels(path_map)[split]
        with tarfile.open(path_map["archive"], "r:*") as archive:
            members = {}
            for member in archive:
                filename = PurePosixPath(member.name).name
                if member.isfile() and filename in labels:
                    if filename in members:
                        raise ValueError(f"Duplicate IP102 image in archive: {filename}.")
                    members[filename] = member
            for filename, label in labels.items():
                member = members[filename]
                with archive.extractfile(member) as source, PILImage.open(source) as original:
                    image = original.convert("RGB")
                yield member.name, {"image": image, "label": label}

    @staticmethod
    def _labels():
        """Class names in the authors' classes.txt order, including original spelling."""
        return [
            "rice leaf roller",
            "rice leaf caterpillar",
            "paddy stem maggot",
            "asiatic rice borer",
            "yellow rice borer",
            "rice gall midge",
            "Rice Stemfly",
            "brown plant hopper",
            "white backed plant hopper",
            "small brown plant hopper",
            "rice water weevil",
            "rice leafhopper",
            "grain spreader thrips",
            "rice shell pest",
            "grub",
            "mole cricket",
            "wireworm",
            "white margined moth",
            "black cutworm",
            "large cutworm",
            "yellow cutworm",
            "red spider",
            "corn borer",
            "army worm",
            "aphids",
            "Potosiabre vitarsis",
            "peach borer",
            "english grain aphid",
            "green bug",
            "bird cherry-oataphid",
            "wheat blossom midge",
            "penthaleus major",
            "longlegged spider mite",
            "wheat phloeothrips",
            "wheat sawfly",
            "cerodonta denticornis",
            "beet fly",
            "flea beetle",
            "cabbage army worm",
            "beet army worm",
            "Beet spot flies",
            "meadow moth",
            "beet weevil",
            "sericaorient alismots chulsky",
            "alfalfa weevil",
            "flax budworm",
            "alfalfa plant bug",
            "tarnished plant bug",
            "Locustoidea",
            "lytta polita",
            "legume blister beetle",
            "blister beetle",
            "therioaphis maculata Buckton",
            "odontothrips loti",
            "Thrips",
            "alfalfa seed chalcid",
            "Pieris canidia",
            "Apolygus lucorum",
            "Limacodidae",
            "Viteus vitifoliae",
            "Colomerus vitis",
            "Brevipoalpus lewisi McGregor",
            "oides decempunctata",
            "Polyphagotars onemus latus",
            "Pseudococcus comstocki Kuwana",
            "parathrene regalis",
            "Ampelophaga",
            "Lycorma delicatula",
            "Xylotrechus",
            "Cicadella viridis",
            "Miridae",
            "Trialeurodes vaporariorum",
            "Erythroneura apicalis",
            "Papilio xuthus",
            "Panonchus citri McGregor",
            "Phyllocoptes oleiverus ashmead",
            "Icerya purchasi Maskell",
            "Unaspis yanonensis",
            "Ceroplastes rubens",
            "Chrysomphalus aonidum",
            "Parlatoria zizyphus Lucus",
            "Nipaecoccus vastalor",
            "Aleurocanthus spiniferus",
            "Tetradacus c Bactrocera minax",
            "Dacus dorsalis(Hendel)",
            "Bactrocera tsuneonis",
            "Prodenia litura",
            "Adristyrannus",
            "Phyllocnistis citrella Stainton",
            "Toxoptera citricidus",
            "Toxoptera aurantii",
            "Aphis citricola Vander Goot",
            "Scirtothrips dorsalis Hood",
            "Dasineura sp",
            "Lawana imitata Melichar",
            "Salurnis marginella Guerr",
            "Deporaus marginatus Pascoe",
            "Chlumetia transversa",
            "Mango flat beak leafhopper",
            "Rhytidodera bowrinii white",
            "Sternochetus frigidus",
            "Cicadellidae",
        ]
