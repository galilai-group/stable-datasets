import os
import subprocess
from pathlib import Path

import datasets

from stable_datasets.utils import BaseDatasetBuilder


class TinyImagenet(BaseDatasetBuilder):
    """
    Tiny ImageNet dataset for image classification tasks.
    It contains 200 classes with 500 training images, 50 validation images, and 50 test images per class.
    """

    VERSION = datasets.Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = {
        "homepage": "https://www.kaggle.com/c/tiny-imagenet",
        "assets": {
            "train": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
            "validation": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
            "test": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        },
        "citation": """@inproceedings{Le2015TinyIV,
                          title={Tiny ImageNet Visual Recognition Challenge},
                          author={Ya Le and Xuan S. Yang},
                          year={2015}
                        }""",
        "license": "MIT License",
    }

    def _info(self):
        source = self._source()
        return datasets.DatasetInfo(
            description="""In Tiny ImageNet, there are 100,000 images divided up into 200 classes. Every image in the
            dataset is downsized to a 64×64 colored image. For every class, there are 500 training images, 50 validating
            images, and 50 test images.""",
            features=datasets.Features(
                {"image": datasets.Image(), "label": datasets.ClassLabel(names=self._labels())}
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
            license=source.get("license", None),
        )

    def _generate_examples(self, data_path, split):
        if os.path.isfile(data_path) and data_path._str.lower().endswith(".zip"):
            extract_dir = Path(
                os.path.join(os.path.dirname(data_path), os.path.splitext(os.path.basename(data_path))[0])
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            print("[tiny_imagenet] Extracting dataset... (this may take a while)")
            subprocess.run(["unzip", "-nq", data_path, "-d", str(extract_dir)], check=True)
            base_path = (
                os.path.join(extract_dir, "tiny-imagenet-200")
                if os.path.isdir(os.path.join(extract_dir, "tiny-imagenet-200"))
                else extract_dir
            )
        else:
            base_path = (
                os.path.join(data_path, "tiny-imagenet-200")
                if os.path.isdir(os.path.join(data_path, "tiny-imagenet-200"))
                else data_path
            )

        print(f"[tiny_imagenet] generating examples for split={split} from {base_path}")

        if split == "train":
            label_dirs = sorted(os.listdir(os.path.join(base_path, "train")))
            total_labels = len(label_dirs)
            processed = 0
            for label_dir in label_dirs:
                class_path = os.path.join(base_path, "train", label_dir, "images")
                files = os.listdir(class_path)
                for image_file in files:
                    image_path = os.path.join(class_path, image_file)
                    yield image_file, {"image": image_path, "label": label_dir}
                processed += 1
                if processed % 50 == 0 or processed == total_labels:
                    print(f"[tiny_imagenet] processed {processed}/{total_labels} classes")

        elif split == "validation":
            annotations = os.path.join(base_path, "val", "val_annotations.txt")
            print(f"[tiny_imagenet] reading validation annotations: {annotations}")
            with open(annotations) as f:
                for i, line in enumerate(f):
                    image_file, label, *_ = line.strip().split("\t")
                    image_path = os.path.join(base_path, "val", "images", image_file)
                    if i and i % 1000 == 0:
                        print(f"[tiny_imagenet] yielded {i} validation examples")
                    yield image_file, {"image": image_path, "label": label}
        elif split == "test":
            test_dir = os.path.join(base_path, "test", "images")
            files = os.listdir(test_dir)
            for i, image_file in enumerate(files):
                image_path = os.path.join(test_dir, image_file)
                if i and i % 1000 == 0:
                    print(f"[tiny_imagenet] yielded {i} test examples")
                yield image_file, {"image": image_path, "label": -1}  # No labels for test set
        else:
            raise ValueError(f"Unknown split: {split} | expected one of 'train', 'validation', 'test'")

    @staticmethod
    def _labels():
        return [
            "n02124075",
            "n04067472",
            "n04540053",
            "n04099969",
            "n07749582",
            "n01641577",
            "n02802426",
            "n09246464",
            "n07920052",
            "n03970156",
            "n03891332",
            "n02106662",
            "n03201208",
            "n02279972",
            "n02132136",
            "n04146614",
            "n07873807",
            "n02364673",
            "n04507155",
            "n03854065",
            "n03838899",
            "n03733131",
            "n01443537",
            "n07875152",
            "n03544143",
            "n09428293",
            "n03085013",
            "n02437312",
            "n07614500",
            "n03804744",
            "n04265275",
            "n02963159",
            "n02486410",
            "n01944390",
            "n09256479",
            "n02058221",
            "n04275548",
            "n02321529",
            "n02769748",
            "n02099712",
            "n07695742",
            "n02056570",
            "n02281406",
            "n01774750",
            "n02509815",
            "n03983396",
            "n07753592",
            "n04254777",
            "n02233338",
            "n04008634",
            "n02823428",
            "n02236044",
            "n03393912",
            "n07583066",
            "n04074963",
            "n01629819",
            "n09332890",
            "n02481823",
            "n03902125",
            "n03404251",
            "n09193705",
            "n03637318",
            "n04456115",
            "n02666196",
            "n03796401",
            "n02795169",
            "n02123045",
            "n01855672",
            "n01882714",
            "n02917067",
            "n02988304",
            "n04398044",
            "n02843684",
            "n02423022",
            "n02669723",
            "n04465501",
            "n02165456",
            "n03770439",
            "n02099601",
            "n04486054",
            "n02950826",
            "n03814639",
            "n04259630",
            "n03424325",
            "n02948072",
            "n03179701",
            "n03400231",
            "n02206856",
            "n03160309",
            "n01984695",
            "n03977966",
            "n03584254",
            "n04023962",
            "n02814860",
            "n01910747",
            "n04596742",
            "n03992509",
            "n04133789",
            "n03937543",
            "n02927161",
            "n01945685",
            "n02395406",
            "n02125311",
            "n03126707",
            "n04532106",
            "n02268443",
            "n02977058",
            "n07734744",
            "n03599486",
            "n04562935",
            "n03014705",
            "n04251144",
            "n04356056",
            "n02190166",
            "n03670208",
            "n02002724",
            "n02074367",
            "n04285008",
            "n04560804",
            "n04366367",
            "n02403003",
            "n07615774",
            "n04501370",
            "n03026506",
            "n02906734",
            "n01770393",
            "n04597913",
            "n03930313",
            "n04118538",
            "n04179913",
            "n04311004",
            "n02123394",
            "n04070727",
            "n02793495",
            "n02730930",
            "n02094433",
            "n04371430",
            "n04328186",
            "n03649909",
            "n04417672",
            "n03388043",
            "n01774384",
            "n02837789",
            "n07579787",
            "n04399382",
            "n02791270",
            "n03089624",
            "n02814533",
            "n04149813",
            "n07747607",
            "n03355925",
            "n01983481",
            "n04487081",
            "n03250847",
            "n03255030",
            "n02892201",
            "n02883205",
            "n03100240",
            "n02415577",
            "n02480495",
            "n01698640",
            "n01784675",
            "n04376876",
            "n03444034",
            "n01917289",
            "n01950731",
            "n03042490",
            "n07711569",
            "n04532670",
            "n03763968",
            "n07768694",
            "n02999410",
            "n03617480",
            "n06596364",
            "n01768244",
            "n02410509",
            "n03976657",
            "n01742172",
            "n03980874",
            "n02808440",
            "n02226429",
            "n02231487",
            "n02085620",
            "n01644900",
            "n02129165",
            "n02699494",
            "n03837869",
            "n02815834",
            "n07720875",
            "n02788148",
            "n02909870",
            "n03706229",
            "n07871810",
            "n03447447",
            "n02113799",
            "n12267677",
            "n03662601",
            "n02841315",
            "n07715103",
            "n02504458",
        ]
