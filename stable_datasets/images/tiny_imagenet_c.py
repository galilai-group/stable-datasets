import glob
import os
import subprocess
from pathlib import Path

import datasets
from loguru import logger as logging

from stable_datasets.utils import BaseDatasetBuilder


class TinyImagenetC(BaseDatasetBuilder):
    """
    Tiny ImageNet-C dataset for image classification tasks with corruptions applied.
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "https://zenodo.org/records/2536630",
        "assets": {
            "test": "https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1",
        },
        "citation": """@article{hendrycks2019robustness,
                        title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
                        author={Dan Hendrycks and Thomas Dietterich},
                        journal={Proceedings of the International Conference on Learning Representations},
                        year={2019}}""",
        "license": "CC BY 4.0",
    }

    def _info(self):
        source = self._source()
        return datasets.DatasetInfo(
            description="""The Tiny ImageNet-C dataset applies multiple corruptions to the Tiny ImageNet images. It
            includes 200 classes and various corruption levels.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                    "corruption_name": datasets.Value("string"),
                    "corruption_level": datasets.Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
            license=source.get("license", None),
        )

    def _generate_examples(self, data_path, split=None):
        data_path = str(data_path)
        if len(glob.glob(os.path.join(data_path[:-4], "Tiny-ImageNet-C", "*/*/*/*"))) == 750000:
            base_path = os.path.join(data_path[:-4], "Tiny-ImageNet-C")
            logging.info(f"[tiny_imagenet_c] Extracted dataset found at {base_path} : Total 750,000 images")
        elif os.path.isfile(data_path) and data_path.lower().endswith(".tar"):
            extract_dir = Path(
                os.path.join(os.path.dirname(data_path), os.path.splitext(os.path.basename(data_path))[0])
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            logging.info("[tiny_imagenet_c] Extracting dataset... (this may take a while)")
            cmd = f"tar -xf {data_path} -C {extract_dir} --skip-old-files"

            subprocess.run(cmd, shell=True)
            base_path = os.path.join(data_path[:-4], "Tiny-ImageNet-C")

        print(f"[tiny_imagenet_c] Generating examples from {base_path}")

        if split == "test":
            corruption_types = [
                "gaussian_noise",
                "shot_noise",
                "impulse_noise",
                "defocus_blur",
                "glass_blur",
                "motion_blur",
                "zoom_blur",
                "snow",
                "frost",
                "fog",
                "brightness",
                "contrast",
                "elastic_transform",
                "pixelate",
                "jpeg_compression",
            ]

            for corruption_name in corruption_types:
                for level in range(1, 6):
                    corruption_dir = os.path.join(base_path, corruption_name, str(level))
                    print(f"[tiny_imagenet_c] processing corruption: {corruption_name}, level: {level}")

                    if not os.path.isdir(corruption_dir):
                        logging.warning(f"[tiny_imagenet_c] missing directory {corruption_dir}, skipping")
                        continue

                    # Each corruption/level folder contains subfolders per label (e.g., n01443537)
                    for label_dir in sorted(os.listdir(corruption_dir)):
                        label_path = os.path.join(corruption_dir, label_dir)
                        if not os.path.isdir(label_path):
                            continue
                        for i, image_file in enumerate(sorted(os.listdir(label_path))):
                            image_path = os.path.join(label_path, image_file)
                            label_str = label_dir
                            yield (
                                f"{corruption_name}_{level}_{label_dir}_{image_file}",
                                {
                                    "image": image_path,
                                    "label": label_str,
                                    "corruption_name": corruption_name,
                                    "corruption_level": level,
                                },
                            )
        else:
            raise ValueError(f"Unknown split: {split} | expected 'test'")

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
