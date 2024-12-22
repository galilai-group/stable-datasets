import os
import pandas as pd
import datasets
from PIL import Image
import tempfile


class HASYv2(datasets.GeneratorBasedBuilder):
    """
    The HASYv2 dataset contains handwritten symbol images of 369 classes.
    Each image is 32x32 pixels in size.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""The HASYv2 dataset contains 32x32 black-and-white images of 369 handwritten symbol classes.
                           It includes over 168,236 samples categorized into various classes like Latin characters, numerals, and symbols.""",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/MartinThoma/HASY",
            citation="""@article{thoma2017hasyv2,
                         title={The hasyv2 dataset},
                         author={Thoma, Martin},
                         journal={arXiv preprint arXiv:1701.08380},
                         year={2017}}""",
        )

    def _split_generators(self, dl_manager):
        url = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1"
        archive_path = dl_manager.download_and_extract(url)

        fold_1_dir = os.path.join(archive_path, "classification-task/fold-1")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"csv_path": os.path.join(fold_1_dir, "train.csv"), "base_dir": archive_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"csv_path": os.path.join(fold_1_dir, "test.csv"), "base_dir": archive_path},
            ),
        ]

    def _generate_examples(self, csv_path, base_dir):
        # Read the CSV file
        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            # Resolve the full path to the image
            image_path = os.path.join(base_dir, row["path"].lstrip("../../"))

            # Open the image and convert to grayscale
            with Image.open(image_path).convert("L") as image:
                # Save the processed image to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    image.save(temp_file.name, format="PNG")
                    temp_image_path = temp_file.name

            yield idx, {
                "image": temp_image_path,  # Provide the path to the temporary file
                "label": str(row["symbol_id"]),  # Pass the label as a string
            }

    @staticmethod
    def _labels():
        return [
                "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
                "51", "52", "53", "54", "55", "56", "59", "70", "71", "72",
                "73", "74", "75", "76", "77", "78", "79", "81", "82", "87",
                "88", "89", "90", "91", "92", "93", "94", "95", "96", "97",
                "98", "99", "100", "101", "102", "103", "104", "105", "106",
                "107", "108", "110", "111", "112", "113", "114", "115", "116",
                "117", "150", "151", "152", "153", "154", "155", "156", "157",
                "158", "159", "160", "161", "162", "163", "164", "165", "166",
                "167", "168", "169", "170", "171", "174", "175", "176", "177",
                "178", "179", "180", "181", "182", "183", "184", "185", "186",
                "187", "188", "189", "190", "191", "192", "193", "194", "195",
                "196", "197", "254", "257", "259", "260", "261", "262", "263",
                "264", "265", "266", "267", "268", "269", "508", "510", "511",
                "512", "513", "514", "517", "520", "521", "523", "524", "526",
                "527", "528", "529", "530", "531", "532", "533", "534", "535",
                "536", "537", "538", "539", "540", "541", "542", "544", "549",
                "550", "553", "555", "562", "564", "574", "577", "582", "583",
                "584", "591", "595", "600", "601", "603", "604", "605", "607",
                "608", "609", "610", "611", "612", "613", "614", "615", "616",
                "617", "618", "620", "621", "622", "630", "631", "634", "635",
                "636", "639", "640", "644", "647", "650", "661", "671", "678",
                "679", "683", "684", "698", "711", "712", "713", "716", "728",
                "739", "741", "743", "748", "751", "753", "756", "757", "758",
                "759", "761", "762", "763", "764", "765", "767", "768", "770",
                "771", "775", "777", "778", "783", "785", "786", "788", "791",
                "792", "801", "809", "812", "817", "822", "823", "827", "837",
                "838", "881", "882", "884", "885", "886", "887", "888", "889",
                "890", "891", "892", "894", "901", "912", "913", "914", "915",
                "916", "917", "918", "919", "920", "921", "922", "923", "924",
                "934", "936", "941", "943", "944", "945", "946", "947", "948",
                "949", "950", "951", "953", "956", "957", "958", "959", "960",
                "965", "968", "971", "972", "973", "974", "977", "992", "993",
                "994", "995", "996", "997", "998", "999", "1000", "1004", "1005",
                "1006", "1007", "1008", "1010", "1011", "1012", "1013", "1016",
                "1018", "1019", "1031", "1037", "1042", "1045", "1046", "1051",
                "1053", "1062", "1064", "1065", "1066", "1074", "1075", "1077",
                "1078", "1079", "1080", "1082", "1086", "1090", "1093", "1101",
                "1102", "1103", "1111", "1112", "1115", "1116", "1117", "1168",
                "1169", "1177", "1184", "1185", "1187", "1314", "1315", "1316",
                "1317", "1369", "1371", "1374", "1382", "1385", "1394", "1395",
                "1396", "1400"
            ]
