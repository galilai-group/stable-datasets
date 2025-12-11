import tarfile
from tqdm import tqdm
import datasets
from PIL import Image
import io


class Country211(datasets.GeneratorBasedBuilder):
    """Country211: Image Classification Dataset for Geolocation.
    This dataset uses a subset of the YFCC100M dataset, filtered by GPS coordinates to include images labeled
    with ISO-3166 country codes. Each country has a balanced sample of images for training, validation, and testing.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Country211 dataset for image classification by country.",
            features=datasets.Features({
                "image": datasets.Image(),
                "label": datasets.ClassLabel(names=self._class_names())
            }),
            supervised_keys=("image", "label"),
            homepage="https://github.com/openai/CLIP/blob/main/data/country211.md",
            citation="""@inproceedings{radford2021learning,
                         title={Learning transferable visual models from natural language supervision},
                         author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
                         booktitle={International conference on machine learning},
                         pages={8748--8763},
                         year={2021},
                         organization={PMLR}}"""
        )

    def _split_generators(self, dl_manager):
        # Define download URL and local path
        urls = "https://openaipublic.azureedge.net/clip/data/country211.tgz"
        archive_path = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": archive_path, "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"archive_path": archive_path, "split": "valid"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"archive_path": archive_path, "split": "test"}
            ),
        ]

    def _generate_examples(self, archive_path, split):
        """Generate examples from the tar archive."""
        with tarfile.open(archive_path, "r:gz") as archive:
            # Navigate to the relevant split directory within the archive
            split_dir = f"country211/{split}"

            # Get the class names
            class_names = self._class_names()
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            # Initialize a counter for unique IDs
            idx = 0

            for member in tqdm(archive.getmembers(), desc=f"Processing {split} split"):
                # Only process files within the specific split directory
                if member.isfile() and member.name.startswith(split_dir):
                    # Extract the country code from the directory name
                    path_parts = member.name.split("/")
                    country_code = path_parts[2]

                    # Check if the country code is valid
                    if country_code in class_to_idx:
                        label = class_to_idx[country_code]

                        # Extract and open the image
                        with archive.extractfile(member) as file:
                            image = Image.open(io.BytesIO(file.read())).convert("RGB")

                            # Use the counter as the unique ID
                            yield idx, {
                                "image": image,
                                "label": label,
                            }
                            idx += 1  # Increment the counter for the next image
                    else:
                        raise ValueError(f"Invalid country code: {country_code}")

    @staticmethod
    def _class_names():
        return ['AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR', 'AT', 'AU', 'AW', 'AX', 'AZ', 'BA', 'BB',
                'BD', 'BE', 'BF', 'BG', 'BH', 'BJ', 'BM', 'BN', 'BO', 'BQ', 'BR', 'BS', 'BT', 'BW', 'BY', 'BZ', 'CA',
                'CD', 'CF', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CV', 'CW', 'CY', 'CZ', 'DE', 'DK',
                'DM', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'ET', 'FI', 'FJ', 'FK', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE',
                'GF', 'GG', 'GH', 'GI', 'GL', 'GM', 'GP', 'GR', 'GS', 'GT', 'GU', 'GY', 'HK', 'HN', 'HR', 'HT', 'HU',
                'ID', 'IE', 'IL', 'IM', 'IN', 'IQ', 'IR', 'IS', 'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KN',
                'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LT', 'LU', 'LV', 'LY', 'MA', 'MC',
                'MD', 'ME', 'MF', 'MG', 'MK', 'ML', 'MM', 'MN', 'MO', 'MQ', 'MR', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY',
                'MZ', 'NA', 'NC', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL',
                'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS', 'RU', 'RW', 'SA', 'SB', 'SC', 'SD', 'SE', 'SG',
                'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SS', 'SV', 'SX', 'SY', 'SZ', 'TG', 'TH', 'TJ', 'TL',
                'TM', 'TN', 'TO', 'TR', 'TT', 'TW', 'TZ', 'UA', 'UG', 'US', 'UY', 'UZ', 'VA', 'VE', 'VG', 'VI', 'VN',
                'VU', 'WS', 'XK', 'YE', 'ZA', 'ZM', 'ZW']
