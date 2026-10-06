import ast
import csv
import io
import re
from zipfile import ZipFile

from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Value,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    bulk_download,
)


class Malaria(BaseDatasetBuilder):
    """Balanced classification of parasitized and uninfected red blood cells.

    The NIH dataset contains 27,558 variable-resolution RGB cell crops and
    has no official train/test split. `patient_id` preserves the identifier
    from NIH's patient-to-cell mappings; `source_image_id` identifies the
    microscopy image before individual cells were cropped. Group by patient
    when constructing splits to avoid sharing a patient's cells across them.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-project.html",
        citation="""@article{rajaraman2018pre,
            title={Pre-trained convolutional neural networks as feature extractors toward
                   improved malaria parasite detection in thin blood smear images},
            author={Rajaraman, Sivaramakrishnan and Antani, Sameer K and Poostchi, Mahdieh
                    and Silamut, Kamolrat and Hossain, Md A and Maude, Richard J and Jaeger,
                    Stefan and Thoma, George R},
            journal={PeerJ},
            volume={6},
            pages={e4568},
            year={2018},
            doi={10.7717/peerj.4568}
        }""",
        assets={
            "images": DownloadInfo(
                url="https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
                checksum="sha256:0a949556b2414159b5100192609805376654c4266d8d187be9b1922fad43c668",
            ),
            "patient_mapping_parasitized": DownloadInfo(
                url="https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_parasitized.csv",
                checksum="sha256:d0367e513397404e980baee2a641bce9ce329a22e62ea9007962dfca2f8418d3",
            ),
            "patient_mapping_uninfected": DownloadInfo(
                url="https://data.lhncbc.nlm.nih.gov/public/Malaria/patientid_cellmapping_uninfected.csv",
                checksum="sha256:a8577b21e7154724f4bbd18326218e2c63a99b22ab372b79a7530d36df6dab78",
            ),
        },
    )

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "NIH Malaria Cell Images: 27,558 RGB red blood cell crops, balanced between "
                "parasitized and uninfected cells. Only train is provided; patient identifiers "
                "are included for constructing patient-disjoint splits."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                    "filename": Value("string"),
                    "patient_id": Value("string"),
                    "source_image_id": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source.homepage,
            citation=source.citation,
        )

    def _candidate_splits(self):
        # The mapping assets are metadata, not additional dataset splits.
        return [Split.TRAIN]

    def _split_generators(self):
        assets = self._source().assets
        asset_keys = list(assets)
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        paths = bulk_download(
            [assets[key] for key in asset_keys], dest_folder=download_dir
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": dict(zip(asset_keys, paths)), "split": "train"},
            )
        ]

    @staticmethod
    def _read_patient_mapping(path):
        """Read NIH's headerless CSV of patient IDs and padded filename lists."""
        mapping = {}
        with open(path, encoding="utf-8-sig", newline="") as file:
            for row_number, row in enumerate(csv.reader(file), start=1):
                if not any(field.strip() for field in row):
                    continue
                patient_id = row[0].strip()
                # In NIH's CSV the Python list spans comma-separated cells;
                # trailing empty cells pad each row to the same width.
                filenames_text = ",".join(
                    field.strip() for field in row[1:] if field.strip()
                )
                try:
                    filenames = ast.literal_eval(filenames_text)
                except (SyntaxError, ValueError) as error:
                    raise ValueError(
                        f"Invalid filename list in {path}, row {row_number}"
                    ) from error
                if not patient_id or not isinstance(filenames, list):
                    raise ValueError(
                        f"Invalid patient mapping in {path}, row {row_number}"
                    )
                for filename in filenames:
                    if not isinstance(filename, str) or not filename:
                        raise ValueError(
                            f"Invalid cell filename in {path}, row {row_number}"
                        )
                    if filename in mapping and mapping[filename] != patient_id:
                        raise ValueError(
                            f"Conflicting patient mappings for {filename} in {path}"
                        )
                    mapping[filename] = patient_id
        return mapping

    def _generate_examples(self, path_map: dict | None = None, _split=None, **_kwargs):
        if path_map is None:
            raise TypeError("path_map is required")
        paths: dict = path_map
        patient_mappings = {
            label: self._read_patient_mapping(paths[f"patient_mapping_{label}"])
            for label in self._labels()
        }
        class_directories = {"Parasitized": "parasitized", "Uninfected": "uninfected"}

        with ZipFile(paths["images"]) as archive:
            for member in sorted(archive.infolist(), key=lambda entry: entry.filename):
                parts = member.filename.split("/")
                if (
                    member.is_dir()
                    or len(parts) != 3
                    or parts[0] != "cell_images"
                    or parts[1] not in class_directories
                    or not parts[2].endswith(".png")
                ):
                    continue

                filename = parts[2]
                label = class_directories[parts[1]]
                match = re.fullmatch(r"(.+)_cell_\d+\.png", filename)
                if match is None:
                    raise ValueError(
                        f"Cannot identify source image for {member.filename}"
                    )
                if filename not in patient_mappings[label]:
                    raise ValueError(f"Missing patient mapping for {member.filename}")

                try:
                    with PILImage.open(
                        io.BytesIO(archive.read(member))
                    ) as source_image:
                        image = source_image.convert("RGB")
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Cannot decode Malaria image {member.filename}"
                    ) from error

                yield (
                    member.filename,
                    {
                        "image": image,
                        "label": label,
                        "filename": filename,
                        "patient_id": patient_mappings[label][filename],
                        "source_image_id": match.group(1),
                    },
                )

    @staticmethod
    def _labels():
        return ["parasitized", "uninfected"]
