"""Original Spatial Envelope eight-category outdoor scene collection."""

import zipfile
from pathlib import Path, PurePosixPath

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Value,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, download


class Scene8(BaseDatasetBuilder):
    """Load all 2,688 original images as ``train``, without inventing a split.

    ``archive_path`` optionally supplies an already downloaded original ZIP.
    Images, including duplicates and conflicting labels, are preserved verbatim.
    Experimental partitions belong to the evaluation protocol, not this builder.
    """

    VERSION = Version("1.0.0")
    CLASS_NAMES = ("coast", "forest", "highway", "insidecity", "mountain", "opencountry", "street", "tallbuilding")
    SOURCE = DatasetSource(
        homepage="https://people.csail.mit.edu/torralba/code/spatialenvelope/",
        citation="""@article{oliva2001modeling,
            title={Modeling the shape of the scene: A holistic representation of the spatial envelope},
            author={Oliva, Aude and Torralba, Antonio},
            journal={International Journal of Computer Vision}, volume={42}, pages={145--175}, year={2001}}
        """,
        assets={
            "train": DownloadInfo(
                url="https://people.csail.mit.edu/torralba/code/spatialenvelope/"
                "spatial_envelope_256x256_static_8outdoorcategories.zip",
                checksum="sha256:e6157352dd9aeb055519be6afe1744bd20be9b2270bcb987f9dd06afb787f460",
            ),
        },
        license="Not specified on the original dataset page; consult the publisher.",
    )

    def __init__(self, archive_path=None, **kwargs):
        self.archive_path = Path(archive_path) if archive_path is not None else None
        super().__init__(**kwargs)

    def _info(self):
        return DatasetInfo(
            description="Spatial Envelope: 2,688 RGB outdoor scene images in eight classes. No official partition.",
            features=Features(
                {"image": Image(), "label": ClassLabel(names=list(self.CLASS_NAMES)), "image_id": Value("string")}
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _split_generators(self):
        path = self.archive_path
        if path is None:
            path = download(self.SOURCE.assets["train"], dest_folder=self._raw_download_dir)
        return [SplitGenerator(Split.TRAIN, {"data_path": path, "split": Split.TRAIN})]

    def _generate_examples(self, data_path, split):
        if split != Split.TRAIN:
            raise ValueError("Scene8 only exposes the original unsplit collection as 'train'.")
        with zipfile.ZipFile(data_path) as archive:
            images = {}
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if member.is_dir() or "__MACOSX" in path.parts or path.suffix.lower() != ".jpg":
                    continue
                image_id = path.stem
                category = image_id.split("_", 1)[0]
                if "_" not in image_id or category not in self.CLASS_NAMES:
                    raise ValueError(f"Unknown Scene8 category or malformed image name: {member.filename}")
                if image_id in images:
                    raise ValueError(f"Duplicate Scene8 image ID: {image_id}")
                images[image_id] = (member, self.CLASS_NAMES.index(category))
            if not images:
                raise ValueError("Scene8 archive contains no JPEG images.")
            for image_id, (member, label) in sorted(images.items()):
                yield image_id, {"image": archive.read(member), "label": label, "image_id": image_id}
