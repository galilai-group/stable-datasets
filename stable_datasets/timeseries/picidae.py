import zipfile

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Sequence,
    Value,
    Version,
)
from stable_datasets.utils import BaseDatasetBuilder

from ._audio_utils import wav_bytes_to_series


PICIDAE_LABELS = [
    "BackgroundNoise",
    "DendrocoposLeucotos-call",
    "DendrocoposLeucotos-drumming",
    "DendrocoposMajor-call",
    "DendrocoposMajor-drumming",
    "DendrocoposMedius-call",
    "DendrocoposMedius-song",
    "DendrocoposMinor-call",
    "DendrocoposMinor-drumming",
    "DryocopusMartius-call",
    "DryocopusMartius-drumming",
    "JynxTorquilla-song",
    "PicusViridis-song",
]


class Picidae(BaseDatasetBuilder):
    """Picidae birdsong classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/574438",
        assets={
            "train": DownloadInfo(
                url="https://zenodo.org/record/574438/files/PicidaeDataset.zip?download=1",
                fallbacks=["https://zenodo.org/records/574438/files/PicidaeDataset.zip"],
                filename="PicidaeDataset.zip",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Picidae birdsong classification dataset with species/song-type labels.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=PICIDAE_LABELS),
                    "label_name": Value("string"),
                    "xc_identifier": Value("string"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with zipfile.ZipFile(data_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".wav") or "/._" in name or name.rsplit("/", 1)[-1].startswith("._"):
                    continue
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                label_name = parts[-2]
                if label_name not in PICIDAE_LABELS:
                    continue
                filename = parts[-1]
                xc_identifier = filename.split("-")[0]
                yield (
                    name,
                    {
                        "series": wav_bytes_to_series(archive.read(name)),
                        "label": PICIDAE_LABELS.index(label_name),
                        "label_name": label_name,
                        "xc_identifier": xc_identifier,
                        "filename": filename,
                    },
                )
