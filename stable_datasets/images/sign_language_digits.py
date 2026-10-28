import io
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class SignLanguageDigits(BaseDatasetBuilder):
    """American Sign Language digit images collected from 218 students."""

    VERSION = Version("1.0.0")
    SOURCE_REVISION = "4b09433ef9d5c60602e8805bb54b8aa2428527f8"
    SOURCE_ARCHIVE_SHA256 = "b002e3b3819b5ec270d3d424ef8c7d9b66372cb6b8f92fbfe3a258abb3b3ccb0"

    SOURCE = DatasetSource(
        homepage="https://github.com/ardamavi/Sign-Language-Digits-Dataset",
        assets={
            "all": DownloadInfo(
                url=(f"https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/{SOURCE_REVISION}.zip"),
                filename=f"sign-language-digits-{SOURCE_REVISION}.zip",
                checksum=f"sha256:{SOURCE_ARCHIVE_SHA256}",
            )
        },
        citation="""@article{mavi2020sign,
                     title={A New Dataset and Proposed Convolutional Neural Network Architecture for Classification of American Sign Language Digits},
                     author={Mavi, Arda},
                     journal={arXiv preprint arXiv:2011.08927},
                     year={2020}}""",
        license="Apache-2.0",
    )

    _IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

    def _info(self):
        source = self._source()
        return DatasetInfo(
            description=(
                "Sign Language Digits contains 2,062 RGB hand-sign images representing the digits 0 through 9. "
                "The source publishes one image pool and does not define train, validation, or test splits."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=source["homepage"],
            citation=source["citation"],
            license=source["license"],
        )

    def _generate_examples(self, data_path, split):
        """Yield images located directly below ``Dataset/<digit>/`` in the ZIP."""
        if split != "all":
            raise ValueError(f"Unknown split {split!r}; SignLanguageDigits only provides the 'all' split.")

        data_path = Path(data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"Sign Language Digits archive not found: {data_path}")

        try:
            with zipfile.ZipFile(data_path) as archive:
                members = self._dataset_members(archive)
                if not members:
                    raise ValueError(
                        "No images found directly under Dataset/<digit>/ in the Sign Language Digits archive."
                    )

                for filename, label, member in members:
                    try:
                        image_bytes = archive.read(member)
                        with PILImage.open(io.BytesIO(image_bytes)) as decoded:
                            decoded.load()
                            image = decoded.convert("RGB")
                    except (OSError, ValueError, zipfile.BadZipFile) as error:
                        raise ValueError(f"Failed to decode image {member.filename!r}: {error}") from error

                    yield filename, {"image": image, "label": label, "filename": filename}
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid Sign Language Digits ZIP archive: {data_path}") from error

    @classmethod
    def _dataset_members(cls, archive):
        members = []
        seen_filenames = set()

        for member in archive.infolist():
            if member.is_dir():
                continue

            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            dataset_positions = [index for index, part in enumerate(parts) if part == "Dataset"]
            if len(dataset_positions) != 1:
                continue

            dataset_index = dataset_positions[0]
            if len(parts) != dataset_index + 3:
                continue

            label_name, basename = parts[-2:]
            if label_name not in cls._labels():
                continue
            if PurePosixPath(basename).suffix.lower() not in cls._IMAGE_EXTENSIONS:
                continue

            filename = f"{label_name}/{basename}"
            if filename in seen_filenames:
                raise ValueError(f"Duplicate dataset filename in archive: {filename!r}")
            seen_filenames.add(filename)
            members.append((filename, int(label_name), member))

        return sorted(members, key=lambda item: item[0])

    @staticmethod
    def _labels():
        return [str(digit) for digit in range(10)]
