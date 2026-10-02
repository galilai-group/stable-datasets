"""Sewer-ML multi-label sewer defect classification dataset."""

import csv
import zipfile
from pathlib import Path

from loguru import logger as logging

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Sequence,
    Value,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download

from ._anomaly_utils import ANOMALY_LABEL_NAMES


SEWER_ML_VERSION = Version("1.0.0")

# The 17 defect codes, in the column order used by the annotation CSVs.
SEWER_ML_DEFECT_CLASSES = [
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
]

_BASE_URL = "https://sciencedata.dk/public/Large%20AAU%20files/Sewer_ML"

# Split -> (annotation CSV, image archives). Archives are ~19 GB each.
_SPLIT_LAYOUT = {
    "train": ("SewerML_Train.csv", [f"train{i:02d}.zip" for i in range(14)]),
    "val": ("SewerML_Val.csv", [f"valid{i:02d}.zip" for i in range(2)]),
    "test": ("SewerML_Test.csv", [f"test{i:02d}.zip" for i in range(2)]),
}

_AUTH_INSTRUCTIONS = (
    "Sewer-ML is hosted on sciencedata.dk as a password-protected share. Request the "
    "download password with the Google form linked from https://vap.aau.dk/sewer-ml/ "
    "(https://forms.gle/hBaPtoweZumZAi4u9).\n\n"
    "There is no username -- the share checks the password only -- so add a netrc entry "
    "with just the password. stable-datasets needs no extra configuration: requests "
    "reads ~/.netrc natively.\n\n"
    "    machine sciencedata.dk\n"
    "    password <the-download-password>\n\n"
    "then chmod 600 ~/.netrc. A 'login' line is accepted but ignored by the server.\n\n"
    "Alternatively pass an already-downloaded Sewer-ML directory with data_dir=."
)


def _is_forbidden(error: BaseException) -> bool:
    """Return True when *error*, or anything it was raised from, is an HTTP 403.

    ``download`` re-raises the underlying ``HTTPError`` as the ``__cause__`` of a
    generic ``RuntimeError``, and ``bulk_download`` ships that across a process
    boundary, so the status code is only reachable through the chain.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error)
        if "403" in message or "Forbidden" in message:
            return True
        error = error.__cause__ or error.__context__
    return False


class SewerML(BaseDatasetBuilder):
    """Sewer-ML: 1.3 million sewer pipe inspection images with 17 defect labels.

    Every image carries a multi-hot vector over 17 defect codes plus the binary
    ``Defect`` flag, used here as the anomaly label. The official test split is
    **unlabeled** -- its annotation CSV lists filenames only -- so its ``label``,
    ``defects`` and ``water_level`` fields are ``None``.

    .. warning::

       The archives total roughly 340 GB (14 train + 2 val + 2 test, ~19 GB each)
       and the host ignores HTTP ``Range``, so an interrupted archive restarts from
       zero. Splits are fetched one at a time so a finished split's cache is
       committed before the next archive starts. Point ``download_dir=`` and
       ``processed_cache_dir=`` at a volume with room before loading.

    Loading one split downloads only that split's annotation CSV and archives, so
    ``SewerML(split="validation")`` (2 archives, ~33 GB) is the cheapest way to
    check that the password and host work before committing to the full download.
    Split names are ``"train"``, ``"validation"`` and ``"test"``; ``split=None``
    builds all three.

    Access requires the share password (no username); a refused download raises
    ``PermissionError`` carrying the setup steps. Images are read straight out of
    the zip archives, so no extracted copy is written to disk.
    """

    VERSION = SEWER_ML_VERSION

    SOURCE = DatasetSource(
        homepage="https://vap.aau.dk/sewer-ml/",
        assets={
            name: DownloadInfo(url=f"{_BASE_URL}/{name}")
            for csv_name, archives in _SPLIT_LAYOUT.values()
            for name in (csv_name, *archives)
        },
        license=(
            "CC BY-NC-SA 4.0. Non-commercial research use; the download password is "
            "requested via the form linked from https://vap.aau.dk/sewer-ml/."
        ),
        citation="""@inproceedings{haurum2021sewer,
  title={Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark},
  author={Haurum, Joakim Bruslund and Moeslund, Thomas B},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13456--13467},
  year={2021}
}""",
    )

    def __init__(self, config_name: str | None = None, data_dir: str | Path | None = None, **kwargs):
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        super().__init__(config_name=config_name, **kwargs)

    def _info(self):
        return DatasetInfo(
            description=(
                "Sewer-ML contains 1.3 million images from 75,618 sewer pipe inspection videos "
                "annotated by professional sewer inspectors across 17 defect classes. Train and "
                "validation splits are fully labeled; the benchmark's test split is unlabeled."
            ),
            features=Features(
                {
                    "image": Image(),
                    # Binary "Defect" flag; None on the unlabeled test split.
                    "label": ClassLabel(names=ANOMALY_LABEL_NAMES),
                    # Multi-hot over SEWER_ML_DEFECT_CLASSES; None on the test split.
                    "defects": Sequence(Value("int8")),
                    "water_level": Value("int8"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self):
        # SOURCE assets are named by file, not by split, so the default
        # asset-key-is-split-name inference does not apply here.
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        # Downloads happen inside _generate_examples so each split is fetched only
        # when it is actually built.
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"split": "train"}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"split": "val"}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"split": "test"}),
        ]

    def _fetch(self, names: list[str]) -> list[Path]:
        """Download *names*, or resolve them under ``data_dir`` when one was given."""
        if self.data_dir is not None:
            paths = []
            for name in names:
                # Accept both a flat directory and the zips/ layout the upstream
                # download script produces.
                candidates = [self.data_dir / name, self.data_dir / "zips" / name]
                match = next((c for c in candidates if c.is_file()), None)
                if match is None:
                    raise FileNotFoundError(f"Expected {name} under {self.data_dir}.\n\n{_AUTH_INSTRUCTIONS}")
                paths.append(match)
            return paths

        assets = self._source()["assets"]
        download_dir = Path(getattr(self, "_raw_download_dir", None) or _default_dest_folder())
        try:
            return bulk_download([assets[name] for name in names], dest_folder=download_dir)
        except Exception as error:
            # The host answers unauthenticated requests with a bare 403 and no
            # WWW-Authenticate header, so surface the setup steps instead of a
            # generic "failed to download from all candidate URLs". download()
            # wraps the HTTPError as the __cause__ of that RuntimeError, so the
            # status only shows up by walking the chain.
            if _is_forbidden(error):
                raise PermissionError(f"Sewer-ML download was refused (HTTP 403).\n\n{_AUTH_INSTRUCTIONS}") from error
            raise

    def _generate_examples(self, split):
        csv_name, archive_names = _SPLIT_LAYOUT[split]

        (annotation_path,) = self._fetch([csv_name])
        with open(annotation_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Sewer-ML annotation file {annotation_path} is empty.")

        # The benchmark withholds test labels: that CSV carries only "Filename".
        labeled = "Defect" in rows[0]
        if not labeled:
            logging.info(f"Sewer-ML {split} split is unlabeled; emitting label/defects/water_level as None.")

        archive_paths = self._fetch(archive_names)

        # Index every member by bare filename so images can be streamed straight
        # out of the archives instead of extracting ~340 GB to disk.
        members: dict[str, tuple[int, str]] = {}
        archives: list[zipfile.ZipFile] = []
        try:
            for index, archive_path in enumerate(archive_paths):
                archive = zipfile.ZipFile(archive_path)
                archives.append(archive)
                for info in archive.infolist():
                    if not info.is_dir():
                        members.setdefault(Path(info.filename).name, (index, info.filename))

            missing = 0
            for row in rows:
                filename = row["Filename"]
                located = members.get(filename)
                if located is None:
                    missing += 1
                    continue

                archive_index, member_name = located
                with archives[archive_index].open(member_name) as handle:
                    image_bytes = handle.read()

                yield (
                    filename,
                    {
                        "image": image_bytes,
                        "label": int(row["Defect"]) if labeled else None,
                        "defects": [int(row[code]) for code in SEWER_ML_DEFECT_CLASSES] if labeled else None,
                        "water_level": int(row["WaterLevel"]) if labeled and row.get("WaterLevel") else None,
                        "filename": filename,
                    },
                )

            if missing:
                logging.warning(f"Sewer-ML {split}: {missing} annotated filenames were absent from the archives.")
        finally:
            for archive in archives:
                archive.close()
