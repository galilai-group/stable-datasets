import io
import re
import zipfile
from pathlib import PurePosixPath

import numpy as np
from scipy.io.wavfile import read as wav_read

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
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_SOURCE_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
_ARCHIVE_SHA256 = "beaca4fb849cad13103b2cffba913b3d7fdb5f15222da4ddd07187b3c38a757a"
_FILENAME_RE = re.compile(r"(?P<digit>[^_]+)_(?P<speaker>[^_]+)_(?P<index>[^_]+)\.wav", re.IGNORECASE)
_EXPECTED_SAMPLE_RATE = 8000


class FSDD(BaseDatasetBuilder):
    """Free Spoken Digit Dataset with its official train/test split."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://github.com/Jakobovski/free-spoken-digit-dataset",
        assets={
            "archive": DownloadInfo(
                url=(f"https://github.com/Jakobovski/free-spoken-digit-dataset/archive/{_SOURCE_REVISION}.zip"),
                fallbacks=[f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{_SOURCE_REVISION}"],
                checksum=f"sha256:{_ARCHIVE_SHA256}",
                filename=f"free-spoken-digit-dataset-{_SOURCE_REVISION}.zip",
            )
        },
        citation="""@misc{jackson2016fsdd,
                         author = {Jackson, Zohar},
                         title = {Free Spoken Digit Dataset},
                         year = {2016},
                         url = {https://github.com/Jakobovski/free-spoken-digit-dataset}}""",
        license="CC BY-SA 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Free Spoken Digit Dataset recordings with the official index-based "
                "train/test split and speaker metadata."
            ),
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(num_classes=10),
                    "speaker_id": Value("string"),
                    "utterance_id": Value("int32"),
                    "filename": Value("string"),
                    "sample_rate": Value("int32"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        source = self._source()
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            raise RuntimeError("Expected _raw_download_dir to be set before split generation.")

        (archive_path,) = bulk_download(
            [self._normalize_download_info(source["assets"]["archive"], asset_name="archive")],
            dest_folder=download_dir,
        )
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_path": archive_path, "split": Split.TRAIN},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_path": archive_path, "split": Split.TEST},
            ),
        ]

    def _generate_examples(self, data_path, split):
        if split not in (Split.TRAIN, Split.TEST):
            raise ValueError(f"Unknown FSDD split {split!r}; expected 'train' or 'test'.")

        recording_count = 0
        selected_count = 0
        seen_filenames = set()

        try:
            with zipfile.ZipFile(data_path) as archive:
                members = sorted(archive.infolist(), key=lambda member: member.filename)
                for member in members:
                    path = PurePosixPath(member.filename)
                    if member.is_dir() or path.parent.name != "recordings" or path.suffix.lower() != ".wav":
                        continue

                    recording_count += 1
                    filename = path.name
                    if filename in seen_filenames:
                        raise ValueError(f"Duplicate FSDD recording filename in archive: {filename!r}.")
                    seen_filenames.add(filename)

                    digit, speaker, utterance_id = _parse_filename(filename)
                    sample_split = Split.TEST if utterance_id <= 4 else Split.TRAIN
                    if sample_split != split:
                        continue

                    series, sample_rate = _decode_wav(archive.read(member), filename)
                    selected_count += 1
                    yield (
                        filename,
                        {
                            "series": series,
                            "label": digit,
                            "speaker_id": speaker,
                            "utterance_id": utterance_id,
                            "filename": filename,
                            "sample_rate": sample_rate,
                        },
                    )
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid FSDD ZIP archive: {data_path}") from error

        if recording_count == 0:
            raise ValueError("FSDD archive contains no WAV files directly under a recordings/ directory.")
        if selected_count == 0:
            raise ValueError(f"FSDD archive contains no recordings for the {split!r} split.")


def _parse_filename(filename: str) -> tuple[int, str, int]:
    match = _FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"Invalid FSDD recording filename {filename!r}; expected '{{digit}}_{{speaker}}_{{index}}.wav'."
        )

    digit_text = match.group("digit")
    speaker = match.group("speaker")
    index_text = match.group("index")
    if len(digit_text) != 1 or not digit_text.isascii() or not digit_text.isdecimal():
        raise ValueError(f"Invalid FSDD digit in filename {filename!r}; expected an integer from 0 to 9.")
    digit = int(digit_text)
    if not 0 <= digit <= 9:
        raise ValueError(f"Invalid FSDD digit in filename {filename!r}; expected an integer from 0 to 9.")
    if not speaker.strip() or speaker != speaker.strip():
        raise ValueError(
            f"Invalid FSDD speaker in filename {filename!r}; speaker must be a non-empty string without outer whitespace."
        )
    if not index_text.isascii() or not index_text.isdecimal():
        raise ValueError(f"Invalid FSDD recording index in {filename!r}; expected an integer from 0 to 49.")
    utterance_id = int(index_text)
    if not 0 <= utterance_id <= 49:
        raise ValueError(f"Invalid FSDD recording index in {filename!r}; expected an integer from 0 to 49.")
    return digit, speaker, utterance_id


def _decode_wav(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    try:
        sample_rate, wav = wav_read(io.BytesIO(data))
    except Exception as error:
        raise ValueError(f"Could not decode FSDD WAV file {filename!r}: {error}") from error

    wav = np.asarray(wav)
    if sample_rate != _EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Invalid sample rate for FSDD recording {filename!r}: "
            f"expected {_EXPECTED_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if wav.ndim != 1:
        channels = wav.shape[1] if wav.ndim == 2 else "unknown"
        raise ValueError(f"FSDD recording {filename!r} must be mono; decoded {channels} channels.")
    if wav.size == 0:
        raise ValueError(f"FSDD recording {filename!r} is empty.")

    series = np.asarray(wav, dtype=np.float32)
    if not np.isfinite(series).all():
        raise ValueError(f"FSDD recording {filename!r} contains non-finite samples.")
    if wav.dtype.kind != "i" or wav.dtype.itemsize != 2:
        raise ValueError(f"FSDD recording {filename!r} must contain 16-bit PCM samples; decoded dtype {wav.dtype}.")

    # Preserve signed 16-bit PCM sample values; conversion to float32 does not normalize amplitudes.
    return series[:, None], int(sample_rate)
