import csv
import io
import json
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
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download

from ._audio_utils import resample, sample_rate_configs, wav_bytes_to_mono
from ._freesound_vocabularies import FSD50K_VOCABULARY
from ._zip_utils import open_zip


_ZENODO = "https://zenodo.org/records/4060432/files/{}?download=1"
NATIVE_SAMPLE_RATE = 44100

# Split archives: parts in disk order (``.z01, .z02, ..., .zip``) with their Zenodo md5 checksums.
_DEV_AUDIO = (
    ("FSD50K.dev_audio.z01", "faa7cf4cc076fc34a44a479a5ed862a3"),
    ("FSD50K.dev_audio.z02", "8f9b66153e68571164fb1315d00bc7bc"),
    ("FSD50K.dev_audio.z03", "1196ef47d267a993d30fa98af54b7159"),
    ("FSD50K.dev_audio.z04", "d088ac4e11ba53daf9f7574c11cccac9"),
    ("FSD50K.dev_audio.z05", "81356521aa159accd3c35de22da28c7f"),
    ("FSD50K.dev_audio.zip", "c480d119b8f7a7e32fdb58f3ea4d6c5a"),
)
_EVAL_AUDIO = (
    ("FSD50K.eval_audio.z01", "3090670eaeecc013ca1ff84fe4442aeb"),
    ("FSD50K.eval_audio.zip", "6fa47636c3a3ad5c7dfeba99f2637982"),
)
_GROUND_TRUTH = ("FSD50K.ground_truth.zip", "ca27382c195e37d2269c4c866dd73485")
_METADATA = ("FSD50K.metadata.zip", "b9ea0c829a411c1d42adb9da539ed237")


def _asset(filename: str, md5: str) -> DownloadInfo:
    return DownloadInfo(
        url=_ZENODO.format(filename),
        fallbacks=[f"https://zenodo.org/record/4060432/files/{filename}?download=1"],
        checksum=f"md5:{md5}",
        filename=filename,
    )


class FSD50K(BaseDatasetBuilder):
    """FSD50K: an open dataset of human-labeled sound events.

    51,197 Freesound clips (108 h, 0.3-30 s, mono) multi-labelled with 200 classes of the AudioSet
    ontology (144 leaves, 56 intermediate nodes). Labels are *smeared* as released: each clip also
    carries every ancestor of its labels up to the ontology root. Splits follow the paper:
    ``train`` (36,796) and ``validation`` (4,170) come from the dev set's official ``split`` column,
    ``test`` is the exhaustively labelled eval set (10,231). Dev and eval share no Freesound uploader.

    Each row holds the waveform as a float32 ``series`` of shape ``[T, 1]`` (like the other audio
    builders) on the raw 16-bit PCM scale (divide by 32768 for ``[-1, 1]``) with its ``sample_rate``,
    the multi-hot target as a list of class ids (``labels``) plus AudioSet MIDs (``mids``), and
    per-clip ``license``/``uploader`` for attribution.
    Configs: ``"native"`` (44.1 kHz, default, ~75 GB processed cache), ``"22khz"`` and ``"16khz"``
    (resampled once at build time, ~53 GB / ~39 GB).

    The audio ships as split ZIP archives (``.z01`` ... ``.zip``); they are read in place by
    :class:`~stable_datasets.timeseries._zip_utils.SplitZipFile`, without merging or extracting.

    The dataset is CC BY 4.0; clips keep their Freesound licenses (CC0, CC-BY, CC-BY-NC, Sampling+).
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = sample_rate_configs(NATIVE_SAMPLE_RATE)
    DEFAULT_CONFIG_NAME = "native"

    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/4060432",
        assets={
            **{name: _asset(name, md5) for name, md5 in _DEV_AUDIO + _EVAL_AUDIO},
            "ground_truth": _asset(*_GROUND_TRUTH),
            "metadata": _asset(*_METADATA),
        },
        license="CC BY 4.0 (dataset); individual clips under their Freesound CC licenses",
        citation="""@article{fonseca2022fsd50k,
                         title={{FSD50K}: An Open Dataset of Human-Labeled Sound Events},
                         author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
                         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
                         volume={30},
                         pages={829--852},
                         year={2022}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="FSD50K multi-label sound event recognition (200 AudioSet-ontology classes).",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "sample_rate": Value("int32"),
                    "labels": Sequence(ClassLabel(names=[name for name, _ in FSD50K_VOCABULARY])),
                    "mids": Sequence(Value("string")),
                    "filename": Value("string"),
                    "freesound_id": Value("int64"),
                    "license": Value("string"),
                    "uploader": Value("string"),
                }
            ),
            supervised_keys=("series", "labels"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self):
        assets = self._source()["assets"]
        names = [name for name, _ in _DEV_AUDIO + _EVAL_AUDIO] + ["ground_truth", "metadata"]
        paths = dict(
            zip(
                names,
                bulk_download(
                    [self._normalize_download_info(assets[n], asset_name=n) for n in names],
                    dest_folder=self._raw_download_dir,
                ),
            )
        )
        # Downloaded files are renamed with a URL hash, so keep the parts in the declared order.
        dev_parts = [paths[name] for name, _ in _DEV_AUDIO]
        eval_parts = [paths[name] for name, _ in _EVAL_AUDIO]
        common = {"ground_truth_path": paths["ground_truth"], "metadata_path": paths["metadata"]}
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"audio_parts": dev_parts, "split": "train", **common}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"audio_parts": dev_parts, "split": "val", **common}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"audio_parts": eval_parts, "split": "eval", **common}),
        ]

    def _generate_examples(self, audio_parts, ground_truth_path, metadata_path, split):
        """Yield clips of one split.

        Args:
            audio_parts: Ordered paths of the dev or eval split archive.
            ground_truth_path: ``FSD50K.ground_truth.zip``.
            metadata_path: ``FSD50K.metadata.zip`` (per-clip license and uploader).
            split: ``"train"`` or ``"val"`` (rows of ``dev.csv``) or ``"eval"``.
        """
        source_set = "eval" if split == "eval" else "dev"
        rows = _load_ground_truth(ground_truth_path, source_set, split)
        clip_info = _load_clip_info(metadata_path, source_set)
        label_ids = {name: idx for idx, (name, _) in enumerate(FSD50K_VOCABULARY)}
        target_rate = self.config.sample_rate

        with open_zip(audio_parts) as archive:
            members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist() if name.endswith(".wav")}
            missing = [row["fname"] for row in rows if f"{row['fname']}.wav" not in members]
            if missing:
                raise ValueError(
                    f"{len(missing)} {split} clips listed in the ground truth are missing from the archive."
                )

            for row in rows:
                fname = row["fname"]
                labels, mids = row["labels"].split(","), row["mids"].split(",")
                unknown = [label for label in labels if label not in label_ids]
                if unknown or len(labels) != len(mids):
                    raise ValueError(f"Malformed FSD50K ground truth for clip {fname}: {row['labels']!r}")
                series, rate = wav_bytes_to_mono(archive.read(members[f"{fname}.wav"]))
                if target_rate is not None:
                    series, rate = resample(series, rate, target_rate), target_rate
                info = clip_info.get(fname, {})
                yield (
                    fname,
                    {
                        "series": series[:, None],
                        "sample_rate": rate,
                        "labels": [label_ids[label] for label in labels],
                        "mids": mids,
                        "filename": f"{fname}.wav",
                        "freesound_id": int(fname),
                        "license": info.get("license", ""),
                        "uploader": info.get("uploader", ""),
                    },
                )


def _load_ground_truth(ground_truth_path, source_set: str, split: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(ground_truth_path) as archive:
        with archive.open("FSD50K.ground_truth/vocabulary.csv") as fh:
            vocabulary = [(name, mid) for _, name, mid in csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))]
        if tuple(vocabulary) != FSD50K_VOCABULARY:
            raise ValueError("FSD50K vocabulary.csv differs from the vocabulary this builder was written for.")
        with archive.open(f"FSD50K.ground_truth/{source_set}.csv") as fh:
            rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
    if source_set == "dev":
        rows = [row for row in rows if row["split"] == split]
    return rows


def _load_clip_info(metadata_path, source_set: str) -> dict[str, dict]:
    with zipfile.ZipFile(metadata_path) as archive:
        return json.loads(archive.read(f"FSD50K.metadata/{source_set}_clips_info_FSD50K.json"))
