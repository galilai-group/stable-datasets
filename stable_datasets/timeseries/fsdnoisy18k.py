import csv
import io
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
from ._freesound_vocabularies import FSDNOISY18K_VOCABULARY


_ZENODO = "https://zenodo.org/records/2529934/files/{}?download=1"
NATIVE_SAMPLE_RATE = 44100


def _asset(filename: str, md5: str) -> DownloadInfo:
    return DownloadInfo(
        url=_ZENODO.format(filename),
        fallbacks=[f"https://zenodo.org/record/2529934/files/{filename}?download=1"],
        checksum=f"md5:{md5}",
        filename=filename,
    )


class FSDnoisy18k(BaseDatasetBuilder):
    """FSDnoisy18k: sound event classification with real-world label noise.

    18,532 Freesound clips (42.5 h, 0.3-30 s, mono) labelled with one of 20 AudioSet-ontology
    classes. The train set (17,585 clips) has a small *clean* subset whose labels were manually
    verified (``manually_verified=True``, 1,772 clips) and a large *noisy* subset labelled from
    Freesound tags only; ``noisy_small`` marks a noisy subset of similar duration to the clean one
    (1,003 clips). The test set (947 clips) is drawn entirely from the clean portion.

    Each row holds the waveform as a float32 ``series`` of shape ``[T, 1]`` (like the other audio
    builders) on the raw 16-bit PCM scale (divide by 32768 for ``[-1, 1]``) together with its
    ``sample_rate``. Configs: ``"native"`` (44.1 kHz, default, ~30 GB processed cache), ``"22khz"``
    and ``"16khz"`` (resampled once at build time, ~21 GB / ~15 GB).

    The dataset is CC BY 4.0; individual clips keep their Freesound licenses (``license`` column).
    """

    VERSION = Version("1.0.0")
    BUILDER_CONFIGS = sample_rate_configs(NATIVE_SAMPLE_RATE)
    DEFAULT_CONFIG_NAME = "native"

    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/2529934",
        assets={
            "train_audio": _asset("FSDnoisy18k.audio_train.zip", "34dc1d34ca44622af5bf439ceb6f0d55"),
            "test_audio": _asset("FSDnoisy18k.audio_test.zip", "1ac73d70b4ef3f81900d98c261a832de"),
            "meta": _asset("FSDnoisy18k.meta.zip", "96e27a4a63b7a2870522ddcedb5d8296"),
            "doc": _asset("FSDnoisy18k.doc.zip", "093a1ca185ec341ca4eac14215e7f96b"),
        },
        license="CC BY 4.0 (dataset); individual clips under their Freesound CC licenses",
        citation="""@inproceedings{fonseca2019learning,
                         title={Learning Sound Event Classifiers from Web Audio with Noisy Labels},
                         author={Fonseca, Eduardo and Plakal, Manoj and Ellis, Daniel P. W. and Font, Frederic and Favory, Xavier and Serra, Xavier},
                         booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
                         year={2019}}""",
    )

    def _info(self):
        return DatasetInfo(
            description="FSDnoisy18k sound event classification with clean and noisy train labels.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "sample_rate": Value("int32"),
                    "label": ClassLabel(names=[name for name, _ in FSDNOISY18K_VOCABULARY]),
                    "filename": Value("string"),
                    "freesound_id": Value("int64"),
                    "manually_verified": Value("bool"),
                    "noisy_small": Value("bool"),
                    "license": Value("string"),
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
        assets = self._source()["assets"]
        names = ["train_audio", "test_audio", "meta", "doc"]
        train_audio, test_audio, meta, doc = bulk_download(
            [self._normalize_download_info(assets[n], asset_name=n) for n in names],
            dest_folder=self._raw_download_dir,
        )
        common = {"meta_path": meta, "doc_path": doc}
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"audio_path": train_audio, "split": "train", **common}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"audio_path": test_audio, "split": "test", **common}),
        ]

    def _generate_examples(self, audio_path, meta_path, doc_path, split):
        rows = _load_ground_truth(meta_path, split)
        licenses = _load_clip_licenses(doc_path)
        label_ids = {name: idx for idx, (name, _) in enumerate(FSDNOISY18K_VOCABULARY)}
        target_rate = self.config.sample_rate

        with zipfile.ZipFile(audio_path) as archive:
            members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist() if name.endswith(".wav")}
            missing = [row["fname"] for row in rows if row["fname"] not in members]
            if missing:
                raise ValueError(f"{len(missing)} {split} clips listed in the metadata are missing from the archive.")

            for row in rows:
                fname = row["fname"]
                if row["label"] not in label_ids:
                    raise ValueError(f"Unknown FSDnoisy18k label {row['label']!r} for {fname}.")
                series, rate = wav_bytes_to_mono(archive.read(members[fname]))
                if target_rate is not None:
                    series, rate = resample(series, rate, target_rate), target_rate
                yield (
                    fname,
                    {
                        "series": series[:, None],
                        "sample_rate": rate,
                        "label": label_ids[row["label"]],
                        "filename": fname,
                        "freesound_id": int(fname.removesuffix(".wav")),
                        # The test set is drawn entirely from the manually verified (clean) portion.
                        "manually_verified": split == "test" or row["manually_verified"] == "1",
                        "noisy_small": row.get("noisy_small") == "1",
                        "license": licenses.get(fname, ""),
                    },
                )


def _load_ground_truth(meta_path, split: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(meta_path) as archive, archive.open(f"FSDnoisy18k.meta/{split}.csv") as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def _load_clip_licenses(doc_path) -> dict[str, str]:
    with zipfile.ZipFile(doc_path) as archive, archive.open("FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv") as fh:
        return {row["fname"]: row["license"] for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))}
