import tarfile

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


SPEECH_COMMAND_LABELS = [
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
]


class SpeechCommands(BaseDatasetBuilder):
    """Speech Commands keyword classification dataset."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html",
        assets={
            "train": DownloadInfo(
                url="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
                filename="speech_commands_v0.01.tar.gz",
            ),
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Speech Commands keyword classification dataset.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "label": ClassLabel(names=SPEECH_COMMAND_LABELS),
                    "label_name": Value("string"),
                    "speaker_id": Value("string"),
                    "utterance_id": Value("int32"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        del split
        with tarfile.open(data_path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.name.lower().endswith(".wav"):
                    continue
                parts = member.name.split("/")
                if len(parts) < 2:
                    continue
                label_name = parts[-2]
                if label_name == "_background_noise_" or label_name not in SPEECH_COMMAND_LABELS:
                    continue
                filename = parts[-1]
                stem = filename[:-4]
                speaker_id = ""
                utterance_id = 0
                if "_nohash_" in stem:
                    speaker_id, utterance = stem.split("_nohash_", 1)
                    try:
                        utterance_id = int(utterance)
                    except ValueError:
                        utterance_id = 0
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield member.name, {
                    "series": wav_bytes_to_series(extracted.read()),
                    "label": SPEECH_COMMAND_LABELS.index(label_name),
                    "label_name": label_name,
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "filename": filename,
                }
