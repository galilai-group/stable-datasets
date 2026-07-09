import io
import tarfile

import numpy as np

from stable_datasets.schema import DatasetInfo, Features, Sequence, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


class LibriSpeech(BaseDatasetBuilder):
    """Automatic Speech Recognition / Speaker Classification.

    `LibriSpeech <https://www.openslr.org/12>`_ is a corpus of approximately 1000
    hours of 16kHz read English speech, derived from audiobooks in the LibriVox
    project. This builder loads the **train-clean-100** subset (100 hours, ~28.5k
    utterances) and the **test-clean** subset (~2.6k utterances).

    Each example contains the raw waveform (as float32 samples), the speaker ID,
    the transcript text, and the sample rate.

    Requires the ``soundfile`` package (``pip install soundfile``).
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.openslr.org/12",
        "assets": {
            "train": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            "test": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        },
        "citation": """@inproceedings{panayotov2015librispeech,
            title={Librispeech: an {ASR} corpus based on public domain audio books},
            author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
            booktitle={2015 IEEE International Conference on Acoustics, Speech and
                       Signal Processing (ICASSP)},
            pages={5206--5210},
            year={2015},
            organization={IEEE}}""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "LibriSpeech is a corpus of approximately 1000 hours of 16kHz "
                "read English speech derived from audiobooks. This builder "
                "provides the train-clean-100 and test-clean subsets."
            ),
            features=Features(
                {
                    "audio": Sequence(Value("float32")),
                    "sample_rate": Value("int32"),
                    "speaker_id": Value("int64"),
                    "transcript": Value("string"),
                }
            ),
            supervised_keys=("audio", "transcript"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the LibriSpeech tar.gz archive.

        The archive structure is::

            LibriSpeech/<subset>/
                <speaker_id>/
                    <chapter_id>/
                        <speaker_id>-<chapter_id>-<utterance_id>.flac
                        <speaker_id>-<chapter_id>.trans.txt
        """
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "LibriSpeech requires the 'soundfile' package. "
                "Install it with: pip install soundfile"
            )

        # First pass: collect all transcripts from .trans.txt files
        transcripts = {}
        with tarfile.open(data_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".trans.txt"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    for line in f.read().decode("utf-8").strip().splitlines():
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            utterance_id, text = parts
                            transcripts[utterance_id] = text

        # Second pass: read audio files and pair with transcripts
        idx = 0
        with tarfile.open(data_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".flac"):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                # Read FLAC audio via soundfile
                audio_bytes = f.read()
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

                # Extract utterance ID and speaker ID from the file path
                # Path: LibriSpeech/<subset>/<speaker>/<chapter>/<spk>-<chap>-<utt>.flac
                filename = member.name.rsplit("/", 1)[-1]
                utterance_id = filename.replace(".flac", "")
                speaker_id = int(utterance_id.split("-")[0])

                transcript = transcripts.get(utterance_id, "")

                yield idx, {
                    "audio": audio_data.astype(np.float32).tolist(),
                    "sample_rate": sample_rate,
                    "speaker_id": speaker_id,
                    "transcript": transcript,
                }
                idx += 1
