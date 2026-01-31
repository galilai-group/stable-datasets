import datasets
import numpy as np

from stable_datasets.utils import BaseDatasetBuilder


class BrainMNISTConfig(datasets.BuilderConfig):
    """BuilderConfig for BrainMNIST with per-device metadata."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_samples: int,
        channel_names: list[str],
        sampling_rate: int,
        hf_dataset_id: str,
        **kwargs,
    ):
        """Initialize BrainMNIST config.

        Args:
            num_channels: Number of EEG channels for this device.
            num_samples: Number of samples per channel per recording.
            channel_names: List of channel names (10-20 system locations).
            sampling_rate: Sampling rate in Hz.
            hf_dataset_id: HuggingFace dataset ID for this variant.
            **kwargs: Additional arguments passed to BuilderConfig.
        """
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.hf_dataset_id = hf_dataset_id


class BrainMNIST(BaseDatasetBuilder):
    """BrainMNIST (MindBigData) EEG Dataset.

    EEG signals captured while subjects viewed MNIST digits (0-9). The dataset
    includes recordings from multiple consumer EEG devices with different numbers
    of channels and sampling rates, making it suitable for brain-computer
    interface research and neural decoding experiments.

    Reference:
    - http://mindbigdata.com/opendb/index.html
    - https://huggingface.co/datasets/DavidVivancos/MindBigData2022
    - Paper: "MindBigData 2022: A Large Dataset of Brain Signals" (arXiv:2212.14746)

    Device Variants:
    - MindWave: 1 EEG channel (FP1), 1024 samples at 512Hz
    - EPOC: 14 EEG channels, 256 samples at 128Hz
    - Muse: 4 EEG channels (TP9, FP1, FP2, TP10), 440 samples at 220Hz
    - Insight: 5 EEG channels (AF3, AF4, T7, T8, PZ), 256 samples at 128Hz

    Labels:
        0-9: Digit being viewed
        10: No stimulus (baseline/rest period between digits)

    Example:
        >>> from stable_datasets.timeseries.brain_mnist import BrainMNIST
        >>> ds = BrainMNIST(split="train", config_name="mindwave")
        >>> sample = ds[0]
        >>> print(sample.keys())  # {'eeg', 'label'}
        >>> print(sample['eeg'].shape)  # (1024,) for MindWave
    """

    VERSION = datasets.Version("1.0.0")

    # Citation for the MindBigData 2022 paper
    _CITATION = """\
@article{vivancos2022mindbigdata,
    title={MindBigData 2022: A Large Dataset of Brain Signals},
    author={Vivancos, David},
    journal={arXiv preprint arXiv:2212.14746},
    year={2022},
    url={https://arxiv.org/abs/2212.14746}
}
"""

    # Static SOURCE for base validation - actual URLs determined by config
    SOURCE = {
        "homepage": "http://mindbigdata.com/opendb/index.html",
        "citation": _CITATION,
        "assets": {
            # We use HuggingFace datasets hub, so assets point to HF dataset pages
            # The actual downloading is handled by the datasets library
            "train": "https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_MW",
            "test": "https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_MW",
        },
    }

    BUILDER_CONFIGS = [
        BrainMNISTConfig(
            name="mindwave",
            description="NeuroSky MindWave - 1 EEG channel (FP1), 1024 samples at 512Hz",
            num_channels=1,
            num_samples=1024,
            channel_names=["FP1"],
            sampling_rate=512,
            hf_dataset_id="DavidVivancos/MindBigData2022_MNIST_MW",
        ),
        BrainMNISTConfig(
            name="epoc",
            description="Emotiv EPOC - 14 EEG channels, 256 samples at 128Hz",
            num_channels=14,
            num_samples=256,
            channel_names=["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
            sampling_rate=128,
            hf_dataset_id="DavidVivancos/MindBigData2022_MNIST_EP",
        ),
        BrainMNISTConfig(
            name="muse",
            description="Interaxon Muse - 4 EEG channels, 440 samples at 220Hz",
            num_channels=4,
            num_samples=440,
            channel_names=["TP9", "FP1", "FP2", "TP10"],
            sampling_rate=220,
            hf_dataset_id="DavidVivancos/MindBigData2022_MNIST_MU",
        ),
        BrainMNISTConfig(
            name="insight",
            description="Emotiv Insight - 5 EEG channels, 256 samples at 128Hz",
            num_channels=5,
            num_samples=256,
            channel_names=["AF3", "AF4", "T7", "T8", "PZ"],
            sampling_rate=128,
            hf_dataset_id="DavidVivancos/MindBigData2022_MNIST_IN",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mindwave"

    def _source(self) -> dict:
        """Return config-specific source information."""
        hf_url = f"https://huggingface.co/datasets/{self.config.hf_dataset_id}"
        return {
            "homepage": "http://mindbigdata.com/opendb/index.html",
            "citation": self._CITATION,
            "assets": {
                "train": hf_url,
                "test": hf_url,
            },
        }

    def _info(self) -> datasets.DatasetInfo:
        """Return dataset metadata."""
        # Labels: 0-9 for digits, 10 for no stimulus (-1 in original)
        label_names = [str(i) for i in range(10)] + ["no_stimulus"]

        # Feature definition depends on number of channels
        if self.config.num_channels == 1:
            # Single channel: 1D sequence
            eeg_feature = datasets.Sequence(
                datasets.Value("float32"),
                length=self.config.num_samples,
            )
        else:
            # Multi-channel: 2D array (channels x samples)
            eeg_feature = datasets.Array2D(
                shape=(self.config.num_channels, self.config.num_samples),
                dtype="float32",
            )

        features = datasets.Features(
            {
                "eeg": eeg_feature,
                "label": datasets.ClassLabel(names=label_names),
            }
        )

        return datasets.DatasetInfo(
            description=f"""BrainMNIST ({self.config.name}): EEG signals captured while viewing MNIST digits.
Device: {self.config.description}
Channels: {self.config.num_channels} ({", ".join(self.config.channel_names)})
Samples per recording: {self.config.num_samples}
Sampling rate: {self.config.sampling_rate} Hz
Labels: 0-9 (digits), 10 (no stimulus)""",
            features=features,
            supervised_keys=("eeg", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager):
        """Load data from HuggingFace datasets hub."""
        # Load the dataset from HuggingFace
        # We override the default _split_generators because we're using HF datasets
        # instead of custom download URLs
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
        ]

    def _generate_examples(self, split: str):
        """Generate examples from the HuggingFace dataset.

        Args:
            split: Dataset split ("train" or "test").

        Yields:
            Tuple of (key, example_dict) where example_dict contains:
                - eeg: numpy array of EEG signal data
                - label: integer class label (0-9 for digits, 10 for no_stimulus)
        """
        # Load from HuggingFace datasets hub
        hf_dataset = datasets.load_dataset(
            self.config.hf_dataset_id,
            split=split,
        )

        for idx, row in enumerate(hf_dataset):
            # Extract label - convert -1 (no stimulus) to 10
            label = row.get("label", row.get("Label", -1))
            if label == -1:
                label = 10  # Map -1 to "no_stimulus" class

            # Extract EEG data from columns
            # Column naming: ChannelName-SampleNum (e.g., FP1-0, FP1-1, ..., FP1-1023)
            eeg_data = self._extract_eeg_data(row)

            yield (
                idx,
                {
                    "eeg": eeg_data,
                    "label": label,
                },
            )

    def _extract_eeg_data(self, row: dict) -> np.ndarray:
        """Extract EEG signal data from a row.

        Args:
            row: Dictionary containing the row data with EEG columns.

        Returns:
            numpy array of shape (num_samples,) for single channel
            or (num_channels, num_samples) for multi-channel.
        """
        num_channels = self.config.num_channels
        num_samples = self.config.num_samples
        channel_names = self.config.channel_names

        if num_channels == 1:
            # Single channel: extract as 1D array
            channel_name = channel_names[0]
            eeg_data = np.array(
                [
                    row.get(f"{channel_name}-{i}", row.get(f"{channel_name.lower()}-{i}", 0.0))
                    for i in range(num_samples)
                ],
                dtype=np.float32,
            )
        else:
            # Multi-channel: extract as 2D array (channels x samples)
            eeg_data = np.zeros((num_channels, num_samples), dtype=np.float32)
            for ch_idx, channel_name in enumerate(channel_names):
                for sample_idx in range(num_samples):
                    # Try different column naming conventions
                    col_name = f"{channel_name}-{sample_idx}"
                    value = row.get(col_name, row.get(col_name.lower(), 0.0))
                    eeg_data[ch_idx, sample_idx] = float(value) if value is not None else 0.0

        return eeg_data
