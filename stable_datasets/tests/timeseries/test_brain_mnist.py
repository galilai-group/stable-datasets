"""Tests for BrainMNIST dataset.

BrainMNIST (MindBigData) is an EEG dataset where brain signals were captured
while subjects viewed MNIST digits (0-9). The dataset includes multiple
device variants with different numbers of EEG channels and sample rates.
"""

import numpy as np
import pytest

from stable_datasets.timeseries.brain_mnist import BrainMNIST, BrainMNISTConfig


def _create_builder_without_download(config_name: str = "mindwave") -> BrainMNIST:
    """Create a BrainMNIST builder instance without triggering download.

    This bypasses the __new__ method that auto-downloads data, allowing us
    to test _info() and other methods without network access.
    """
    # Find the config
    config = None
    for c in BrainMNIST.BUILDER_CONFIGS:
        if c.name == config_name:
            config = c
            break

    if config is None:
        raise ValueError(f"Config '{config_name}' not found")

    # Create instance using object.__new__ to bypass auto-download
    builder = object.__new__(BrainMNIST)
    # Manually initialize the builder with minimal setup
    builder.config = config
    builder.name = "brain_mnist"
    return builder


class TestBrainMNISTMetadata:
    """Tests for dataset metadata and configuration (no network required)."""

    def test_version_is_set(self):
        """VERSION should be a proper datasets.Version, not 0.0.0."""
        import datasets

        assert isinstance(BrainMNIST.VERSION, datasets.Version)
        assert str(BrainMNIST.VERSION) != "0.0.0", "VERSION should be updated from stub"

    def test_source_has_required_fields(self):
        """SOURCE must have homepage, citation, and assets."""
        source = BrainMNIST.SOURCE
        assert "homepage" in source
        assert "citation" in source
        assert "assets" in source
        assert source["homepage"] is not None
        assert source["citation"] != "TBD"

    def test_has_builder_configs(self):
        """Should have multiple device configurations."""
        assert hasattr(BrainMNIST, "BUILDER_CONFIGS")
        assert len(BrainMNIST.BUILDER_CONFIGS) > 0

    def test_config_names(self):
        """Should have configs for different EEG devices."""
        config_names = [c.name for c in BrainMNIST.BUILDER_CONFIGS]
        # At minimum, we expect the MindWave variant (simplest, single channel)
        assert "mindwave" in config_names


class TestBrainMNISTConfig:
    """Tests for BrainMNISTConfig attributes (no network required)."""

    def test_mindwave_config_attributes(self):
        """MindWave config should have correct channel/sample counts."""
        configs = {c.name: c for c in BrainMNIST.BUILDER_CONFIGS}
        mw = configs["mindwave"]
        # MindWave: 1 channel, 1024 samples
        assert mw.num_channels == 1
        assert mw.num_samples == 1024
        assert hasattr(mw, "channel_names")

    def test_epoc_config_attributes(self):
        """EPOC config should have correct channel/sample counts."""
        configs = {c.name: c for c in BrainMNIST.BUILDER_CONFIGS}
        epoc = configs["epoc"]
        assert epoc.num_channels == 14
        assert epoc.num_samples == 256

    def test_muse_config_attributes(self):
        """Muse config should have correct channel/sample counts."""
        configs = {c.name: c for c in BrainMNIST.BUILDER_CONFIGS}
        muse = configs["muse"]
        assert muse.num_channels == 4
        assert muse.num_samples == 440

    def test_insight_config_attributes(self):
        """Insight config should have correct channel/sample counts."""
        configs = {c.name: c for c in BrainMNIST.BUILDER_CONFIGS}
        insight = configs["insight"]
        assert insight.num_channels == 5
        assert insight.num_samples == 256

    def test_all_configs_have_required_attributes(self):
        """All configs should have required attributes."""
        for config in BrainMNIST.BUILDER_CONFIGS:
            assert hasattr(config, "num_channels")
            assert hasattr(config, "num_samples")
            assert hasattr(config, "channel_names")
            assert hasattr(config, "sampling_rate")
            assert hasattr(config, "hf_dataset_id")
            assert len(config.channel_names) == config.num_channels


class TestBrainMNISTInfo:
    """Tests for the _info() method (no network required)."""

    def test_info_returns_dataset_info(self):
        """_info() should return a valid DatasetInfo."""
        import datasets

        builder = _create_builder_without_download("mindwave")
        info = builder._info()
        assert isinstance(info, datasets.DatasetInfo)

    def test_info_has_features(self):
        """DatasetInfo should define features."""
        builder = _create_builder_without_download("mindwave")
        info = builder._info()
        assert info.features is not None
        # Should have 'eeg' (signal data) and 'label' (digit 0-9)
        assert "eeg" in info.features
        assert "label" in info.features

    def test_label_is_class_label(self):
        """Label should be a ClassLabel with 11 classes (0-9 + no_stimulus)."""
        import datasets

        builder = _create_builder_without_download("mindwave")
        info = builder._info()
        label_feature = info.features["label"]
        assert isinstance(label_feature, datasets.ClassLabel)
        # Labels: 0-9 digits + 10 for no stimulus
        assert label_feature.num_classes == 11

    def test_eeg_feature_shape_single_channel(self):
        """Single-channel EEG feature should be a 1D Sequence."""
        import datasets

        builder = _create_builder_without_download("mindwave")
        info = builder._info()
        eeg_feature = info.features["eeg"]
        # Single channel should be a Sequence
        assert isinstance(eeg_feature, datasets.Sequence)
        assert eeg_feature.length == 1024

    def test_eeg_feature_shape_multi_channel(self):
        """Multi-channel EEG feature should be a 2D Array."""
        import datasets

        builder = _create_builder_without_download("epoc")
        info = builder._info()
        eeg_feature = info.features["eeg"]
        # Multi-channel should be Array2D
        assert isinstance(eeg_feature, datasets.Array2D)
        assert eeg_feature.shape == (14, 256)


@pytest.mark.download
class TestBrainMNISTDataLoading:
    """Integration tests that require downloading data.

    These tests are marked with @pytest.mark.download and should only
    run when explicitly requested, as they download data from the internet.
    """

    def test_load_train_split(self):
        """Should be able to load the train split."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        assert len(ds) > 0

    def test_load_test_split(self):
        """Should be able to load the test split."""
        ds = BrainMNIST(split="test", config_name="mindwave")
        assert len(ds) > 0

    def test_sample_has_expected_keys(self):
        """Each sample should have 'eeg' and 'label' keys."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        sample = ds[0]
        assert "eeg" in sample
        assert "label" in sample

    def test_eeg_shape_mindwave(self):
        """MindWave EEG data should have shape (1, 1024) or (1024,)."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        sample = ds[0]
        eeg = np.array(sample["eeg"])
        # MindWave: 1 channel, 1024 samples
        # Could be (1024,) for single channel or (1, 1024)
        assert eeg.shape in [(1024,), (1, 1024)]

    def test_label_range(self):
        """Labels should be valid class indices."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        sample = ds[0]
        label = sample["label"]
        # Label should be a valid integer (0-10 typically)
        assert isinstance(label, int)
        assert 0 <= label <= 10  # 0-9 for digits, possibly 10 for no_stimulus

    def test_train_split_size(self):
        """Train split should have expected number of samples (80% of total)."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        # MindWave dataset has ~67k total samples, 80% train = ~54k
        assert len(ds) > 50000

    def test_test_split_size(self):
        """Test split should have expected number of samples (20% of total)."""
        ds = BrainMNIST(split="test", config_name="mindwave")
        # 20% test = ~13k
        assert len(ds) > 10000

    def test_with_format_torch(self):
        """Should be compatible with PyTorch format."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        ds_torch = ds.with_format("torch")
        sample = ds_torch[0]
        import torch

        assert isinstance(sample["eeg"], torch.Tensor)

    def test_with_format_numpy(self):
        """Should be compatible with NumPy format."""
        ds = BrainMNIST(split="train", config_name="mindwave")
        ds_np = ds.with_format("numpy")
        sample = ds_np[0]
        assert isinstance(sample["eeg"], np.ndarray)


@pytest.mark.download
class TestBrainMNISTMultipleConfigs:
    """Tests for multiple device configurations.

    These tests verify that all device variants work correctly.
    """

    @pytest.mark.parametrize(
        "config_name,expected_channels,expected_samples",
        [
            ("mindwave", 1, 1024),
            ("epoc", 14, 256),
            ("muse", 4, 440),
            ("insight", 5, 256),
        ],
    )
    def test_config_eeg_shape(self, config_name, expected_channels, expected_samples):
        """Each config should produce EEG data with the correct shape."""
        import datasets as hf_datasets
        
        # Get the HuggingFace dataset ID for this config
        configs = {c.name: c for c in BrainMNIST.BUILDER_CONFIGS}
        config = configs[config_name]
        
        # Load just the first row using split slicing (avoids full dataset load)
        hf_ds = hf_datasets.load_dataset(
            config.hf_dataset_id,
            split="train[:1]",
        )
        row = hf_ds[0]
        
        # Extract EEG data from the row
        builder = _create_builder_without_download(config_name)
        eeg = builder._extract_eeg_data(row)

        if expected_channels == 1:
            # Single channel can be 1D or 2D
            assert eeg.shape in [(expected_samples,), (expected_channels, expected_samples)]
        else:
            # Multi-channel should be 2D: (channels, samples)
            assert eeg.shape == (expected_channels, expected_samples)


class TestBrainMNISTExtractEEGData:
    """Tests for the _extract_eeg_data helper method."""

    def test_extract_single_channel(self):
        """Should correctly extract single-channel EEG data."""
        builder = _create_builder_without_download("mindwave")

        # Create mock row data
        row = {"label": 5}
        for i in range(1024):
            row[f"FP1-{i}"] = float(i)

        eeg = builder._extract_eeg_data(row)
        assert eeg.shape == (1024,)
        assert eeg[0] == 0.0
        assert eeg[100] == 100.0

    def test_extract_multi_channel(self):
        """Should correctly extract multi-channel EEG data."""
        builder = _create_builder_without_download("epoc")

        # Create mock row data for EPOC (14 channels, 256 samples)
        row = {"label": 3}
        channel_names = builder.config.channel_names
        for ch_idx, channel in enumerate(channel_names):
            for i in range(256):
                row[f"{channel}-{i}"] = float(ch_idx * 256 + i)

        eeg = builder._extract_eeg_data(row)
        assert eeg.shape == (14, 256)
        # First channel, first sample
        assert eeg[0, 0] == 0.0
        # Second channel, first sample
        assert eeg[1, 0] == 256.0
