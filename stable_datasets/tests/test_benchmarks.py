"""Tests for the benchmarks package."""

import pytest
import torch

from benchmarks.dataset import (
    DATASET_CONFIGS,
    IMAGE_DATASET_CONFIGS,
    INCLUDED_IMAGE_DATASETS,
    TIMESERIES_DATASET_CONFIGS,
    DatasetConfig,
    _get_dataset_class,
    _load_validation_split,
    get_config,
)
from benchmarks.models import (
    build_module,
    collate_multicrop,
    collate_multiview,
    collate_single,
    create_backbone,
    create_projector,
    get_embedding_dim,
    get_transforms,
    ssl_augmentation,
    val_transform,
)
from stable_datasets.schema import ClassLabel, Features, Image, Sequence


# A minimal ds_config for testing (no real data needed)
DUMMY_CONFIG = DatasetConfig(
    name="test",
    display_name="Test",
    num_classes=10,
    channels=3,
    mean=[0.5, 0.5, 0.5],
    std=[0.2, 0.2, 0.2],
)

DUMMY_GRAY = DatasetConfig(
    name="test_gray",
    display_name="Test Gray",
    num_classes=10,
    channels=1,
    mean=[0.5],
    std=[0.2],
)


# Dataset config


class TestDatasetConfig:
    def test_get_config_known(self):
        cfg = get_config("cifar10")
        assert cfg.name == "cifar10"
        assert cfg.display_name == "CIFAR-10"
        assert cfg.num_classes == 10
        assert cfg.channels == 3
        assert cfg.image_size == (224, 224)

    def test_get_config_case_insensitive(self):
        assert get_config("CIFAR10").name == "cifar10"
        assert get_config("Cifar10").name == "cifar10"

    def test_get_config_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_config("nonexistent_dataset_xyz")

    def test_grayscale_has_one_channel(self):
        cfg = get_config("fashionmnist")
        assert cfg.channels == 1
        assert len(cfg.mean) == 1
        assert len(cfg.std) == 1

    def test_all_configs_have_display_names(self):
        for key, cfg in DATASET_CONFIGS.items():
            assert cfg.display_name, f"{key} missing display_name"
            assert cfg.display_name != cfg.name, f"{key} display_name is just the key"

    def test_all_configs_resolve_to_class_labeled_builders(self):
        for key, cfg in DATASET_CONFIGS.items():
            cls = _get_dataset_class(cfg)
            builder = object.__new__(cls)
            cls.__init__(builder, **cfg.builder_kwargs)
            label_feature = builder.info.features.get(cfg.label_key)
            assert isinstance(label_feature, ClassLabel), f"{key} label is not ClassLabel"
            assert label_feature.num_classes == cfg.num_classes

    def test_image_normalization_matches_channels(self):
        for key, cfg in IMAGE_DATASET_CONFIGS.items():
            assert len(cfg.mean) == cfg.channels, f"{key} mean length mismatch"
            assert len(cfg.std) == cfg.channels, f"{key} std length mismatch"

    def test_included_image_dataset_set_matches_expected_policy(self):
        assert "galaxy10" in INCLUDED_IMAGE_DATASETS
        assert "awa2" in INCLUDED_IMAGE_DATASETS
        assert "imagenet" not in INCLUDED_IMAGE_DATASETS
        assert "tinyimagenet" not in INCLUDED_IMAGE_DATASETS

    def test_timeseries_normalization_matches_channels(self):
        for key, cfg in TIMESERIES_DATASET_CONFIGS.items():
            assert cfg.input_key == "series"
            assert cfg.label_key == "label"
            assert len(cfg.mean) == cfg.num_channels, f"{key} mean length mismatch"
            assert len(cfg.std) == cfg.num_channels, f"{key} std length mismatch"

    def test_timeseries_builders_have_series_feature(self):
        for key, cfg in TIMESERIES_DATASET_CONFIGS.items():
            cls = _get_dataset_class(cfg)
            builder = object.__new__(cls)
            cls.__init__(builder, **cfg.builder_kwargs)
            assert isinstance(builder.info.features["series"], Sequence), key

    def test_validation_prefers_validation_before_test(self):
        class FakeDataset:
            features = Features({"image": Image(), "label": ClassLabel(num_classes=2)})

            def __init__(self, split):
                self.split = split

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {"label": 1}

        class FakeBuilder:
            def __new__(cls, split, **kwargs):
                if split in {"train", "validation", "test"}:
                    return FakeDataset(split)
                raise ValueError(split)

        cfg = DatasetConfig("fake", "Fake", 2, channels=3, mean=[0.5] * 3, std=[0.2] * 3)
        assert _load_validation_split(FakeBuilder, cfg).split == "validation"

    def test_validation_skips_unlabeled_test_for_holdout(self):
        class FakeDataset:
            features = Features({"image": Image(), "label": ClassLabel(num_classes=2)})

            def __init__(self, split):
                self.split = split

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {"label": -1}

        class FakeBuilder:
            def __new__(cls, split, **kwargs):
                if split == "test":
                    return FakeDataset(split)
                raise ValueError(split)

        cfg = DatasetConfig("fake", "Fake", 2, channels=3, mean=[0.5] * 3, std=[0.2] * 3)
        assert _load_validation_split(FakeBuilder, cfg) is None


# Backbone and projector


class TestBackboneAndProjector:
    def test_create_backbone_vit_small(self):
        class FakeBackboneCfg:
            type = "vit"
            size = "small"
            patch_size = 16

        backbone = create_backbone(FakeBackboneCfg(), DUMMY_CONFIG)
        assert hasattr(backbone, "embed_dim")
        assert backbone.embed_dim == 384

    def test_get_embedding_dim(self):
        class FakeBackboneCfg:
            type = "vit"
            size = "small"
            patch_size = 16

        backbone = create_backbone(FakeBackboneCfg(), DUMMY_CONFIG)
        assert get_embedding_dim(backbone) == 384

    def test_create_projector_shape(self):
        proj = create_projector(384, 2048, 128)
        x = torch.randn(4, 384)
        out = proj(x)
        assert out.shape == (4, 128)


# Transforms


class TestTransforms:
    def test_val_transform_output_shape(self):
        t = val_transform(DUMMY_CONFIG)
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        sample = t({"image": img, "label": 0})
        assert sample["image"].shape == (3, 224, 224)
        assert sample["label"] == 0

    def test_ssl_augmentation_rgb(self):
        t = ssl_augmentation(DUMMY_CONFIG, (224, 224), (0.08, 1.0))
        from PIL import Image

        img = Image.new("RGB", (32, 32))
        sample = t({"image": img, "label": 5})
        assert sample["image"].shape == (3, 224, 224)

    def test_ssl_augmentation_grayscale_skips_color_jitter(self):
        # Should not raise even though input is grayscale
        t = ssl_augmentation(DUMMY_GRAY, (224, 224), (0.08, 1.0))
        from PIL import Image

        img = Image.new("L", (32, 32))
        sample = t({"image": img, "label": 0})
        # RGB() converts to 3 channels
        assert sample["image"].shape[0] == 3

    @pytest.mark.parametrize(
        "model_name",
        [
            "simclr",
            "dino",
            "mae",
            "lejepa",
            "nnclr",
            "barlow_twins",
            "supervised",
        ],
    )
    def test_get_transforms_returns_triple(self, model_name):
        train_t, val_t, collate_fn = get_transforms(model_name, DUMMY_CONFIG)
        assert callable(train_t)
        assert callable(val_t)
        assert callable(collate_fn)


# Collation


def _make_single_sample():
    return {"image": torch.randn(3, 224, 224), "label": 0}


def _make_multiview_sample(n_views=2):
    views = [{"image": torch.randn(3, 224, 224), "label": 0} for _ in range(n_views)]
    return {"views": views}


def _make_multicrop_sample():
    return {
        "global_1": {"image": torch.randn(3, 224, 224), "label": 0},
        "global_2": {"image": torch.randn(3, 224, 224), "label": 0},
        "local_1": {"image": torch.randn(3, 112, 112), "label": 0},
    }


class TestCollation:
    def test_collate_single(self):
        batch = [_make_single_sample() for _ in range(4)]
        out = collate_single(batch)
        assert out["image"].shape == (4, 3, 224, 224)
        assert out["label"].shape == (4,)

    def test_collate_multiview(self):
        batch = [_make_multiview_sample(2) for _ in range(4)]
        out = collate_multiview(batch)
        assert len(out["views"]) == 2
        assert out["views"][0]["image"].shape == (4, 3, 224, 224)
        assert out["views"][1]["image"].shape == (4, 3, 224, 224)

    def test_collate_multicrop(self):
        batch = [_make_multicrop_sample() for _ in range(4)]
        out = collate_multicrop(batch)
        assert out["global_1"]["image"].shape == (4, 3, 224, 224)
        assert out["local_1"]["image"].shape == (4, 3, 112, 112)


# Smoke test: full pipeline with tiny config


class TestSmoke:
    @pytest.mark.parametrize(
        "model_name",
        [
            "simclr",
            "mae",
            "lejepa",
            "nnclr",
            "barlow_twins",
            "supervised",
        ],
    )
    def test_build_module(self, model_name):
        """Build each model and verify it produces a module with correct embed_dim."""
        from omegaconf import OmegaConf

        backbone_cfg = OmegaConf.create(
            {
                "name": "vit_small",
                "type": "vit",
                "size": "small",
                "patch_size": 16,
            }
        )

        # Minimal model configs per model
        model_cfgs = {
            "simclr": {
                "name": "simclr",
                "projector": {"hidden_dim": 64, "output_dim": 32},
                "loss": {"temperature": 0.5},
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
            "mae": {
                "name": "mae",
                "mask_ratio": 0.75,
                "norm_pix_loss": True,
                "decoder": {"embed_dim": 64, "depth": 1, "num_heads": 4},
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.95]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
            "lejepa": {
                "name": "lejepa",
                "projector": {"hidden_dim": 64, "output_dim": 32},
                "loss": {"lamb": 0.02, "t_max": 3.0, "n_points": 17, "num_slices": 16},
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
            "nnclr": {
                "name": "nnclr",
                "projector": {"hidden_dim": 64, "output_dim": 32},
                "predictor": {"hidden_dim": 64, "output_dim": 32},
                "loss": {"temperature": 0.1},
                "queue_size": 64,
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
            "barlow_twins": {
                "name": "barlow_twins",
                "projector": {"hidden_dim": 64, "output_dim": 32},
                "loss": {"lambda_coeff": 0.005},
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
            "supervised": {
                "name": "supervised",
                "vit_optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.999]},
                "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            },
        }

        cfg = OmegaConf.create(
            {
                "model": model_cfgs[model_name],
                "backbone": backbone_cfg,
            }
        )

        module, embed_dim = build_module(cfg, DUMMY_CONFIG)
        assert embed_dim == 384  # vit_small
        assert hasattr(module, "backbone")
