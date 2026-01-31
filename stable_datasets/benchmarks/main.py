"""SSL Baselines for stable-datasets.

Usage:
    python -m stable_datasets.benchmarks.main --datasets cifar10 --models simclr
    python -m stable_datasets.benchmarks.main --datasets all --models all
"""

from __future__ import annotations

import logging, argparse, sys
from pathlib import Path
from unittest.mock import patch

import stable_datasets as sds


def _mock_downloads_on_macos():
    """Mock download functions on MacOS to skip actual downloads."""
    if sys.platform != "darwin":
        return None

    logging.info("MacOS detected: mocking download functions")

    def mock_download(url, dest_folder=None, **kwargs) -> Path:
        """Mock download that returns a fake path."""
        from urllib.parse import urlparse
        import os
        filename = os.path.basename(urlparse(url).path)
        if dest_folder is None:
            dest_folder = Path.home() / ".stable_datasets" / "downloads"
        return Path(dest_folder) / filename

    def mock_bulk_download(urls, dest_folder, **kwargs) -> list[Path]:
        """Mock bulk download that returns fake paths."""
        return [mock_download(url, dest_folder) for url in urls]

    # Patch the download functions in stable_datasets.utils
    patches = [
        patch("stable_datasets.utils.download", mock_download),
        patch("stable_datasets.utils.bulk_download", mock_bulk_download),
    ]
    for p in patches:
        p.start()

    return patches

from stable_datasets.benchmarks.dataset_factory       import create_dataset
from stable_datasets.benchmarks.experiment_factory    import AVAILABLE_MODELS, create_experiment


def _get_all_dataset_classes() -> list[type]:
    """Return all dataset builder classes from stable_datasets.images."""
    return [
        cls for cls in vars(sds.images).values()
        if (
            isinstance(cls, type) 
            and issubclass(cls, sds.BaseDatasetBuilder) 
            # below takes too long
            and cls.__name__.lower() != 'cars196' 
            and cls.__name__.lower() != 'cifar10c'
            and cls.__name__.lower() != 'cifar100c'
        )
    ]


DATASET_CLASSES = _get_all_dataset_classes()
MODELS = list(AVAILABLE_MODELS)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    all_dataset_names = [cls.__name__.lower() for cls in DATASET_CLASSES]
    dataset_choices = all_dataset_names + ["all"]
    model_choices = MODELS + ["all"]

    parser = argparse.ArgumentParser(description="Run SSL baselines on stable-datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=dataset_choices,
        help="Datasets to run experiments on",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=model_choices,
        help="SSL models to train",
    )
    return parser.parse_args()


def main() -> None:
    """Run SSL baseline experiments."""
    logging.basicConfig(level=logging.INFO)
    _mock_downloads_on_macos()
    args = parse_args()

    all_dataset_names = [cls.__name__.lower() for cls in DATASET_CLASSES]

    dataset_names = args.datasets
    if 'all' in dataset_names:
        dataset_names = all_dataset_names

    model_names = args.models
    if 'all' in model_names:
        model_names = MODELS

    datasets_loaded = []
    for name in dataset_names:
        if name == 'emnist':
            datasets_loaded.append(create_dataset(name, config_name="balanced"))
        elif name == 'medmnist':
            datasets_loaded.append(create_dataset(name, config_name="pneumoniamnist"))
        else:
            datasets_loaded.append(create_dataset(name))
    
    experiments = [
        (
            create_experiment(model_name=model_name, data_module=data, config=config),
            model_name, data
        )
        for model_name   in model_names
        for data, config in datasets_loaded
    ]

    for experiment, model_name, data_cfg in experiments:
        logging.info(f'Running {experiment} with {model_name=} and dataset "{data_cfg.name}"')
        experiment()


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--datasets", "all", "--models", "all"]
    main()
