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
    # Datasets to skip
    SKIP_DATASETS = {
        'cars196',      # Takes too long to download
        'cifar10c',     # Takes too long
        'cifar100c',    # Takes too long
        'clevrer',      # Video dataset - not yet supported
    }
    return [
        cls for cls in vars(sds.images).values()
        if (
            isinstance(cls, type) 
            and issubclass(cls, sds.BaseDatasetBuilder) 
            and cls.__name__.lower() not in SKIP_DATASETS
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

    # Print experiment summary
    total_experiments = len(dataset_names) * len(model_names)
    logging.info(f"\n{'='*60}")
    logging.info(f"SSL Baseline Experiments")
    logging.info(f"{'='*60}")
    logging.info(f"Models ({len(model_names)}): {', '.join(model_names)}")
    logging.info(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")
    logging.info(f"Total experiments: {total_experiments}")
    logging.info(f"{'='*60}\n")

    # Run experiments one at a time to avoid callback queue reuse issues
    for dataset_name in dataset_names:
        for model_name in model_names:
            # Clear OnlineQueue class-level caches to prevent dimension mismatches
            # between experiments with different embedding dimensions
            from stable_pretraining.callbacks.queue import OnlineQueue
            OnlineQueue._shared_queues.clear()
            OnlineQueue._queue_info.clear()
            
            # Load dataset with model-specific transforms
            if dataset_name == 'emnist':
                data, config = create_dataset(dataset_name, model_name=model_name, config_name="balanced")
            elif dataset_name == 'medmnist':
                data, config = create_dataset(dataset_name, model_name=model_name, config_name="pneumoniamnist")
            else:
                data, config = create_dataset(dataset_name, model_name=model_name)
            
            # Create experiment fresh (new callbacks with empty queues)
            experiment, experiment_info = create_experiment(
                model_name=model_name, 
                data_module=data, 
                config=config, 
                max_epochs=2
            )
            
            logging.info(f'Running {model_name} on "{config.name}" {experiment_info}')
            experiment()


if __name__ == "__main__":
    import sys
    sys.argv[1:] = ["--datasets", "all", "--models", "all"]
    main()
