"""SSL Baselines for stable-datasets.

Usage:
    python -m stable_datasets.benchmarks.main --datasets cifar10 --models simclr
    python -m stable_datasets.benchmarks.main --datasets all --models all
"""

from __future__ import annotations

import logging, argparse, stable_datasets as sds

from stable_datasets.benchmarks.dataset_factory       import create_dataset
from stable_datasets.benchmarks.experiment_factory    import AVAILABLE_MODELS, create_experiment


def _get_all_dataset_classes() -> list[type]:
    """Return all dataset builder classes from stable_datasets.images."""
    return [
        cls for cls in vars(sds.images).values()
        if isinstance(cls, type) and issubclass(cls, sds.BaseDatasetBuilder)
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
    args = parse_args()

    all_dataset_names = [cls.__name__.lower() for cls in DATASET_CLASSES]

    dataset_names = args.datasets
    if 'all' in dataset_names:
        dataset_names = all_dataset_names

    model_names = args.models
    if 'all' in model_names:
        model_names = MODELS

    datasets_loaded = [create_dataset(name) for name in dataset_names]
    
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
