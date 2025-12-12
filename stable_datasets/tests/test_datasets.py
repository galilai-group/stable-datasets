import inspect

import pytest

import stable_datasets.images as images
from stable_datasets.utils import BaseDatasetBuilder


def get_all_dataset_classes():
    datasets = []
    for name, obj in inspect.getmembers(images):
        if not inspect.isclass(obj):
            continue

        if issubclass(obj, BaseDatasetBuilder) and obj is not BaseDatasetBuilder:
            datasets.append((name, obj))

    return datasets


DATASETS = get_all_dataset_classes()


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset(dataset):
    dataset_name, dataset_class = dataset

    configs = [None]
    if hasattr(dataset_class, "BUILDER_CONFIGS"):
        configs = dataset_class.BUILDER_CONFIGS

    for config in configs:
        dataset = dataset_class(config_name=config.name if config else None)

    return
