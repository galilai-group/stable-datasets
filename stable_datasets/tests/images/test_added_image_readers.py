import pytest

import stable_datasets.images as images
from stable_datasets.utils import BaseDatasetBuilder


@pytest.mark.parametrize("name", ["AID", "RESISC45", "JenaFlowers30", "Flavia", "IP102"])
def test_added_image_reader_is_public(name):
    assert name in images.__all__
    assert issubclass(getattr(images, name), BaseDatasetBuilder)
