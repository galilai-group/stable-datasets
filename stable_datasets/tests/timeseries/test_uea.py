import zipfile

import numpy as np
import pytest

from stable_datasets.timeseries._uea import iter_examples


_TS = """#A tiny UEA-style file: two cases, two channels, three steps.
@problemName Toy
@timeStamps false
@missing false
@univariate false
@dimensions 2
@equalLength true
@seriesLength 3
@classLabel true Up Down
@data
1,2,3:4,5,6:Up
7,8,9:10,11,12:Down
"""


@pytest.fixture
def toy_zip(tmp_path):
    path = tmp_path / "Toy.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Toy_TRAIN.ts", _TS)
    return path


def test_iter_examples_reads_steps_by_channels(toy_zip):
    rows = list(iter_examples(toy_zip, "Toy_TRAIN.ts", (3, 2), ["down", "up"]))
    assert [key for key, _ in rows] == [0, 1]
    first = rows[0][1]
    assert first["series"].dtype == np.float32
    np.testing.assert_array_equal(first["series"], [[1, 4], [2, 5], [3, 6]])
    assert [example["label"] for _, example in rows] == [1, 0]


def test_iter_examples_rejects_unexpected_shape(toy_zip):
    with pytest.raises(ValueError, match="Unexpected series shape"):
        list(iter_examples(toy_zip, "Toy_TRAIN.ts", (2, 3), ["down", "up"]))


def test_iter_examples_rejects_unknown_labels(toy_zip):
    with pytest.raises(ValueError, match="Unexpected labels"):
        list(iter_examples(toy_zip, "Toy_TRAIN.ts", (3, 2), ["up"]))
