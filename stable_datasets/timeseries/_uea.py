"""Shared loading code for datasets from the UEA multivariate time series classification archive."""

import tempfile
import zipfile

import numpy as np

from stable_datasets.utils import load_from_tsfile_to_dataframe


# Citation of the archive, which every UEA dataset cites next to its original source.
UEA_ARCHIVE_CITATION = """@article{bagnall2018uea,
    title={The UEA multivariate time series classification archive, 2018},
    author={Bagnall, Anthony and Dau, Hoang Anh and Lines, Jason and Flynn, Michael and Large, James and Bostrom, Aaron and Southam, Paul and Keogh, Eamonn},
    journal={arXiv preprint arXiv:1811.00075},
    year={2018}
}"""


def read_ts_from_zip(zip_path, member):
    """Return ``(series, labels)`` from a ``.ts`` file in an archive zip, with series of shape (cases, steps, channels).

    Labels come back in lower case, because ``load_from_tsfile_to_dataframe`` lower-cases every line.
    """
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory() as tmp:
        x_df, labels = load_from_tsfile_to_dataframe(archive.extract(member, tmp))
    series = np.stack([np.stack(x_df[col].map(np.asarray)) for col in x_df.columns], axis=-1)
    return series.astype(np.float32), [str(label) for label in labels]


def iter_examples(zip_path, member, shape, labels):
    """Yield ``(idx, {"series", "label"})`` from an equal-length ``.ts`` file, checking every series has ``shape``.

    ``shape`` is ``(steps, channels)``. ``labels`` lists the labels as they appear in the file (lower case), in
    class id order, so ``labels[i]`` becomes class ``i``.
    """
    series, raw_labels = read_ts_from_zip(zip_path, member)
    if series.shape[1:] != tuple(shape):
        raise ValueError(
            f"Unexpected series shape in {member}: {series.shape}, expected (cases, {shape[0]}, {shape[1]})"
        )
    unknown = sorted(set(raw_labels) - set(labels))
    if unknown:
        raise ValueError(f"Unexpected labels in {member}: {unknown}")
    label_to_id = {name: idx for idx, name in enumerate(labels)}
    for idx, (x, y) in enumerate(zip(series, raw_labels)):
        yield idx, {"series": x, "label": label_to_id[y]}
