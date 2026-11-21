import hashlib
import io
import zipfile

import numpy as np
import pyarrow as pa
import pytest

from stable_datasets.timeseries import MHEALTH


def test_mhealth_official_archive(tmp_path):
    downloads = tmp_path / "downloads"
    data = MHEALTH(download_dir=downloads, processed_cache_dir=tmp_path / "cache")
    assert set(data) == {"train"}
    recordings = data["train"]
    assert len(recordings) == 10
    assert recordings.table.schema.field("series").type == pa.list_(pa.list_(pa.float32()))
    assert recordings.table.schema.field("subject_id").type == pa.int32()
    paths = list(downloads.glob("MHEALTH.*.zip"))
    assert len(paths) == 1
    path = paths[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MHEALTH.SOURCE["assets"]["train"].checksum.split(":")[1]

    lengths = [161280, 130561, 122112, 116736, 119808, 98304, 104448, 129024, 135168, 98304]
    assert sum(lengths) == 1_215_745
    with zipfile.ZipFile(path) as archive:
        for index, length in enumerate(lengths):
            record = recordings[index]
            assert set(record) == {"series", "label", "subject_id"}
            assert record["subject_id"] == index + 1
            signals = np.asarray(record["series"], dtype=np.float32)
            labels = np.asarray(record["label"], dtype=np.int64)
            assert signals.shape == (length, 23)
            assert labels.shape == (length,)
            assert np.isfinite(signals).all()
            assert set(labels) == set(range(13))
            with archive.open(f"MHEALTHDATASET/mHealth_subject{index + 1}.log") as stream:
                source = np.loadtxt(stream, dtype=np.float32)
            np.testing.assert_array_equal(signals, source[:, :23])
            np.testing.assert_array_equal(labels, source[:, 23])


def _archive(tmp_path, invalid=None):
    path = tmp_path / "mhealth.zip"
    with zipfile.ZipFile(path, "w") as z:
        for subject in range(1, 11):
            signals = np.arange(69, dtype=np.float32).reshape(3, 23) + subject * 100
            rows = np.column_stack([signals, [0, 1, 12]])
            if subject == 1 and invalid is not None:
                rows = invalid
            text = io.StringIO()
            np.savetxt(text, rows)
            z.writestr(f"MHEALTHDATASET/mHealth_subject{subject}.log", text.getvalue())
    return path


def _builder():
    builder = object.__new__(MHEALTH)
    builder.__init__()
    return builder


def test_mhealth_preserves_recordings_labels_and_cache(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr("stable_datasets.utils.bulk_download", lambda *a, **k: [archive])
    cache = tmp_path / "cache"
    data = MHEALTH(processed_cache_dir=cache, download_dir=tmp_path)
    assert set(data) == {"train"}
    recordings = data["train"]
    assert len(recordings) == 10
    assert [recordings[i]["subject_id"] for i in range(10)] == list(range(1, 11))
    first = recordings[0]
    assert set(first) == {"series", "label", "subject_id"}
    np.testing.assert_array_equal(first["series"], np.arange(69).reshape(3, 23) + 100)
    assert first["label"] == [0, 1, 12]  # Includes null labels and preserves chronology.
    assert MHEALTH.CHANNEL_NAMES[3:5] == ("ecg_1", "ecg_2")
    assert _builder().info.features["label"].feature.names[0] == "null"

    def no_download(*args, **kwargs):
        pytest.fail("A complete processed cache must not download again")

    monkeypatch.setattr("stable_datasets.utils.bulk_download", no_download)
    cached = MHEALTH(split="train", processed_cache_dir=cache, download_dir=tmp_path)
    assert cached[9]["label"] == [0, 1, 12]
    with pytest.raises(ValueError, match="Split 'all' not found"):
        MHEALTH(split="all", processed_cache_dir=cache, download_dir=tmp_path)


@pytest.mark.parametrize("problem", ["columns", "nan", "negative_label", "large_label", "fractional_label"])
def test_mhealth_rejects_invalid_data(tmp_path, problem):
    rows = np.ones((2, 24))
    if problem == "columns":
        rows = rows[:, :23]
    elif problem == "nan":
        rows[0, 0] = np.nan
    elif problem == "negative_label":
        rows[0, 23] = -1
    elif problem == "large_label":
        rows[0, 23] = 13
    else:
        rows[0, 23] = 1.5
    archive = _archive(tmp_path, rows)
    with pytest.raises(ValueError, match="columns|finite|labels"):
        list(_builder()._generate_examples(archive, "train"))
