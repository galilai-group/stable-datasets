import io
import wave
import zipfile

import numpy as np
import pytest
from scipy.io.wavfile import write as wav_write

from stable_datasets.schema import ClassLabel, Sequence, Value
from stable_datasets.timeseries import FSDD


SOURCE_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
ARCHIVE_SHA256 = "beaca4fb849cad13103b2cffba913b3d7fdb5f15222da4ddd07187b3c38a757a"


def _wav_bytes(samples: np.ndarray, sample_rate: int = 8000, channels: int = 1) -> bytes:
    samples = np.asarray(samples, dtype="<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buf.getvalue()


def _float_wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    wav_write(buf, sample_rate, np.asarray(samples, dtype=np.float32))
    return buf.getvalue()


def _write_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _builder():
    builder = object.__new__(FSDD)
    builder.__init__()
    return builder


def test_fsdd_metadata_and_public_export():
    builder = _builder()
    archive_info = FSDD.SOURCE["assets"]["archive"]

    assert SOURCE_REVISION in archive_info.url
    assert archive_info.checksum == f"sha256:{ARCHIVE_SHA256}"
    assert FSDD.SOURCE["license"] == "CC BY-SA 4.0"
    assert builder.info.supervised_keys == ("series", "label")
    assert list(builder.info.features) == [
        "series",
        "label",
        "speaker_id",
        "utterance_id",
        "filename",
        "sample_rate",
    ]
    assert isinstance(builder.info.features["series"], Sequence)
    assert isinstance(builder.info.features["series"].feature, Sequence)
    assert builder.info.features["series"].feature.feature.dtype == "float32"
    assert isinstance(builder.info.features["label"], ClassLabel)
    assert builder.info.features["label"].num_classes == 10
    assert builder.info.features["speaker_id"].dtype == "string"
    assert builder.info.features["utterance_id"].dtype == "int32"
    assert isinstance(builder.info.features["sample_rate"], Value)
    assert builder.info.features["sample_rate"].dtype == "int32"


def test_fsdd_official_split_pcm_values_and_cache(tmp_path, monkeypatch):
    archive_path = tmp_path / "fsdd.zip"
    test_samples = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    train_samples = np.array([4, -5, 6], dtype=np.int16)
    _write_zip(
        archive_path,
        {
            "free-spoken-digit-dataset/recordings/4_george_4.wav": _wav_bytes(test_samples),
            "free-spoken-digit-dataset/recordings/5_jackson_5.wav": _wav_bytes(train_samples),
            "free-spoken-digit-dataset/recordings/9_theo_49.WAV": _wav_bytes(np.array([7, 8], dtype=np.int16)),
            "free-spoken-digit-dataset/utils/0_ignored_0.wav": _wav_bytes(np.array([9], dtype=np.int16)),
            "free-spoken-digit-dataset/recordings/README.md": b"not audio",
        },
    )
    download_calls = 0

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        nonlocal download_calls
        del dest_folder, checksums
        download_calls += 1
        assert len(urls) == 1
        return [archive_path]

    monkeypatch.setattr("stable_datasets.timeseries.fsdd.bulk_download", _fake_bulk_download)
    processed = tmp_path / "processed"

    train = FSDD(split="train", processed_cache_dir=str(processed))
    test = FSDD(split="test", processed_cache_dir=str(processed))
    cached_train = FSDD(split="train", processed_cache_dir=str(processed))

    assert download_calls == 1
    assert len(train) == len(cached_train) == 2
    assert len(test) == 1
    assert [train[i]["utterance_id"] for i in range(len(train))] == [5, 49]
    assert train[0]["speaker_id"] == "jackson"
    assert train[0]["filename"] == "5_jackson_5.wav"
    assert train[0]["sample_rate"] == 8000
    assert test[0]["label"] == 4
    assert test[0]["speaker_id"] == "george"
    assert test[0]["utterance_id"] == 4
    assert test[0]["filename"] == "4_george_4.wav"
    assert np.asarray(test[0]["series"]).shape == (len(test_samples), 1)
    np.testing.assert_array_equal(np.asarray(test[0]["series"])[:, 0], test_samples.astype(np.float32))
    assert train.table.schema.field("series").type.value_type.value_type.to_pandas_dtype() == np.float32


def test_fsdd_generator_returns_float32_column_vector(tmp_path):
    archive_path = tmp_path / "fsdd.zip"
    samples = np.array([1, -2, 3], dtype=np.int16)
    _write_zip(
        archive_path,
        {"snapshot/recordings/2_nicolas_5.wav": _wav_bytes(samples)},
    )

    [(key, example)] = list(_builder()._generate_examples(archive_path, "train"))

    assert key == "2_nicolas_5.wav"
    assert example["series"].dtype == np.float32
    assert example["series"].shape == (3, 1)
    np.testing.assert_array_equal(example["series"][:, 0], samples.astype(np.float32))
    assert example["speaker_id"] == "nicolas"


@pytest.mark.parametrize(
    ("filename", "wav_data", "split", "error"),
    [
        ("not-an-fsdd-name.wav", _wav_bytes(np.array([1], dtype=np.int16)), "train", "filename"),
        ("10_jackson_5.wav", _wav_bytes(np.array([1], dtype=np.int16)), "train", "digit"),
        ("1_ jackson_5.wav", _wav_bytes(np.array([1], dtype=np.int16)), "train", "speaker"),
        ("1_jackson_50.wav", _wav_bytes(np.array([1], dtype=np.int16)), "train", "index"),
        ("1_jackson_5.wav", _wav_bytes(np.array([1], dtype=np.int16), sample_rate=16000), "train", "sample rate"),
        (
            "1_jackson_5.wav",
            _wav_bytes(np.array([[1, -1], [2, -2]], dtype=np.int16), channels=2),
            "train",
            "mono",
        ),
        ("1_jackson_5.wav", _wav_bytes(np.array([], dtype=np.int16)), "train", "empty"),
        (
            "1_jackson_5.wav",
            _float_wav_bytes(np.array([0.0, np.nan], dtype=np.float32)),
            "train",
            "non-finite",
        ),
    ],
)
def test_fsdd_rejects_invalid_recordings(tmp_path, filename, wav_data, split, error):
    archive_path = tmp_path / "invalid.zip"
    _write_zip(archive_path, {f"snapshot/recordings/{filename}": wav_data})

    with pytest.raises(ValueError, match=error):
        list(_builder()._generate_examples(archive_path, split))


def test_fsdd_rejects_unknown_split(tmp_path):
    archive_path = tmp_path / "fsdd.zip"
    _write_zip(
        archive_path,
        {"snapshot/recordings/0_george_0.wav": _wav_bytes(np.array([1], dtype=np.int16))},
    )

    with pytest.raises(ValueError, match="Unknown FSDD split"):
        list(_builder()._generate_examples(archive_path, "validation"))


def test_fsdd_rejects_invalid_zip(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_bytes(b"not a ZIP archive")

    with pytest.raises(ValueError, match="Invalid FSDD ZIP archive"):
        list(_builder()._generate_examples(archive_path, "train"))


def test_fsdd_rejects_archive_without_recordings(tmp_path):
    archive_path = tmp_path / "empty.zip"
    _write_zip(archive_path, {"snapshot/README.md": b"no recordings here"})

    with pytest.raises(ValueError, match="contains no WAV files"):
        list(_builder()._generate_examples(archive_path, "train"))


def test_fsdd_rejects_empty_requested_split(tmp_path):
    archive_path = tmp_path / "test-only.zip"
    _write_zip(
        archive_path,
        {"snapshot/recordings/0_george_0.wav": _wav_bytes(np.array([1], dtype=np.int16))},
    )

    with pytest.raises(ValueError, match="no recordings for the 'train' split"):
        list(_builder()._generate_examples(archive_path, "train"))
