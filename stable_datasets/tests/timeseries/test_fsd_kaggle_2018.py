import csv
import io
import wave
import zipfile

import numpy as np

from stable_datasets.timeseries import FSDKaggle2018


def _wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


def _write_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_fsdkaggle2018_builder_smoke(tmp_path, monkeypatch):
    train_audio = tmp_path / "train.zip"
    test_audio = tmp_path / "test.zip"
    meta = tmp_path / "meta.zip"

    _write_zip(
        train_audio,
        {
            "audio_train/train_1.wav": _wav_bytes(np.array([0, 1000, -1000, 500], dtype=np.int16)),
        },
    )
    _write_zip(
        test_audio,
        {
            "audio_test/test_1.wav": _wav_bytes(np.array([0, -500, 500], dtype=np.int16)),
        },
    )

    train_csv = io.StringIO()
    train_writer = csv.DictWriter(train_csv, fieldnames=["fname", "label", "manually_verified", "fsID"])
    train_writer.writeheader()
    train_writer.writerow(
        {"fname": "train_1.wav", "label": "Acoustic_guitar", "manually_verified": "1", "fsID": "101"}
    )

    test_csv = io.StringIO()
    test_writer = csv.DictWriter(test_csv, fieldnames=["fname", "label", "usage", "fsID"])
    test_writer.writeheader()
    test_writer.writerow({"fname": "test_1.wav", "label": "Bark", "usage": "Public", "fsID": "202"})

    _write_zip(
        meta,
        {
            "FSDKaggle2018.meta/train_post_competition.csv": train_csv.getvalue().encode("utf-8"),
            "FSDKaggle2018.meta/test_post_competition_scoring_clips.csv": test_csv.getvalue().encode("utf-8"),
        },
    )

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        return [train_audio, test_audio, meta]

    monkeypatch.setattr("stable_datasets.timeseries.fsd_kaggle_2018.bulk_download", _fake_bulk_download)

    train_ds = FSDKaggle2018(split="train", processed_cache_dir=str(tmp_path / "processed"))
    train_sample = train_ds[0]
    assert set(train_sample) == {"series", "label", "filename", "fsid", "verified", "usage"}
    assert train_sample["filename"] == "train_1.wav"
    assert train_sample["fsid"] == "101"
    assert train_sample["verified"] == "1"
    assert train_sample["usage"] == ""
    assert len(train_sample["series"]) == 4
    assert len(train_sample["series"][0]) == 1

    test_ds = FSDKaggle2018(split="test", processed_cache_dir=str(tmp_path / "processed"))
    test_sample = test_ds[0]
    assert test_sample["filename"] == "test_1.wav"
    assert test_sample["fsid"] == "202"
    assert test_sample["verified"] == ""
    assert test_sample["usage"] == "Public"
    assert len(test_sample["series"]) == 3
    assert len(test_sample["series"][0]) == 1
