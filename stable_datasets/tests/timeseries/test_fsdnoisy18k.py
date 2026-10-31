import csv
import io

import numpy as np
import pytest

from stable_datasets.schema import ClassLabel, Sequence, Value
from stable_datasets.timeseries import FSDnoisy18k
from stable_datasets.timeseries._freesound_vocabularies import FSDNOISY18K_VOCABULARY


MODULE = "stable_datasets.timeseries.fsdnoisy18k"
CLASSES = [name for name, _ in FSDNOISY18K_VOCABULARY]

# fname -> (label, manually_verified, noisy_small, n_samples)
TRAIN = {
    "101.wav": ("Rain", "1", "0", 900),
    "202.wav": ("Wind", "0", "1", 1200),
    "303.wav": ("Acoustic_guitar", "0", "0", 441),
}
TEST = {
    "404.wav": ("Writing", None, None, 700),
    "505.wav": ("Fire", None, None, 333),
}
LICENSES = {
    "101.wav": "http://creativecommons.org/publicdomain/zero/1.0/",
    "202.wav": "http://creativecommons.org/licenses/by/3.0/",
    "303.wav": "http://creativecommons.org/licenses/by-nc/3.0/",
    "404.wav": "http://creativecommons.org/licenses/by/3.0/",
    "505.wav": "http://creativecommons.org/licenses/sampling+/1.0/",
}


def _samples(fname: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(fname.removesuffix(".wav")))
    return rng.integers(-32768, 32767, size=n, dtype=np.int16)


def _csv(fieldnames, rows) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


@pytest.fixture
def raw_files(tmp_path, wav_bytes, write_zip):
    """Synthetic FSDnoisy18k download with the official archive layout."""

    def build(train=TRAIN, test=TEST, drop_audio=()):
        audio = {
            "FSDnoisy18k.audio_train.zip": {
                f"FSDnoisy18k.audio_train/{f}": wav_bytes(_samples(f, n)) for f, (*_, n) in train.items()
            },
            "FSDnoisy18k.audio_test.zip": {
                f"FSDnoisy18k.audio_test/{f}": wav_bytes(_samples(f, n)) for f, (*_, n) in test.items()
            },
        }
        for archive in audio.values():
            for member in [m for m in archive if m.rsplit("/", 1)[-1] in drop_audio]:
                del archive[member]
        train_rows = [
            {"fname": f, "label": label, "aso_id": "/m/x", "manually_verified": mv, "noisy_small": ns}
            for f, (label, mv, ns, _) in train.items()
        ]
        test_rows = [{"fname": f, "label": label, "aso_id": "/m/x"} for f, (label, *_) in test.items()]
        meta = {
            "FSDnoisy18k.meta/train.csv": _csv(
                ["fname", "label", "aso_id", "manually_verified", "noisy_small"], train_rows
            ),
            "FSDnoisy18k.meta/test.csv": _csv(["fname", "label", "aso_id"], test_rows),
        }
        doc = {
            "FSDnoisy18k.doc/LICENSE-INDIVIDUAL-CLIPS.csv": _csv(
                ["fname", "license"], [{"fname": f, "license": lic} for f, lic in LICENSES.items()]
            )
        }
        files = {name: write_zip(tmp_path / name, members) for name, members in audio.items()}
        files["FSDnoisy18k.meta.zip"] = write_zip(tmp_path / "FSDnoisy18k.meta.zip", meta)
        files["FSDnoisy18k.doc.zip"] = write_zip(tmp_path / "FSDnoisy18k.doc.zip", doc)
        return files

    return build


def _load(tmp_path, split, **kwargs):
    return FSDnoisy18k(split=split, processed_cache_dir=str(tmp_path / "processed"), **kwargs)


def test_schema():
    builder = object.__new__(FSDnoisy18k)
    FSDnoisy18k.__init__(builder)
    features = builder.info.features
    series = features["series"]  # [T, C] like the other audio builders
    assert isinstance(series, Sequence) and isinstance(series.feature, Sequence)
    assert isinstance(series.feature.feature, Value) and series.feature.feature.dtype == "float32"
    assert isinstance(features["label"], ClassLabel)
    assert features["label"].names == CLASSES
    assert len(CLASSES) == 20
    assert builder.info.supervised_keys == ("series", "label")
    assert [config.name for config in FSDnoisy18k.BUILDER_CONFIGS] == ["native", "22khz", "16khz"]


def test_train_and_test_rows_match_the_metadata(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files())
    train, test = _load(tmp_path, "train"), _load(tmp_path, "test")
    assert (len(train), len(test)) == (len(TRAIN), len(TEST))

    for split_ds, expected in ((train, TRAIN), (test, TEST)):
        assert [row["filename"] for row in split_ds] == list(expected)  # metadata order
        for row in split_ds:
            label, mv, ns, n = expected[row["filename"]]
            assert CLASSES[row["label"]] == label
            assert row["sample_rate"] == 44100
            assert row["freesound_id"] == int(row["filename"].removesuffix(".wav"))
            assert row["license"] == LICENSES[row["filename"]]
            # Lossless: raw int16 PCM values, stored as a float32 [T, 1] series.
            np.testing.assert_array_equal(
                np.asarray(row["series"]), _samples(row["filename"], n).astype(np.float32)[:, None]
            )
            if expected is TRAIN:
                assert row["manually_verified"] == (mv == "1")
                assert row["noisy_small"] == (ns == "1")
            else:  # the test set is drawn entirely from the clean (manually verified) portion
                assert row["manually_verified"] is True
                assert row["noisy_small"] is False

    assert not {row["filename"] for row in train} & {row["filename"] for row in test}


def test_resampled_config_has_its_own_cache(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files())
    native = _load(tmp_path, "train")
    resampled = _load(tmp_path, "train", config_name="16khz")
    for a, b in zip(native, resampled):
        assert b["sample_rate"] == 16000
        assert len(b["series"]) == int(np.ceil(len(a["series"]) * 16000 / 44100))
        assert a["filename"] == b["filename"] and a["label"] == b["label"]
    assert len(native[0]["series"]) == 900  # the native cache was not overwritten


def test_cache_reopens_without_downloading(tmp_path, raw_files, fake_downloads, monkeypatch):
    fake_downloads(MODULE, raw_files())
    first = _load(tmp_path, "test")[0]

    def _no_download(*args, **kwargs):
        raise AssertionError("bulk_download must not be called on a cache hit")

    monkeypatch.setattr(f"{MODULE}.bulk_download", _no_download)
    again = _load(tmp_path, "test")[0]
    assert again["filename"] == first["filename"]
    np.testing.assert_array_equal(again["series"], first["series"])


def test_missing_audio_is_an_error(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files(drop_audio={"202.wav"}))
    with pytest.raises(ValueError, match="missing from the archive"):
        _load(tmp_path, "train")


def test_unknown_label_is_an_error(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files(train={"101.wav": ("Not_a_class", "1", "0", 100)}))
    with pytest.raises(ValueError, match="Unknown FSDnoisy18k label"):
        _load(tmp_path, "train")
