import csv
import io
import json

import numpy as np
import pytest

from stable_datasets.schema import ClassLabel, Sequence
from stable_datasets.timeseries import FSD50K
from stable_datasets.timeseries._freesound_vocabularies import FSD50K_VOCABULARY
from stable_datasets.timeseries._zip_utils import SplitZipFile


MODULE = "stable_datasets.timeseries.fsd50k"
NAMES = [name for name, _ in FSD50K_VOCABULARY]
MID = dict(FSD50K_VOCABULARY)

# fname -> (labels, split, n_samples); dev rows carry the official train/val split.
DEV = {
    "64760": (
        ["Electric_guitar", "Guitar", "Plucked_string_instrument", "Musical_instrument", "Music"],
        "train",
        1500,
    ),
    "16399": (["Bark", "Dog", "Domestic_animals_and_pets", "Animal"], "train", 2300),
    "425873": (["Water", "Liquid"], "train", 800),
    "11111": (["Speech", "Human_voice"], "val", 1900),
    "22222": (["Accordion", "Musical_instrument", "Music"], "val", 650),
}
EVAL = {
    "37199": (["Electric_guitar", "Guitar", "Plucked_string_instrument", "Musical_instrument", "Music"], None, 1700),
    "175151": (["Rain", "Water"], None, 1300),
    "253463": (["Accordion", "Musical_instrument", "Music"], None, 900),
}


def _samples(fname: str, n: int) -> np.ndarray:
    return np.random.default_rng(int(fname)).integers(-32768, 32767, size=n, dtype=np.int16)


def _csv(fieldnames, rows, header=True) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def _info(fname):
    return {"title": f"clip {fname}", "license": f"license-{fname}", "uploader": f"user-{fname}"}


@pytest.fixture
def raw_files(tmp_path, wav_bytes, write_zip, write_split_zip):
    """Synthetic FSD50K download: split audio archives + ground truth + metadata, official layout."""

    def build(vocabulary=FSD50K_VOCABULARY, dev=DEV, eval_=EVAL, mids_override=None):
        files = {}
        for stem, rows in (("FSD50K.dev_audio", dev), ("FSD50K.eval_audio", eval_)):
            members = {f"{stem}/{f}.wav": wav_bytes(_samples(f, n)) for f, (*_, n) in rows.items()}
            # Same part counts as the real archives (6 dev, 2 eval).
            n_parts = 6 if stem == "FSD50K.dev_audio" else 2
            parts = write_split_zip(tmp_path, stem, members, n_parts=n_parts)
            if n_parts > 2:
                assert _has_member_straddling_parts(parts)
            names = [f"{stem}.z{i:02d}" for i in range(1, n_parts)] + [f"{stem}.zip"]
            files.update(dict(zip(names, parts)))

        def gt_rows(rows, with_split):
            out = []
            for f, (labels, split, _) in rows.items():
                mids = mids_override or [MID[label] for label in labels]
                row = {"fname": f, "labels": ",".join(labels), "mids": ",".join(mids)}
                if with_split:
                    row["split"] = split
                out.append(row)
            return out

        vocab_csv = _csv(
            ["i", "name", "mid"], [{"i": i, "name": n, "mid": m} for i, (n, m) in enumerate(vocabulary)], header=False
        )
        files["FSD50K.ground_truth.zip"] = write_zip(
            tmp_path / "gt.zip",
            {
                "FSD50K.ground_truth/vocabulary.csv": vocab_csv,
                "FSD50K.ground_truth/dev.csv": _csv(["fname", "labels", "mids", "split"], gt_rows(dev, True)),
                "FSD50K.ground_truth/eval.csv": _csv(["fname", "labels", "mids"], gt_rows(eval_, False)),
            },
        )
        files["FSD50K.metadata.zip"] = write_zip(
            tmp_path / "meta.zip",
            {
                "FSD50K.metadata/dev_clips_info_FSD50K.json": json.dumps({f: _info(f) for f in dev}).encode(),
                "FSD50K.metadata/eval_clips_info_FSD50K.json": json.dumps({f: _info(f) for f in eval_}).encode(),
            },
        )
        return files

    return build


def _has_member_straddling_parts(parts) -> bool:
    with SplitZipFile(parts) as archive:
        sizes = [part.stat().st_size for part in parts]
        return any(info.header_offset + 30 + info.compress_size > sizes[info.disk] for info in archive.infolist())


def _load(tmp_path, split, **kwargs):
    return FSD50K(split=split, processed_cache_dir=str(tmp_path / "processed"), **kwargs)


def test_schema():
    builder = object.__new__(FSD50K)
    FSD50K.__init__(builder)
    features = builder.info.features
    assert isinstance(features["labels"], Sequence) and isinstance(features["labels"].feature, ClassLabel)
    assert features["labels"].feature.names == NAMES
    assert len(NAMES) == 200 and len(set(NAMES)) == 200
    assert builder.info.supervised_keys == ("series", "labels")
    # One asset per archive part, all with md5 checksums.
    assets = FSD50K.SOURCE["assets"]
    assert len(assets) == 6 + 2 + 2
    assert all(info.checksum.startswith("md5:") for info in assets.values())


def test_splits_follow_the_official_ground_truth(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files())
    loaded = {split: _load(tmp_path, split) for split in ("train", "validation", "test")}
    expected = {
        "train": {f: v for f, v in DEV.items() if v[1] == "train"},
        "validation": {f: v for f, v in DEV.items() if v[1] == "val"},
        "test": EVAL,
    }
    seen = set()
    for split, ds in loaded.items():
        rows = list(ds)
        assert [row["freesound_id"] for row in rows] == [int(f) for f in expected[split]]
        names = {row["filename"] for row in rows}
        assert not names & seen, f"{split} overlaps another split"
        seen |= names
        for row in rows:
            fname = row["filename"].removesuffix(".wav")
            labels, _, n = expected[split][fname]
            # Multi-label round trip: class ids decode back to the ground-truth names and MIDs.
            assert [NAMES[i] for i in row["labels"]] == labels
            assert row["mids"] == [MID[label] for label in labels]
            assert row["sample_rate"] == 44100
            assert row["license"] == f"license-{fname}" and row["uploader"] == f"user-{fname}"
            np.testing.assert_array_equal(np.asarray(row["series"]), _samples(fname, n).astype(np.float32)[:, None])


def test_resampled_config(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files())
    row = _load(tmp_path, "test", config_name="22khz")[0]
    assert row["sample_rate"] == 22050
    assert len(row["series"]) == int(np.ceil(EVAL["37199"][2] * 22050 / 44100))


def test_vocabulary_mismatch_is_an_error(tmp_path, raw_files, fake_downloads):
    changed = (("Renamed_class", FSD50K_VOCABULARY[0][1]), *FSD50K_VOCABULARY[1:])
    fake_downloads(MODULE, raw_files(vocabulary=changed))
    with pytest.raises(ValueError, match="vocabulary"):
        _load(tmp_path, "test")


def test_labels_and_mids_must_align(tmp_path, raw_files, fake_downloads):
    fake_downloads(MODULE, raw_files(mids_override=["/m/only_one"]))
    with pytest.raises(ValueError, match="Malformed FSD50K ground truth"):
        _load(tmp_path, "test")
