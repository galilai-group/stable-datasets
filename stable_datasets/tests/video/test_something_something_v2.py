import io
import json
import tarfile
import zipfile

import pytest

from stable_datasets.video.something_something_v2 import _EXTRACTION_SENTINEL
from stable_datasets.schema import VideoRef
from stable_datasets.video import SSv2, SomethingSomethingV2


def _builder():
    builder = object.__new__(SomethingSomethingV2)
    builder._video_index_cache = {}
    return builder


def _write_local_ssv2(root):
    labels = root / "labels"
    videos = root / "videos"
    labels.mkdir(parents=True)
    videos.mkdir(parents=True)

    (labels / "labels.json").write_text(
        json.dumps(
            {
                "Holding something": "16",
                "Putting something on a surface": "109",
            }
        )
    )
    (labels / "train.json").write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "label": "holding cup",
                    "template": "Holding [something]",
                    "placeholders": ["cup"],
                }
            ]
        )
    )
    (labels / "validation.json").write_text(
        json.dumps(
            [
                {
                    "id": "2",
                    "label": "putting cup on a table",
                    "template": "Putting [something] on a surface",
                    "placeholders": ["cup"],
                }
            ]
        )
    )
    (labels / "test.json").write_text(json.dumps([{"id": "3"}]))
    (labels / "test-answers.csv").write_text("3;Holding something\n")

    (videos / "1.webm").write_bytes(b"train video")
    (videos / "2.webm").write_bytes(b"validation video")
    (videos / "3.webm").write_bytes(b"test video")


def test_ssv2_local_data_dir_builds_video_refs(tmp_path):
    data_dir = tmp_path / "raw"
    _write_local_ssv2(data_dir)

    ds = SomethingSomethingV2(
        split="train",
        data_dir=data_dir,
        processed_cache_dir=tmp_path / "processed",
        download_dir=tmp_path / "downloads",
    )

    assert len(ds) == 1
    row = ds[0]
    assert isinstance(row["video"], VideoRef)
    assert row["video"].extension == ".webm"
    assert row["video"].bytes == b"train video"
    assert row["video_id"] == "1"
    assert row["video_filename"] == "1.webm"
    assert row["label"] == 16
    assert row["text"] == "holding cup"
    assert row["template"] == "Holding something"
    assert json.loads(row["placeholders_json"]) == ["cup"]
    assert row["split"] == "train"


def test_ssv2_alias_and_test_answers(tmp_path):
    data_dir = tmp_path / "raw"
    _write_local_ssv2(data_dir)

    ds = SSv2(
        split="test",
        data_dir=data_dir,
        processed_cache_dir=tmp_path / "processed",
        download_dir=tmp_path / "downloads",
    )

    row = ds[0]
    assert row["video_id"] == "3"
    assert row["label"] == 16
    assert row["text"] == "Holding something"
    assert row["template"] == "Holding something"


def test_ssv2_zip_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.json", "{}")

    with pytest.raises(ValueError, match="Unsafe path"):
        _builder()._safe_zip_extract(archive, tmp_path / "dest")


def test_ssv2_tar_extraction_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar"
    payload = b"oops"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises((ValueError, tarfile.TarError)):
        _builder()._extract_tar(archive, tmp_path / "dest")


def test_ssv2_label_extraction_uses_completion_sentinel(tmp_path):
    archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("labels.json", json.dumps({"Holding something": "16"}))
        zf.writestr("train.json", "[]")

    partial = tmp_path / "ssv2_labels"
    partial.mkdir()
    (partial / "train.json").write_text("partial")

    labels_dir = _builder()._ensure_labels_dir(archive, tmp_path)

    assert labels_dir == partial
    assert (labels_dir / _EXTRACTION_SENTINEL).exists()
    assert json.loads((labels_dir / "labels.json").read_text()) == {"Holding something": "16"}


def test_ssv2_test_answer_csv_fallback_for_small_files(tmp_path):
    answers = tmp_path / "test-answers.csv"
    answers.write_text("3;Holding something\n")

    assert _builder()._load_test_answers(answers) == {"3": "Holding something"}
