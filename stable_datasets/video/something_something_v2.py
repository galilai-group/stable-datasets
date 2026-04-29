import csv
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

from loguru import logger as logging

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version, Video
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


_VIDEO_EXTENSIONS = (".webm", ".mp4", ".mkv", ".avi", ".mov")
_EXTRACTION_SENTINEL = ".extraction_complete"


def _normalize_template(value: str | None) -> str:
    # SSv2 class templates mark placeholder objects as "[something]"; labels
    # files omit the brackets, so normalize both forms to the same key.
    if value is None:
        return ""
    return str(value).replace("[", "").replace("]", "").strip()


class SomethingSomethingV2(BaseDatasetBuilder):
    """Something-Something V2 action-recognition videos.

    The Qualcomm package is large. Set ``STABLE_DATASETS_CACHE_DIR`` or pass
    ``download_dir=`` / ``processed_cache_dir=`` before loading it on machines
    with small home quotas.
    """

    VERSION = Version("2.0.0")

    SOURCE = DatasetSource(
        homepage="https://developer.qualcomm.com/software/ai-datasets/something-something",
        assets={
            "video_part_00": DownloadInfo(
                url="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00",
                filename="20bn-something-something-v2-00",
            ),
            "video_part_01": DownloadInfo(
                url="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01",
                filename="20bn-something-something-v2-01",
            ),
            "labels": DownloadInfo(
                url="https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip",
                filename="20bn-something-something-download-package-labels.zip",
            ),
        },
        license="Qualcomm data license agreement for research use. Users must obtain and use the dataset under Qualcomm's terms.",
        citation="""@inproceedings{goyal2017something,
  title={The "Something Something" Video Database for Learning and Evaluating Visual Common Sense},
  author={Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={5842--5850},
  year={2017}
}""",
    )

    def __init__(self, config_name: str | None = None, data_dir: str | Path | None = None, **kwargs):
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        self._video_index_cache: dict[str, dict[str, Path]] = {}
        super().__init__(config_name=config_name, **kwargs)

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description=(
                "Something-Something V2 contains short crowd-sourced videos of "
                "humans performing fine-grained actions with everyday objects."
            ),
            features=Features(
                {
                    "video": Video(storage="path", allowed_extensions=_VIDEO_EXTENSIONS),
                    "video_id": Value("string"),
                    "video_filename": Value("string"),
                    "label": Value("int32"),
                    "text": Value("string"),
                    "template": Value("string"),
                    "placeholders_json": Value("string"),
                    "split": Value("string"),
                }
            ),
            supervised_keys=("video", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _candidate_splits(self) -> list:
        return [Split.TRAIN, Split.VALIDATION, Split.TEST]

    def _split_generators(self) -> list[SplitGenerator]:
        work_dir = Path(getattr(self, "_raw_download_dir", _default_dest_folder()))
        work_dir.mkdir(parents=True, exist_ok=True)

        if self.data_dir is None:
            source = self._source()
            assets = source["assets"]
            downloaded = bulk_download(
                [assets["video_part_00"], assets["video_part_01"], assets["labels"]],
                dest_folder=work_dir,
            )
            video_parts = downloaded[:2]
            labels_source = downloaded[2]
        else:
            video_parts, labels_source = self._resolve_local_inputs(self.data_dir)

        labels_dir = self._ensure_labels_dir(labels_source, work_dir)
        videos_dir = self._ensure_videos_dir(video_parts, work_dir)
        label_map = self._load_label_map(labels_dir)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "annotation_file": self._find_split_json(labels_dir, "train"),
                    "videos_dir": videos_dir,
                    "split": Split.TRAIN,
                    "label_map": label_map,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": self._find_split_json(labels_dir, "validation"),
                    "videos_dir": videos_dir,
                    "split": Split.VALIDATION,
                    "label_map": label_map,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "annotation_file": self._find_split_json(labels_dir, "test"),
                    "videos_dir": videos_dir,
                    "split": Split.TEST,
                    "label_map": label_map,
                    "test_answers_file": self._find_optional_file(
                        labels_dir,
                        ("test-answers.csv", "test_answers.csv"),
                    ),
                },
            ),
        ]

    def _generate_examples(
        self,
        annotation_file: Path,
        videos_dir: Path,
        split: str,
        label_map: dict[str, int],
        test_answers_file: Path | None = None,
    ) -> Iterator[tuple[str, dict]]:
        annotations = self._load_annotations(annotation_file)
        test_templates = self._load_test_answers(test_answers_file)
        video_index = self._video_index(videos_dir)

        missing = 0
        missing_examples = []
        for annotation in annotations:
            video_id = str(annotation["id"])
            video_path = video_index.get(video_id)
            if video_path is None:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(video_id)
                continue

            template = _normalize_template(annotation.get("template"))
            if not template and video_id in test_templates:
                template = _normalize_template(test_templates[video_id])

            label = int(label_map.get(template, -1)) if template else -1
            text = str(annotation.get("label") or template)
            placeholders = annotation.get("placeholders") or []

            yield (
                video_id,
                {
                    "video": str(video_path),
                    "video_id": video_id,
                    "video_filename": video_path.name,
                    "label": label,
                    "text": text,
                    "template": template,
                    "placeholders_json": json.dumps(placeholders),
                    "split": split,
                },
            )
        if missing:
            logging.warning(
                f"SSv2 {split}: skipped {missing}/{len(annotations)} annotations "
                f"because matching video files were missing. First missing ids: {missing_examples}"
            )

    def _resolve_local_inputs(self, data_dir: Path) -> tuple[list[Path], Path]:
        if not data_dir.exists():
            raise FileNotFoundError(f"SSv2 data_dir does not exist: {data_dir}")

        labels_dir = data_dir / "labels"
        if labels_dir.exists():
            labels_source = labels_dir
        else:
            label_zips = sorted(data_dir.glob("*labels*.zip"))
            if not label_zips:
                raise FileNotFoundError(f"Could not find labels/ or a *labels*.zip file under {data_dir}")
            labels_source = label_zips[0]

        videos_dir = data_dir / "videos"
        if videos_dir.exists():
            return [videos_dir], labels_source

        local_videos = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS]
        if local_videos:
            return [data_dir], labels_source

        archive_candidates = sorted(
            p for p in data_dir.iterdir() if p.is_file() and p.name.startswith("20bn-something-something-v2")
        )
        if not archive_candidates:
            raise FileNotFoundError(
                f"Could not find videos/, video files, or 20bn-something-something-v2* archives under {data_dir}"
            )
        return archive_candidates, labels_source

    def _ensure_labels_dir(self, labels_source: Path, work_dir: Path) -> Path:
        labels_source = Path(labels_source)
        if labels_source.is_dir():
            return labels_source
        labels_dir = work_dir / "ssv2_labels"
        if self._is_extraction_complete(labels_dir):
            return labels_dir
        if not zipfile.is_zipfile(labels_source):
            raise ValueError(f"SSv2 labels package is not a zip file: {labels_source}")
        self._extract_archives_to_complete_dir([labels_source], labels_dir)
        return labels_dir

    def _ensure_videos_dir(self, video_sources: list[Path], work_dir: Path) -> Path:
        video_sources = [Path(path) for path in video_sources]
        if len(video_sources) == 1 and video_sources[0].is_dir():
            return video_sources[0]

        videos_dir = work_dir / "ssv2_videos"
        if self._is_extraction_complete(videos_dir):
            return videos_dir

        if all(zipfile.is_zipfile(path) for path in video_sources):
            self._extract_archives_to_complete_dir(video_sources, videos_dir)
            return videos_dir

        if len(video_sources) == 1:
            self._extract_archives_to_complete_dir(video_sources, videos_dir)
            return videos_dir

        concatenated = self._concatenate_parts(video_sources, work_dir)
        self._extract_archives_to_complete_dir([concatenated], videos_dir)
        return videos_dir

    def _extract_archive(self, archive: Path, dest: Path) -> None:
        if tarfile.is_tarfile(archive):
            self._extract_tar(archive, dest)
            return
        if zipfile.is_zipfile(archive):
            self._safe_zip_extract(archive, dest)
            return
        raise ValueError(f"Unsupported SSv2 video archive format: {archive}")

    def _extract_tar(self, archive: Path, dest: Path) -> None:
        with tarfile.open(archive, "r:*") as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                self._safe_tar_extract_legacy(tf, dest)

    def _safe_tar_extract_legacy(self, tf: tarfile.TarFile, dest: Path) -> None:
        dest = Path(dest).resolve()
        for member in tf.getmembers():
            target = self._safe_archive_target(dest, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe member in tar archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            src = tf.extractfile(member)
            if src is None:
                raise ValueError(f"Could not read tar member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)

    def _safe_zip_extract(self, archive: Path, dest: Path) -> None:
        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                target = self._safe_archive_target(dest, info.filename)
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"Unsafe member in zip archive: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)

    def _safe_archive_target(self, dest: Path, name: str) -> Path:
        target = (dest / name).resolve()
        if os.path.commonpath([str(dest), str(target)]) != str(dest):
            raise ValueError(f"Unsafe path in archive: {name}")
        return target

    def _extract_archives_to_complete_dir(self, archives: list[Path], dest: Path) -> None:
        dest = Path(dest)
        if self._is_extraction_complete(dest):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}_tmp_"))
        try:
            for archive in archives:
                self._extract_archive(Path(archive), tmp_dir)
            (tmp_dir / _EXTRACTION_SENTINEL).write_text("ok\n")
            if dest.exists():
                shutil.rmtree(dest)
            os.rename(tmp_dir, dest)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _is_extraction_complete(self, dest: Path) -> bool:
        return (Path(dest) / _EXTRACTION_SENTINEL).is_file()

    def _concatenate_parts(self, video_sources: list[Path], work_dir: Path) -> Path:
        parts = sorted((Path(path) for path in video_sources), key=lambda p: p.name)
        concatenated = Path(work_dir) / "20bn-something-something-v2.concatenated"
        manifest_path = concatenated.with_suffix(concatenated.suffix + ".json")
        manifest = {
            "parts": [
                {
                    "name": part.name,
                    "size": part.stat().st_size,
                    "mtime_ns": part.stat().st_mtime_ns,
                }
                for part in parts
            ]
        }
        if concatenated.exists() and manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                existing = None
            if existing == manifest:
                return concatenated

        fd, tmp_name = tempfile.mkstemp(
            dir=work_dir,
            prefix=f".{concatenated.name}_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                for part in parts:
                    with open(part, "rb") as f:
                        shutil.copyfileobj(f, out)
            os.replace(tmp_path, concatenated)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            tmp_path.unlink(missing_ok=True)
            raise
        return concatenated

    def _find_split_json(self, labels_dir: Path, split: str) -> Path:
        aliases = {
            "train": ("train.json", "something-something-v2-train.json"),
            "validation": ("validation.json", "val.json", "something-something-v2-validation.json"),
            "test": ("test.json", "something-something-v2-test.json"),
        }[split]
        found = self._find_optional_file(labels_dir, aliases)
        if found is None:
            raise FileNotFoundError(f"Could not find SSv2 {split} annotation JSON under {labels_dir}")
        return found

    def _find_optional_file(self, root: Path, names: tuple[str, ...]) -> Path | None:
        for name in names:
            direct = root / name
            if direct.exists():
                return direct
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
        return None

    def _load_label_map(self, labels_dir: Path) -> dict[str, int]:
        labels_file = self._find_optional_file(
            labels_dir,
            ("labels.json", "something-something-v2-labels.json"),
        )
        if labels_file is None:
            return {}
        raw = json.loads(labels_file.read_text())
        if isinstance(raw, dict):
            return {_normalize_template(name): int(idx) for name, idx in raw.items()}
        raise ValueError(f"Unsupported SSv2 labels file format: {labels_file}")

    def _load_annotations(self, path: Path) -> list[dict]:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, list):
            raise ValueError(f"SSv2 annotation file must contain a list: {path}")
        return raw

    def _load_test_answers(self, path: Path | None) -> dict[str, str]:
        if path is None:
            return {}
        answers = {}
        with open(path, newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=";,")
                reader = csv.reader(f, dialect)
            except csv.Error:
                reader = csv.reader(f, delimiter=";" if ";" in sample else ",")
            for row in reader:
                if len(row) >= 2:
                    answers[str(row[0])] = row[1]
        return answers

    def _video_index(self, videos_dir: Path) -> dict[str, Path]:
        key = str(Path(videos_dir).resolve())
        cached = self._video_index_cache.get(key)
        if cached is not None:
            return cached
        index = {}
        for path in self._iter_video_files(Path(videos_dir)):
            index[path.stem] = path
        self._video_index_cache[key] = index
        return index

    def _iter_video_files(self, root: Path) -> Iterator[Path]:
        stack = [Path(root)]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            path = Path(entry.path)
                            if path.suffix.lower() in _VIDEO_EXTENSIONS:
                                yield path
            except OSError as exc:
                logging.warning(f"Could not scan SSv2 video directory {current}: {exc}")


SSv2 = SomethingSomethingV2
