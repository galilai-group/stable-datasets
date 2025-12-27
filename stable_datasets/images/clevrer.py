import json
import os
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import datasets

from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder


def _wget_download(url: str, dest_folder: Path) -> Path:
    """Download a file using wget with resume support.
    """
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    local_path = dest_folder / filename

    cmd = ["wget", "-c", "--progress=bar:force:noscroll", "-P", str(dest_folder), url]

    print(f"Downloading (or resuming): {url}")
    subprocess.run(cmd, check=True, cwd=str(dest_folder))

    return local_path


class CLEVRER(BaseDatasetBuilder):
    """CLEVRER: CoLlision Events for Video REpresentation and Reasoning.

    A diagnostic video dataset for systematic evaluation of computational models
    on a wide range of reasoning tasks. The dataset includes four types of questions:
    descriptive (e.g., "what color"), explanatory ("what's responsible for"),
    predictive ("what will happen next"), and counterfactual ("what if").

    The dataset contains 20,000 synthetic videos of moving and colliding objects.
    Each video is 5 seconds long and contains 128 frames with resolution 480 x 320.

    Splits:
        - train: 10,000 videos (index 0 - 9999)
        - validation: 5,000 videos (index 10000 - 14999)
        - test: 5,000 videos (index 15000 - 19999)
    """

    VERSION = datasets.Version("1.0.0")

    SOURCE = {
        "homepage": "http://clevrer.csail.mit.edu/",
        "assets": {
            "train_videos": "http://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
            "train_annotations": "http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip",
            "train_questions": "http://data.csail.mit.edu/clevrer/questions/train.json",
            "validation_videos": "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
            "validation_annotations": "http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip",
            "validation_questions": "http://data.csail.mit.edu/clevrer/questions/validation.json",
            "test_videos": "http://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
            "test_questions": "http://data.csail.mit.edu/clevrer/questions/test.json",
        },
        "citation": """@inproceedings{yi2020clevrer,
            title={CLEVRER: CoLlision Events for Video REpresentation and Reasoning},
            author={Yi, Kexin and Gan, Chuang and Li, Yunzhu and Kohli, Pushmeet and Wu, Jiajun and Torralba, Antonio and Tenenbaum, Joshua B},
            booktitle={International Conference on Learning Representations},
            year={2020}
        }""",
    }

    def _info(self):
        return datasets.DatasetInfo(
            description="""CLEVRER is a diagnostic video dataset for temporal and causal reasoning.
            It contains 20,000 synthetic videos of moving and colliding objects, with four types
            of questions: descriptive, explanatory, predictive, and counterfactual.""",
            features=datasets.Features(
                {
                    "video": datasets.Video(),
                    "scene_index": datasets.Value("int32"),
                    "video_filename": datasets.Value("string"),
                    # Store questions as JSON string to avoid nested Sequence issues
                    "questions_json": datasets.Value("string"),
                    # Store annotations as JSON string
                    "annotations_json": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            license="CC0",
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self, dl_manager):
        source = self._source()
        assets = source["assets"]

        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        download_dir = Path(download_dir)

        # Download all files sequentially using wget (handles redirects, supports resume)
        url_to_path = {}
        for key, url in assets.values():
            url_to_path[url] = _wget_download(url, download_dir)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "videos_path": url_to_path[assets["train_videos"]],
                    "annotations_path": url_to_path[assets["train_annotations"]],
                    "questions_path": url_to_path[assets["train_questions"]],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "videos_path": url_to_path[assets["validation_videos"]],
                    "annotations_path": url_to_path[assets["validation_annotations"]],
                    "questions_path": url_to_path[assets["validation_questions"]],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "videos_path": url_to_path[assets["test_videos"]],
                    "annotations_path": None,  # Test split has no annotations
                    "questions_path": url_to_path[assets["test_questions"]],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, videos_path, annotations_path, questions_path, split):
        # Load questions
        with open(questions_path) as f:
            questions_data = json.load(f)

        # Create a mapping from scene_index to questions
        scene_to_questions = {item["scene_index"]: item for item in questions_data}

        # Load annotations if available (not available for test split)
        scene_to_annotations = {}
        if annotations_path is not None:
            with zipfile.ZipFile(annotations_path, "r") as ann_zip:
                for filename in ann_zip.namelist():
                    if filename.endswith(".json"):
                        with ann_zip.open(filename) as f:
                            ann_data = json.load(f)
                            scene_to_annotations[ann_data["scene_index"]] = ann_data

        # Extract videos directory
        extract_dir = Path(videos_path).parent / f"clevrer_{split}_videos"
        if not extract_dir.exists():
            with zipfile.ZipFile(videos_path, "r") as vid_zip:
                vid_zip.extractall(extract_dir)

        # Generate examples
        for scene_index, question_item in scene_to_questions.items():
            video_filename = question_item["video_filename"]

            # Find the video file
            video_path = self._find_video_file(extract_dir, video_filename)
            if video_path is None:
                continue

            # Get questions from the question file
            questions = question_item.get("questions", [])

            # Get annotations if available
            annotations = {}
            if scene_index in scene_to_annotations:
                annotations = scene_to_annotations[scene_index]

            yield (
                scene_index,
                {
                    "video": str(video_path),
                    "scene_index": scene_index,
                    "video_filename": video_filename,
                    "questions_json": json.dumps(questions),
                    "annotations_json": json.dumps(annotations),
                },
            )

    def _find_video_file(self, extract_dir, video_filename):
        """Find a video file in the extracted directory structure."""
        # Videos are arranged in folders per 1000 files
        for root, dirs, files in os.walk(extract_dir):
            if video_filename in files:
                return Path(root) / video_filename
        return None
