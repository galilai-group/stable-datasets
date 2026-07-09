import io
import json
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import DatasetInfo, Features, Image, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


class CLEVR(BaseDatasetBuilder):
    """CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.

    CLEVR contains 100,000 rendered images of simple 3D objects (cubes, spheres, cylinders)
    with varying colors, materials, sizes, and positions. Each image is paired with ground-truth
    scene metadata (object attributes and spatial relations) and automatically generated
    question-answer pairs that test a range of visual reasoning skills.

    Splits:
        - train: 70,000 images with scene and question annotations
        - val:   15,000 images with scene and question annotations
        - test:  15,000 images with questions only (no answers, no scene data)
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://cs.stanford.edu/people/jcjohns/clevr/",
        "assets": {
            "train": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
            "val": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
            "test": "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
        },
        "citation": """@inproceedings{johnson2017clevr,
            title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
            author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
            pages={2901--2910},
            year={2017}
        }""",
    }

    # Number of images per split, used to iterate image indices.
    _SPLIT_SIZES = {"train": 70000, "val": 15000, "test": 15000}

    def _info(self):
        return DatasetInfo(
            description="""CLEVR is a diagnostic dataset for compositional language and elementary visual
                           reasoning. It contains 100,000 images of 3D-rendered objects (cubes, spheres,
                           cylinders) in varying colors, materials, and sizes, alongside ground-truth scene
                           graphs and question-answer pairs that test counting, comparison, querying, and
                           spatial reasoning.""",
            features=Features(
                {
                    "image": Image(),
                    "image_filename": Value("string"),
                    "image_index": Value("int32"),
                    # JSON-serialised scene dict (keys: objects, relations, directions, etc.)
                    # Empty JSON object '{}' for the test split, which has no scene annotations.
                    "scene_json": Value("string"),
                    # JSON-serialised list of question dicts for this image.
                    # Test-split questions omit 'answer' and 'program' fields.
                    "questions_json": Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Generate examples from the CLEVR_v1.0.zip archive."""
        with zipfile.ZipFile(data_path, "r") as zf:
            # ------------------------------------------------------------------
            # 1. Scene annotations (train / val only; test has none)
            # ------------------------------------------------------------------
            scene_lookup = {}
            if split in ("train", "val"):
                scene_path = f"CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json"
                with zf.open(scene_path) as f:
                    scenes_data = json.load(f)
                for scene in scenes_data["scenes"]:
                    scene_lookup[scene["image_index"]] = scene

            # ------------------------------------------------------------------
            # 2. Question annotations (all splits; test lacks answers/programs)
            # ------------------------------------------------------------------
            question_path = f"CLEVR_v1.0/questions/CLEVR_{split}_questions.json"
            with zf.open(question_path) as f:
                questions_data = json.load(f)

            questions_by_image = {}
            for q in questions_data["questions"]:
                questions_by_image.setdefault(q["image_index"], []).append(q)

            # ------------------------------------------------------------------
            # 3. Yield one example per image
            # ------------------------------------------------------------------
            num_images = self._SPLIT_SIZES[split]
            for image_index in range(num_images):
                image_filename = f"CLEVR_{split}_{image_index:06d}.png"
                image_zip_path = f"CLEVR_v1.0/images/{split}/{image_filename}"

                with zf.open(image_zip_path) as img_file:
                    image = PILImage.open(io.BytesIO(img_file.read())).convert("RGB")

                scene = scene_lookup.get(image_index, {})
                questions = questions_by_image.get(image_index, [])

                yield (
                    image_index,
                    {
                        "image": image,
                        "image_filename": image_filename,
                        "image_index": image_index,
                        "scene_json": json.dumps(scene),
                        "questions_json": json.dumps(questions),
                    },
                )
