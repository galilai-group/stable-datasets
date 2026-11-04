from io import BytesIO

import pyarrow.parquet as pq
from PIL import Image as PILImage
from tqdm import tqdm

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder



# The original TU-Berlin download is not reliably available, so this builder uses a public Hugging Face re-upload of
# the same 20,000 sketches (sdiaeyu6n/tu-berlin). The URLs are pinned to a commit so the files cannot change.
_REVISION = "e7c6c0dedcf2aa4ca060878893bca8d77ef699f4"
_BASE_URL = f"https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/resolve/{_REVISION}/data"


class TUBerlin(BaseDatasetBuilder):
    """TU-Berlin Sketch Dataset

    The TU-Berlin dataset contains 20,000 human-drawn sketches, evenly distributed over 250 object categories
    (80 sketches per category). Sketches were collected via crowdsourcing by asking participants to draw an object
    from a given category.

    The original release does not define train/validation/test splits. This builder uses the splits of the Hugging Face
    re-upload ``sdiaeyu6n/tu-berlin``: a random (not class-stratified) 80% / 10% / 10% split made with
    ``sklearn.model_selection.train_test_split(random_state=42)``, giving 16,000 / 2,000 / 2,000 sketches. Class
    counts per split therefore vary slightly. Labels are integers 0-249, indexing the categories in alphabetical order.
    Images are converted to single-channel grayscale (mode "L").
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/",
        assets={
            "train": DownloadInfo(url=f"{_BASE_URL}/train-00000-of-00001.parquet"),
            "validation": DownloadInfo(url=f"{_BASE_URL}/validation-00000-of-00001.parquet"),
            "test": DownloadInfo(url=f"{_BASE_URL}/test-00000-of-00001.parquet"),
        },
        citation="""@article{eitz2012hdhso,
                    author  = {Eitz, Mathias and Hays, James and Alexa, Marc},
                    title   = {How Do Humans Sketch Objects?},
                    journal = {ACM Trans. Graph. (Proc. SIGGRAPH)},
                    year    = {2012},
                    volume  = {31},
                    number  = {4},
                    pages   = {44:1--44:10}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""The TU-Berlin sketch dataset contains 20,000 human-drawn sketches
                           across 250 object categories (80 per category).""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(num_classes=250),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE.get("license", None),
        )

    def _generate_examples(self, data_path, split):
        parquet_file = pq.ParquetFile(data_path)
        progress = tqdm(total=parquet_file.metadata.num_rows, desc=f"Processing {split} split")

        idx = 0
        # Stream in small batches: the train file is ~470 MB, so avoid loading it all at once.
        for batch in parquet_file.iter_batches(batch_size=256, columns=["image", "label"]):
            for row in batch.to_pylist():
                image = self._to_grayscale(PILImage.open(BytesIO(row["image"]["bytes"])))
                yield idx, {"image": image, "label": int(row["label"])}
                idx += 1
                progress.update(1)
        progress.close()

    @staticmethod
    def _to_grayscale(image):
        """Convert to mode "L", compositing any transparency onto a white background first."""
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = PILImage.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image)
            image = background
        return image.convert("L")